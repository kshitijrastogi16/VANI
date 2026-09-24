"""
VANI V3 LLM Client - FIXED VERSION  
Enforcing: Proper error handling, retries, timeouts, resource management
"""

import os
import time
import logging
import asyncio
from typing import List, Dict, Optional, Generator, Any
from dataclasses import dataclass
from datetime import datetime
import json

try:
    from groq import Groq
    import httpx
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logging.warning("⚠️ Groq library not installed. Run: pip install groq")

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    id: str
    name: str
    type: str
    max_tokens: int
    best_for: str
    timeout: int = 30
    priority: int = 1

@dataclass
class GenerationStats:
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    avg_response_time: float
    model_usage: Dict[str, int]

class GroqClient:
    """
    Production-Grade Multi-Model Groq Client with Comprehensive Error Handling
    Features: Retry logic, circuit breaker, resource management, performance tracking
    """
    
    def __init__(self, 
                 api_key: str, 
                 models: List[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 1024,
                 timeout: int = None,
                 max_retries: int = None,
                 retry_delay: float = 1.0):
        from app.settings import settings as app_settings
        
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library not available. Install with: pip install groq")
        
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY environment variable.")
            
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout or app_settings.llm_timeout
        self.max_retries = max_retries or app_settings.llm_max_retries
        self.retry_delay = retry_delay
        

        # Model configurations with priority ordering.
        #
        # Groq retired the previous Llama 3.1/3.3 IDs used by VANI.
        # Keep model IDs in this single catalog and allow GROQ_MODELS to
        # override the defaults at runtime.
        model_catalog = {
            "openai/gpt-oss-120b": ModelConfig(
                id="openai/gpt-oss-120b",
                name="GPT-OSS 120B",
                type="production",
                max_tokens=32768,
                best_for="General production queries",
                priority=1,
            ),
            "qwen/qwen3.8-27b": ModelConfig(
                id="qwen/qwen3.8-27b",
                name="Qwen 3.8 27B",
                type="complex",
                max_tokens=32768,
                best_for="Complex narrative understanding",
                priority=2,
            ),
            "openai/gpt-oss-20b": ModelConfig(
                id="openai/gpt-oss-20b",
                name="GPT-OSS 20B",
                type="speed",
                max_tokens=32768,
                best_for="Fast responses",
                priority=3,
            ),
        }

        configured_models = models if models is not None else app_settings.groq_models
        configured_models = [m.strip() for m in configured_models if isinstance(m, str) and m.strip()]

        # Preserve custom model support: if a caller supplies a model ID that
        # is not in the catalog, build a conservative generic configuration.
        self.model_configs = []
        for index, model_id in enumerate(configured_models, start=1):
            template = model_catalog.get(model_id)
            if template:
                self.model_configs.append(
                    ModelConfig(
                        id=template.id,
                        name=template.name,
                        type=template.type,
                        max_tokens=template.max_tokens,
                        best_for=template.best_for,
                        timeout=template.timeout,
                        priority=template.priority,
                    )
                )
            else:
                self.model_configs.append(
                    ModelConfig(
                        id=model_id,
                        name=model_id,
                        type="production",
                        max_tokens=32768,
                        best_for="Configured custom Groq model",
                        priority=index,
                    )
                )
        # Optional runtime denylist for decommissioned/unsupported models
        denylist = set((os.getenv('GROQ_SKIP_MODELS') or '').split(',')) if os.getenv('GROQ_SKIP_MODELS') else set()
        if denylist:
            self.model_configs = [cfg for cfg in self.model_configs if cfg.id not in denylist]
        
        # Filter available models
        if models:
            available_models = [cfg for cfg in self.model_configs if cfg.id in models]
            if available_models:
                self.model_configs = available_models
        
        # Sort by priority
        self.model_configs.sort(key=lambda x: x.priority)
        
        # Performance tracking (must be initialized before clients)
        self.stats = GenerationStats(0, 0, 0, 0, 0.0, {})
        self._total_response_time = 0.0
        
        # Last successful model for smart fallback
        self.last_successful_model = None

        # Initialize clients with connection pooling
        self.clients = {}
        self._initialize_clients()
        
        # Circuit breaker state per model
        self.circuit_breakers = {}
        for config in self.model_configs:
            self.circuit_breakers[config.id] = {
                'failures': 0,
                'last_failure': 0,
                'state': 'closed',  # closed, open, half-open
                'failure_threshold': 5,
                'recovery_time': 300  # 5 minutes
            }
        
        logger.info(f"✅ GroqClient initialized with {len(self.clients)} models")
    
    def _initialize_clients(self):
        """Initialize Groq clients with proper configuration"""
        
        for config in self.model_configs:
            try:
                # Create client with connection pooling
                client = Groq(
                    api_key=self.api_key,
                    timeout=httpx.Timeout(self.timeout),
                    max_retries=0  # We handle retries ourselves
                )
                
                self.clients[config.id] = client
                self.stats.model_usage[config.id] = 0
                
                logger.debug(f"Initialized client for {config.name}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize {config.name}: {e}")
        
        if not self.clients:
            raise RuntimeError("No Groq models available")
    
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None,
                query_type: str = "general",
                stream: bool = False,
                priority_model: Optional[str] = None) -> str:
        """
        Main generation method with comprehensive error handling and retries
        """
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
            
        start_time = time.time()
        
        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Select model order
            model_order = self._get_model_order(query_type, priority_model)
            
            # Try models with circuit breaker logic
            last_error = None
            
            for config in model_order:
                if not self._is_circuit_closed(config.id):
                    logger.debug(f"Circuit open for {config.id}, skipping")
                    continue
                
                try:
                    result = self._generate_with_model(config, messages, stream)
                    
                    if result:
                        # Success - update stats and circuit breaker
                        self._record_success(config.id, start_time)
                        return result
                        
                except Exception as e:
                    last_error = e
                    self._record_failure(config.id, e)
                    
                    # Log but continue to next model
                    logger.warning(f"Model {config.name} failed: {e}")
                    continue
            
            # All models failed
            self._record_total_failure(start_time)
            
            # Return graceful fallback
            fallback_msg = ("I apologize, but I'm unable to process your question at the moment. "
                          "This might be due to high demand or technical issues. Please try again in a few moments.")
            
            logger.error(f"All LLM models failed. Last error: {last_error}")
            return fallback_msg
            
        except Exception as e:
            self._record_total_failure(start_time)
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            return "I encountered an error while processing your request. Please try again."
    
    def _generate_with_model(self, config: ModelConfig, messages: List[Dict], stream: bool) -> Optional[str]:
        """Generate with specific model using retry logic"""
        
        client = self.clients.get(config.id)
        if not client:
            return None
        
        for attempt in range(self.max_retries):
            try:
                # Adjust parameters for model capabilities
                model_max_tokens = min(self.max_tokens, config.max_tokens)
                
                response = client.chat.completions.create(
                    model=config.id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=model_max_tokens,
                    stream=stream
                )
                
                if stream:
                    # Handle streaming response
                    full_content = ""
                    try:
                        for chunk in response:
                            if chunk.choices and chunk.choices[0].delta.content:
                                full_content += chunk.choices[0].delta.content
                        return full_content.strip() if full_content else None
                    except Exception as e:
                        logger.error(f"Streaming error: {e}")
                        return None
                else:
                    # Handle standard response
                    if (response.choices and 
                        response.choices[0].message and 
                        response.choices[0].message.content):
                        
                        content = response.choices[0].message.content.strip()
                        # Inspect finish_reason to detect truncation
                        try:
                            finish_reason = getattr(response.choices[0], 'finish_reason', None)
                        except Exception:
                            finish_reason = None
                        
                        # Update token usage if available
                        if hasattr(response, 'usage') and response.usage:
                            self.stats.total_tokens += response.usage.total_tokens
                        
                        if finish_reason:
                            logger.debug(f"✅ {config.name} finish_reason={finish_reason}; generated {len(content)} characters")
                            if str(finish_reason).lower() == 'length':
                                logger.warning(
                                    f"Model response truncated due to max_tokens={self.max_tokens}. "
                                    f"Consider increasing max_tokens or enabling a continuation step."
                                )
                        else:
                            logger.debug(f"✅ {config.name} generated {len(content)} characters")
                        return content
                
            except (httpx.TimeoutException, httpx.ConnectTimeout) as e:
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Timeout on attempt {attempt + 1}, retrying in {delay}s")
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"Timeout after {self.max_retries} attempts: {e}")
                    
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                # Retry only transient HTTP/network failures. Model-not-found,
                # decommissioned, invalid-request, and permission errors are
                # permanent for this model and should immediately fall through
                # to the next configured model.
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code in [400, 401, 403, 404]:
                    raise Exception(f"Non-retryable HTTP error ({status_code}): {e}")

                if attempt < self.max_retries - 1:
                    if status_code in [429, 500, 502, 503, 504] or status_code is None:
                        delay = self.retry_delay * (2 ** attempt)
                        logger.warning(
                            f"HTTP error {status_code or 'network'} on attempt "
                            f"{attempt + 1}, retrying in {delay}s"
                        )
                        time.sleep(delay)
                        continue
                raise Exception(f"HTTP error: {e}")
                
            except Exception as e:
                error_text = str(e).lower()

                # Groq can surface model lifecycle/access failures as generic
                # exceptions from the SDK. Do not waste retries on these.
                permanent_markers = (
                    "model_not_found",
                    "model_decommissioned",
                    "does not exist or you do not have access",
                    "decommissioned and is no longer supported",
                )
                if any(marker in error_text for marker in permanent_markers):
                    raise Exception(f"Non-retryable model error: {e}")

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Error on attempt {attempt + 1}: {e}, retrying in {delay}s"
                    )
                    time.sleep(delay)
                    continue
                else:
                    raise
        
        return None
    
    def _get_model_order(self, query_type: str, priority_model: Optional[str]) -> List[ModelConfig]:
        """Get ordered list of models to try based on query type and preferences"""
        
        # If priority model specified, try it first
        if priority_model:
            priority_config = next((cfg for cfg in self.model_configs if cfg.id == priority_model), None)
            if priority_config:
                other_configs = [cfg for cfg in self.model_configs if cfg.id != priority_model]
                return [priority_config] + other_configs
        
        # Smart ordering based on query type
        type_preferences = {
            "relationship": ["production", "complex", "reasoning"],
            "narrative": ["complex", "production", "reasoning"],
            "analysis": ["reasoning", "complex", "production"],
            "quick": ["speed", "production"],
            "sanskrit": ["multilingual", "complex", "production"],
            "general": ["production", "complex", "reasoning"]
        }
        
        preferred_types = type_preferences.get(query_type, ["production"])
        
        # Sort by type preference, then by priority
        def sort_key(config):
            type_priority = preferred_types.index(config.type) if config.type in preferred_types else len(preferred_types)
            return (type_priority, config.priority)
        
        return sorted(self.model_configs, key=sort_key)
    
    def _is_circuit_closed(self, model_id: str) -> bool:
        """Check if circuit breaker allows requests"""
        
        breaker = self.circuit_breakers.get(model_id, {})
        
        if breaker.get('state') == 'open':
            # Check if recovery time has passed
            if time.time() - breaker.get('last_failure', 0) > breaker.get('recovery_time', 300):
                breaker['state'] = 'half-open'
                logger.info(f"Circuit breaker for {model_id} moved to half-open")
                return True
            return False
        
        return True
    
    def _record_success(self, model_id: str, start_time: float):
        """Record successful generation"""
        
        response_time = time.time() - start_time
        
        # Update general stats
        self.stats.total_requests += 1
        self.stats.successful_requests += 1
        self.stats.model_usage[model_id] += 1
        self._total_response_time += response_time
        self.stats.avg_response_time = self._total_response_time / self.stats.total_requests
        
        # Update circuit breaker
        breaker = self.circuit_breakers[model_id]
        breaker['failures'] = 0
        breaker['state'] = 'closed'
        
        # Update last successful model
        self.last_successful_model = model_id
        
        logger.debug(f"✅ Success with {model_id} in {response_time:.2f}s")
    
    def _record_failure(self, model_id: str, error: Exception):
        """Record failed generation attempt"""
        
        breaker = self.circuit_breakers[model_id]
        breaker['failures'] += 1
        breaker['last_failure'] = time.time()
        
        # Open circuit if threshold reached
        if breaker['failures'] >= breaker['failure_threshold']:
            breaker['state'] = 'open'
            logger.warning(f"Circuit breaker opened for {model_id} after {breaker['failures']} failures")
    
    def _record_total_failure(self, start_time: float):
        """Record complete failure of all models"""
        
        response_time = time.time() - start_time
        
        self.stats.total_requests += 1
        self.stats.failed_requests += 1
        self._total_response_time += response_time
        
        if self.stats.total_requests > 0:
            self.stats.avg_response_time = self._total_response_time / self.stats.total_requests
    
    async def generate_async(self, 
                           prompt: str, 
                           system_prompt: Optional[str] = None,
                           query_type: str = "general") -> str:
        """Async version of generate"""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.generate, prompt, system_prompt, query_type
        )
    
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Simple completion method for compatibility"""
        return self.generate(user_prompt, system_prompt)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive client statistics"""
        
        return {
            "available_models": [config.name for config in self.model_configs if config.id in self.clients],
            "last_successful_model": self.last_successful_model,
            "generation_stats": {
                "total_requests": self.stats.total_requests,
                "successful_requests": self.stats.successful_requests,
                "failed_requests": self.stats.failed_requests,
                "success_rate": (self.stats.successful_requests / max(1, self.stats.total_requests)) * 100,
                "total_tokens": self.stats.total_tokens,
                "avg_response_time": round(self.stats.avg_response_time, 3),
            },
            "model_usage": self.stats.model_usage,
            "circuit_breakers": {
                model_id: {
                    "state": breaker["state"],
                    "failures": breaker["failures"]
                }
                for model_id, breaker in self.circuit_breakers.items()
            },
            "total_models": len(self.model_configs),
            "active_models": len(self.clients)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all models"""
        
        results = {}
        test_prompt = "Hello"
        
        for config in self.model_configs:
            if config.id not in self.clients:
                results[config.id] = {"status": "unavailable", "error": "Client not initialized"}
                continue
                
            if not self._is_circuit_closed(config.id):
                results[config.id] = {"status": "circuit_open", "error": "Circuit breaker open"}
                continue
            
            try:
                start_time = time.time()
                response = self._generate_with_model(
                    config,
                    [{"role": "user", "content": test_prompt}],
                    stream=False
                )
                response_time = time.time() - start_time
                
                results[config.id] = {
                    "status": "healthy" if response else "degraded",
                    "response_time": round(response_time, 3),
                    "response_length": len(response) if response else 0
                }
                
            except Exception as e:
                results[config.id] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        overall_status = "healthy" if any(r.get("status") == "healthy" for r in results.values()) else "unhealthy"
        
        return {
            "overall_status": overall_status,
            "model_health": results,
            "timestamp": time.time()
        }
    
    def cleanup(self):
        """Clean up resources"""
        
        logger.info("Cleaning up LLM client resources...")
        
        # Close HTTP clients if they have cleanup methods
        for model_id, client in self.clients.items():
            try:
                if hasattr(client, 'close'):
                    client.close()
            except Exception as e:
                logger.warning(f"Error closing client {model_id}: {e}")
        
        self.clients.clear()
        logger.info("✅ LLM client cleanup complete")

# Factory function for external compatibility
def create_groq_client(api_key: str, models: List[str] = None) -> GroqClient:
    """Factory function to create GroqClient with production configuration"""
    
    return GroqClient(
        api_key=api_key,
        models=models,
        temperature=0.3,
        max_tokens=1024,
        timeout=30,
        max_retries=3,
        retry_delay=1.0
    )
