"""
VANI V3 LLM Prompts - Structured Prompt Templates
Enforcing: Query-type specific prompts for optimal responses
"""

from typing import Dict, List
from enum import Enum
from app.settings import settings


class QueryType(Enum):
    GENERAL = "general"
    RELATIONSHIP = "relationship" 
    NARRATIVE = "narrative"
    ANALYSIS = "analysis"


class VANIPrompts:
    """Centralized prompt management for VANI"""
    
    BASE_SYSTEM_PROMPT = """You are VANI (Voice of Ancient Narrative Intelligence), an expert assistant specializing in the Ramayana.

Your role:
- Provide accurate, respectful answers about the Ramayana
- Never invent or fabricate citations or sources  
- Maintain cultural sensitivity and reverence for the sacred text
- Keep responses concise but comprehensive

Source usage rules:
- Use ONLY the provided sources enumerated as [S1], [S2], ... when forming your answer.
- Do NOT use any knowledge outside these sources (unless asked for general background); if unsure, say you do not have enough evidence.

Response guidelines:
- Write a cohesive answer (short paragraphs) that synthesizes evidence from multiple provided sources.
- Do NOT include citations inline; the system will list citations separately.
- Prefer English phrasing. Be specific and coherent.
- If the sources are insufficient, state that you don't have enough evidence to answer confidently.
- Do NOT include any chain-of-thought, analysis steps, or <think> blocks in the output."""

    QUERY_SPECIFIC_PROMPTS = {
        QueryType.GENERAL: """Focus on providing a clear, comprehensive answer supported by the most relevant verses.""",
        
        QueryType.RELATIONSHIP: """Focus on family relationships, character connections, and genealogies. Be precise about relationships. Be short.""",
        
        QueryType.NARRATIVE: """Focus on storytelling elements and plot progression. Help the user understand the sequence of events and their significance in the larger epic.""",
        
        QueryType.ANALYSIS: """Provide deeper analysis of themes, symbolism, and meanings. Draw connections between verses and explain significance in the context of dharma, devotion, and moral teachings."""
    }

    @classmethod
    def get_system_prompt(cls, query_type: QueryType = QueryType.GENERAL) -> str:
        """Get complete system prompt for query type"""
        
        base_prompt = cls.BASE_SYSTEM_PROMPT
        # Apply Valmiki-only policy and Uttara Kanda suppression if configured
        policy_lines: List[str] = []
        try:
            if getattr(settings, 'restrict_to_valmiki', False):
                policy_lines.append(
                    "Scope restriction: Use ONLY the Valmiki Ramayana (Baala, Ayodhya, Aranya, Kishkindha, Sundara, Yuddha).\n"
                    "Do not rely on external texts or later interpolations."
                )
            if getattr(settings, 'disallow_uttara_kanda_mentions', False):
                policy_lines.append("Do NOT mention 'Uttara Kanda' anywhere in the answer.")
        except Exception:
            pass
        if policy_lines:
            base_prompt = f"{base_prompt}\n\nPolicy:\n- " + "\n- ".join(policy_lines)
        specific_guidance = cls.QUERY_SPECIFIC_PROMPTS.get(query_type, "")
        
        if specific_guidance:
            return f"{base_prompt}\n\nSpecific guidance for this query:\n{specific_guidance}"
        
        return base_prompt

    @classmethod 
    def create_user_prompt(cls, question: str, sources: List[Dict], query_type: QueryType = QueryType.GENERAL) -> str:
        """Create formatted user prompt with minimal S[] blocks (Target + Previous/Next English only)"""
        
        # Format sources as [S#] with minimal context
        source_lines = []
        for i, source in enumerate(sources, 1):
            # Build display citation in K.S.V or K.S.V1-V2
            _vid = str(source.get('verse_id', 'Unknown'))
            parts = _vid.split('-')
            if len(parts) == 3:
                verse_id_display = f"{parts[0]}.{parts[1]}.{parts[2]}"
            elif len(parts) == 4:
                verse_id_display = f"{parts[0]}.{parts[1]}.{parts[2]}-{parts[3]}"
            else:
                verse_id_display = _vid.replace('-', '.')
            kanda_name = source.get('kanda_name', 'Unknown')
            sarga_num = source.get('sarga_num', 'Unknown')

            header = f"[S{i}] {verse_id_display} {kanda_name}, Sarga {sarga_num}"
            source_lines.append(header)

            # Minimal fields: Target (English), plus Previous/Next (English) if present
            def _cln(text: str) -> str:
                # Normalize whitespace and remove newlines; do not truncate
                return (text or '').replace('\n', ' ').strip()

            target = _cln(source.get('translation_en') or '')
            if target:
                source_lines.append(f"    Target: {target}")

            prev_en = _cln(source.get('prev_en') or '')
            if prev_en:
                source_lines.append(f"    Previous: {prev_en}")

            next_en = _cln(source.get('next_en') or '')
            if next_en:
                source_lines.append(f"    Next: {next_en}")
        
        sources_text = '\n\n'.join(source_lines)
        
        # Create user prompt
        user_prompt = f"""Question: {question}
Provided sources (use ONLY these):

{sources_text}

Instructions:
 - Answer using ONLY the content from the provided sources [S1], [S2], ...
 - Write a coherent answer (short paragraphs). Do NOT include citations inside the text.
 - Focus on clarity, context, and specificity; integrate evidence from multiple sources. Use each verse's Target along with its Previous/Next lines to understand immediate narrative context within the same sarga.
 - If the sources are insufficient to answer confidently, say you don't have enough evidence.
 - Do not include any chain-of-thought, 'think' tags, or step-by-step reasoning in the output.
"""

        # Append policy reminder to the user prompt as an extra safeguard
        try:
            policy_lines: List[str] = []
            if getattr(settings, 'restrict_to_valmiki', False):
                policy_lines.append(
                    "Use only Valmiki Ramayana content (Bala, Ayodhya, Aranya, Kishkindha, Sundara, Yuddha)."
                )
            if getattr(settings, 'disallow_uttara_kanda_mentions', False):
                policy_lines.append("Do not mention 'Uttara Kanda'.")
            if policy_lines:
                user_prompt += "\n\nPolicy: " + " ".join(policy_lines)
        except Exception:
            pass

        return user_prompt

    @classmethod
    def create_count_response_template(cls, count_type: str, count: int, target: str = "") -> str:
        """Create template response for count queries"""
        
        templates = {
            "kandas": f"The Ramayana is divided into {count} kandas (books).",
            "verses": f"The complete Ramayana contains {count:,} verses across all kandas.",
            "sargas": f"The {target} contains {count} sargas (chapters)." if target else f"There are {count} sargas.",
            "verses_in_kanda": f"The {target} contains {count:,} verses." if target else f"There are {count:,} verses."
        }
        
        return templates.get(count_type, f"Count result: {count}")

    @classmethod
    def create_no_results_response(cls) -> str:
        """Response when no relevant verses found"""
        return "No relevant verses were found to answer this question. You may want to rephrase your query or try different terms related to the Ramayana."

    @classmethod
    def create_error_response(cls, error_type: str = "general") -> str:
        """Error response templates"""
        
        error_templates = {
            "general": "I apologize, but I encountered an error while processing your question. Please try again.",
            "no_llm": "The AI response system is currently unavailable. Here are the most relevant verses I found for your question.",
            "invalid_citation": "I found relevant verses but cannot provide a complete answer at this time. Please refer to the source verses listed.",
            "timeout": "Your query is taking longer than expected. Please try a simpler question or try again later."
        }
        
        return error_templates.get(error_type, error_templates["general"])


# Convenience functions
def get_prompt_for_query(question: str, sources: List[Dict], query_type: str = "general") -> tuple:
    """
    Get system and user prompts for a query
    
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    
    # Convert string to enum
    try:
        query_enum = QueryType(query_type.lower())
    except ValueError:
        query_enum = QueryType.GENERAL
    
    system_prompt = VANIPrompts.get_system_prompt(query_enum)
    user_prompt = VANIPrompts.create_user_prompt(question, sources, query_enum)
    
    return system_prompt, user_prompt


def format_count_response(count_type: str, count: int, target: str = "") -> str:
    """Format count query response"""
    return VANIPrompts.create_count_response_template(count_type, count, target)
