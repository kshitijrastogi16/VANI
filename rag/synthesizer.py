"""
VANI V3 Response Synthesizer - LLM Integration with Citation Management
Enforcing: Comprehensive citations with verse details (Sanskrit, IAST, English, Hindi)
"""

import re
import logging
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import sqlite3

from llm.client import GroqClient
from llm.prompts import VANIPrompts, QueryType
from rag.citations import (
    extract_citations,
    remove_invalid_citations,
    denormalize_verse_id,
    normalize_verse_id,
)
from core.search.hybrid_engine import SearchResult
from app.settings import settings
from app.metrics import RAG_ABSTENTIONS, RAG_STRICT_CITATION_FALLBACK

logger = logging.getLogger(__name__)

KANDA_NAMES_HI = {
    "baala kanda": "बालकाण्ड",
    "ayodhya kanda": "अयोध्याकाण्ड",
    "aranya kanda": "अरण्यकाण्ड",
    "kishkindha kanda": "किष्किन्धाकाण्ड",
    "sundara kanda": "सुन्दरकाण्ड",
    "yuddha kanda": "युद्धकाण्ड",
}


@dataclass
class SynthesizedResponse:
    answer: str
    sources: List[Dict[str, Any]]
    citations: List[str]
    confidence: float
    query_type: str
    llm_used: bool
    processing_time: float


class ResponseSynthesizer:
    """
    Production Response Synthesizer with Comprehensive Citation Support
    Features: LLM integration, citation validation, fallback templates
    """
    
    def __init__(self, llm_client: Optional[GroqClient] = None):
        self.llm_client = llm_client
        self.last_context_summary_llm_used = False
        logger.info(f"✅ Response synthesizer initialized (LLM: {'Available' if llm_client else 'None'})")
    
    def _sanitize_summary_text(self, text: str) -> str:
        """Remove any chain-of-thought style content, think tags, and code fences.
        Keep only final answer sentences."""
        try:
            # Remove <think>...</think> and similar custom tags
            text = re.sub(r"<\s*(?:think|analysis|reasoning)[^>]*>[\s\S]*?<\s*/\s*(?:think|analysis|reasoning)\s*>", "", text, flags=re.IGNORECASE)
            # Remove fenced blocks labeled as think/analysis/reasoning
            text = re.sub(r"```\s*(?:think|analysis|reasoning)[\s\S]*?```", "", text, flags=re.IGNORECASE)
            # Remove common chain-of-thought lead-ins
            text = re.sub(r"(?im)^(?:thought|reasoning|analysis|let's think step by step|scratchpad)\s*:\s*.*$", "", text)
            # Collapse whitespace
            text = re.sub(r"\n{2,}", "\n", text)
            text = text.strip()
            return text
        except Exception:
            return text.strip() if isinstance(text, str) else ""

    def _apply_policy_sanitization(self, text: str) -> str:
        """Apply Valmiki-only policy sanitization to LLM output.
        Currently removes/suppresses mentions of 'Uttara Kanda' if configured.
        """
        try:
            if not isinstance(text, str):
                return text
            if getattr(settings, 'disallow_uttara_kanda_mentions', False):
                # Remove common variants (case-insensitive)
                patterns = [
                    r"(?i)uttara\s*kanda",
                    r"(?i)uttarakanda",
                    r"(?i)uttar\s*k[aā]nda",
                ]
                sanitized = text
                for p in patterns:
                    sanitized = re.sub(p, "", sanitized)
                # Remove any leftover doubled spaces
                sanitized = re.sub(r"\s{2,}", " ", sanitized).strip()
                return sanitized
            return text
        except Exception:
            return text

    def _limit_sentences(self, text: str, max_sentences: int = 3) -> str:
        """Limit text to a maximum number of sentences using simple punctuation heuristics."""
        try:
            # Quick split on sentence-ending punctuation
            parts = re.split(r"(?<=[.!?])\s+", text.strip())
            if len(parts) <= max_sentences:
                return text.strip()
            limited = " ".join(parts[:max_sentences]).strip()
            return limited
        except Exception:
            return text.strip()

    def _get_prev_next_en(
        self,
        kanda_num: int,
        sarga_num: int,
        prev_index: Optional[int],
        next_index: Optional[int],
    ) -> tuple[str, str]:
        """Fetch Previous and Next English translations for given verse indices within the same sarga.
        Returns (prev_en, next_en); empty strings when not available.
        """
        try:
            if not kanda_num or not sarga_num or (prev_index is None and next_index is None):
                return "", ""

            indices: List[int] = []
            params: List[Any] = [int(kanda_num), int(sarga_num)]
            if isinstance(prev_index, int) and prev_index > 0:
                indices.append(prev_index)
            if isinstance(next_index, int) and next_index > 0 and next_index != prev_index:
                indices.append(next_index)
            if not indices:
                return "", ""

            placeholders = ",".join(["?"] * len(indices))
            query = (
                """
                SELECT verse_index, translation_en
                FROM verses
                WHERE kanda_num = ? AND sarga_num = ? AND verse_index IN ("""
                + placeholders
                + ")"
            )

            conn = sqlite3.connect(settings.database_path)
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.execute(query, params + indices)
                rows = cur.fetchall()
            finally:
                conn.close()

            lookup = {int(r["verse_index"]): (r["translation_en"] or "").strip() for r in (rows or [])}
            prev_en = lookup.get(prev_index, "") if isinstance(prev_index, int) else ""
            next_en = lookup.get(next_index, "") if isinstance(next_index, int) else ""
            return prev_en, next_en
        except Exception:
            return "", ""
    
    def _create_system_prompt(self, qtype: QueryType) -> str:
        """Get system prompt for query type via VANIPrompts"""
        return VANIPrompts.get_system_prompt(qtype)
    
    def synthesize(
        self,
        query: str,
        documents: List[SearchResult],
        query_type: str = "general",
        max_sources: int = 5,
        response_language: str = "en",
    ) -> SynthesizedResponse:
        """
        Main synthesis method with LLM integration and fallback
        """
        
        import time
        start_time = time.time()
        
        logger.info(f"Synthesizing response for query type: {query_type}")
        
        if not documents:
            return self._create_no_results_response(query, start_time, response_language)
        
        # If evidence is weak, optionally abstain early
        try:
            abstain_threshold = float(getattr(settings, 'abstain_below_score', 0.0) or 0.0)
            if abstain_threshold > 0.0 and (documents[0].final_score or 0.0) < abstain_threshold:
                # Metrics: abstention due to low fused score
                try:
                    RAG_ABSTENTIONS.labels(reason="low_score").inc()
                except Exception:
                    pass
                return self._create_insufficient_evidence_response(query, documents, start_time, response_language)
        except Exception:
            pass
        
        # Prepare sources for processing
        top_documents = documents[:max_sources]
        formatted_sources = self._format_sources_for_llm(top_documents)
        
        # Attempt LLM generation
        if self.llm_client:
            try:
                llm_response = self._generate_with_llm(query, top_documents, query_type, response_language)
                if llm_response:
                    processed = self._process_llm_response(
                        query, llm_response, top_documents, query_type, start_time, True,
                        response_language=response_language,
                    )
                    # Strict citation handling: if enabled and no valid citations, fall back to template
                    try:
                        if getattr(settings, 'strict_citations_required', False) and not processed.citations:
                            # Metrics: strict citation fallback used
                            try:
                                RAG_STRICT_CITATION_FALLBACK.inc()
                            except Exception:
                                pass
                            template_response = self._generate_template_response(query, top_documents, response_language)
                            return self._process_llm_response(
                                query, template_response, top_documents, query_type, start_time, False,
                                response_language=response_language,
                            )
                    except Exception:
                        pass
                    return processed
            except Exception as e:
                logger.warning("LLM generation failed (%s)", type(e).__name__)
        
        # Fallback to template response
        logger.info("Using template fallback response")
        template_response = self._generate_template_response(query, top_documents, response_language)
        
        return self._process_llm_response(
            query, template_response, top_documents, query_type, start_time, False,
            response_language=response_language,
        )
    
    def _format_sources_for_llm(self, documents: List[SearchResult]) -> str:
        """
        Format search results for LLM consumption
        """
        
        formatted_lines = []
        
        for i, doc in enumerate(documents, 1):
            # Create citation reference in K.S.V[-V2] display format
            citation = denormalize_verse_id(doc.verse_id)
            
            # Create contextual header
            context = ""
            if doc.sarga_titles:
                context = f" - {doc.sarga_titles[0]}"
            # Display fallbacks for unknowns
            try:
                kanda_display = (doc.kanda_name or '').strip() or 'Unknown Kanda'
            except Exception:
                kanda_display = 'Unknown Kanda'
            try:
                sarga_n = int(getattr(doc, 'sarga_num', 0) or 0)
                sarga_display = sarga_n if sarga_n > 0 else '?'
            except Exception:
                sarga_display = '?'

            header = f"{i}. {citation} {kanda_display}, Sarga {sarga_display}{context}"
            
            # Add translations
            content_lines = [header]

            # Order: Sanskrit (Devanagari), IAST, English, Hindi (include when present)
            if doc.shloka_sanskrit:
                content_lines.append(f"   Sanskrit: {doc.shloka_sanskrit}")
            if doc.shloka_iast:
                content_lines.append(f"   IAST: {doc.shloka_iast}")
            if doc.translation_en:
                content_lines.append(f"   English: {doc.translation_en}")
            if doc.translation_hi:
                content_lines.append(f"   Hindi: {doc.translation_hi}")

            # Include window-level concatenated content when available (trimmed)
            try:
                if getattr(doc, 'window_translation_en', None):
                    w_en = str(doc.window_translation_en)
                    if len(w_en) > 800:
                        w_en = w_en[:797].rstrip() + "..."
                    content_lines.append(f"   Window English: {w_en}")
            except Exception:
                pass
            try:
                if getattr(doc, 'window_shloka_iast', None):
                    w_iast = str(doc.window_shloka_iast)
                    if len(w_iast) > 800:
                        w_iast = w_iast[:797].rstrip() + "..."
                    content_lines.append(f"   Window IAST: {w_iast}")
            except Exception:
                pass
            
            formatted_lines.append('\n'.join(content_lines))
        
        return '\n\n'.join(formatted_lines)
    
    def _generate_with_llm(
        self,
        query: str,
        sources: List[SearchResult],
        query_type: str,
        response_language: str = "en",
    ) -> Optional[str]:
        """Generate response using LLM client with VANIPrompts"""
        # Build structured sources for prompt builder
        structured_sources = []
        for doc in sources:
            # Derive kanda_num/sarga_num and immediate neighbor indices from verse_id
            k_num = 0
            s_num = int(getattr(doc, 'sarga_num', 0) or 0)
            prev_idx: Optional[int] = None
            next_idx: Optional[int] = None
            try:
                parts = str(getattr(doc, 'verse_id', '') or '').split('-')
                if len(parts) >= 2:
                    k_num = int(parts[0])
                    # prefer parsed sarga if doc's sarga_num missing
                    if not s_num:
                        s_num = int(parts[1])
                if len(parts) == 3:
                    v = int(parts[2])
                    prev_idx = v - 1 if v > 1 else None
                    next_idx = v + 1
                elif len(parts) == 4:
                    v_start = int(parts[2])
                    v_end = int(parts[3])
                    prev_idx = v_start - 1 if v_start > 1 else None
                    next_idx = v_end + 1
                else:
                    # Fallback: use verse_index if available
                    vi = int(getattr(doc, 'verse_index', 0) or 0)
                    if vi:
                        prev_idx = vi - 1 if vi > 1 else None
                        next_idx = vi + 1
            except Exception:
                prev_idx, next_idx = None, None

            prev_en, next_en = self._get_prev_next_en(k_num, s_num, prev_idx, next_idx)

            structured_sources.append({
                'verse_id': doc.verse_id,
                'kanda_name': doc.kanda_name,
                'sarga_num': doc.sarga_num,
                'verse_index': getattr(doc, 'verse_index', 0),
                'translation_en': doc.translation_en,
                'translation_hi': doc.translation_hi,
                'shloka_sanskrit': doc.shloka_sanskrit,
                'shloka_iast': doc.shloka_iast,
                'sarga_titles': doc.sarga_titles,
                # Minimal neighbor context for prompt formatting
                'prev_en': prev_en,
                'next_en': next_en,
                # Pass window context through to prompts (not used in minimal format)
                'window_translation_en': getattr(doc, 'window_translation_en', ''),
                'window_shloka_iast': getattr(doc, 'window_shloka_iast', ''),
            })

        # Resolve query type enum
        try:
            qtype = QueryType(query_type.lower())
        except Exception:
            qtype = QueryType.GENERAL

        system_prompt = self._create_system_prompt(qtype)
        language_instructions = {
            "en": "Respond in clear English.",
            "hi": "उत्तर केवल शुद्ध हिन्दी में देवनागरी लिपि में दें। अंग्रेज़ी वाक्य न मिलाएँ। पात्रों और स्थानों के प्रचलित नाम सुरक्षित रखें।",
            "hinglish": "Answer in natural Hinglish using Roman script, mixing Hindi and English naturally. Do not use Devanagari unless quoting source text.",
        }
        system_prompt += "\n\nRESPONSE LANGUAGE REQUIREMENT: " + language_instructions.get(
            response_language, language_instructions["en"]
        )
        user_prompt = VANIPrompts.create_user_prompt(query, structured_sources, qtype)

        try:
            response = self.llm_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                query_type=query_type
            )
            if response and response.strip():
                normalized = response.strip().lower()
                provider_fallbacks = (
                    "i apologize, but i'm unable to process your question",
                    "i encountered an error while processing your request",
                )
                if any(normalized.startswith(prefix) for prefix in provider_fallbacks):
                    return None
                # Sanitize out any chain-of-thought / <think> blocks proactively
                clean = self._sanitize_summary_text(response.strip())
                return clean
        except Exception as e:
            logger.error("LLM generation error (%s)", type(e).__name__)
        return None
    
    def _generate_template_response(
        self, query: str, documents: List[SearchResult], response_language: str = "en"
    ) -> str:
        """
        Generate template-based fallback response
        """
        
        if not documents:
            return "No relevant verses found for this query."
        
        best_result = documents[0]
        citation = denormalize_verse_id(best_result.verse_id)
        
        # Create contextual response
        response_parts = []
        
        # Main reference
        context = ""
        if best_result.sarga_titles:
            context = f" - {best_result.sarga_titles[0]}"
        # Display-friendly fallbacks
        try:
            kanda_display = (best_result.kanda_name or '').strip() or 'Unknown Kanda'
        except Exception:
            kanda_display = 'Unknown Kanda'
        try:
            sarga_n = int(getattr(best_result, 'sarga_num', 0) or 0)
            sarga_display = sarga_n if sarga_n > 0 else '?'
        except Exception:
            sarga_display = '?'

        if response_language == "hi":
            kanda_display = KANDA_NAMES_HI.get(kanda_display.lower(), kanda_display)
            response_parts.append(
                f"इस प्रश्न के लिए सबसे प्रासंगिक संदर्भ {citation} है। यह {kanda_display} के सर्ग {sarga_display}{context} से है।"
            )
            source_translation = best_result.translation_hi or best_result.translation_en
        elif response_language == "hinglish":
            response_parts.append(
                f"Is question ke liye sabse relevant reference {citation} hai. Yeh {kanda_display} ke Sarga {sarga_display}{context} se hai."
            )
            source_translation = best_result.translation_hi or best_result.translation_en
        else:
            response_parts.append(
                f"The most relevant reference is {citation} from {kanda_display}, "
                f"Sarga {sarga_display}{context}."
            )
            source_translation = best_result.translation_en or best_result.translation_hi

        # Add only a translation actually present in the dataset.
        if source_translation:
            response_parts.append(f"\n\"{source_translation}\"")
        
        # Add additional references if available
        if len(documents) > 1:
            additional_refs = []
            for doc in documents[1:3]:  # Up to 2 additional references
                add_citation = denormalize_verse_id(doc.verse_id)
                additional_refs.append(f"{add_citation}")
            
            if additional_refs:
                if response_language == "hi":
                    prefix = "\n\nअन्य संदर्भ: "
                elif response_language == "hinglish":
                    prefix = "\n\nAur dekhein: "
                else:
                    prefix = "\n\nSee also: "
                response_parts.append(prefix + ", ".join(ref for ref in additional_refs if ref))
        
        return ''.join(response_parts)
    
    def _process_llm_response(
        self,
        query: str,
        response_text: str,
        documents: List[SearchResult],
        query_type: str,
        start_time: float,
        llm_used: bool,
        response_language: str = "en",
    ) -> SynthesizedResponse:
        """
        Process and validate LLM response, extract citations
        """
        
        # Derive citations from selected documents (no inline citations in text)
        derived_citations: List[str] = []
        seen: set[str] = set()
        for doc in documents:
            try:
                # Anchor citation
                if getattr(doc, 'verse_id', None):
                    dot = denormalize_verse_id(str(doc.verse_id))
                    if dot not in seen:
                        seen.add(dot)
                        derived_citations.append(dot)
                # Range citation for windows within same kanda/sarga
                vlist = getattr(doc, 'verse_ids', None) or []
                # Only add a window-derived range when the doc's verse_id is a single verse.
                # If verse_id is already a compact range (k-s-v1-v2), prefer that and skip the window range.
                is_doc_range = False
                try:
                    parts = str(getattr(doc, 'verse_id', '')).split('-')
                    is_doc_range = (len(parts) == 4)
                except Exception:
                    is_doc_range = False
                if vlist and (len(vlist) >= 2) and (not is_doc_range):
                    first = str(vlist[0]).split('-')
                    last = str(vlist[-1]).split('-')
                    if len(first) >= 3 and len(last) >= 3 and first[0] == last[0] and first[1] == last[1]:
                        range_id = f"{first[0]}.{first[1]}.{first[2]}-{last[2]}"
                        if range_id not in seen:
                            seen.add(range_id)
                            derived_citations.append(range_id)
            except Exception:
                continue

        # Preserve source-backed template citations. For model-generated text,
        # remove only citations that do not exist in the retrieved source set.
        try:
            if llm_used:
                allowed_verse_ids = {
                    str(doc.verse_id) for doc in documents
                    if getattr(doc, "verse_id", None)
                }
                response_text = remove_invalid_citations(response_text, allowed_verse_ids)
            response_text = re.sub(r"\s{2,}", " ", response_text).strip()
        except Exception:
            response_text = response_text.strip() if isinstance(response_text, str) else ""

        valid_citations = derived_citations

        # Enforce strict citations if configured: if none, fallback to deterministic template
        try:
            if getattr(settings, 'strict_citations_required', False) and not valid_citations:
                # Metrics: strict citation fallback used
                try:
                    RAG_STRICT_CITATION_FALLBACK.inc()
                except Exception:
                    pass
                template_response = self._generate_template_response(query, documents, response_language)
                # Derive citations again from documents (same as above)
                valid_citations = [denormalize_verse_id(getattr(d, 'verse_id', '')) for d in documents if getattr(d, 'verse_id', None)]
                response_text = template_response
        except Exception:
            pass
        
        # Create comprehensive source information
        sources = []
        for doc in documents:
            source = {
                "verse_id": doc.verse_id,
                # Human-friendly dot format for display
                "citation_format": denormalize_verse_id(doc.verse_id),
                "kanda_name": doc.kanda_name,
                "sarga_num": doc.sarga_num,
                "verse_index": doc.verse_index,
                "translation_en": doc.translation_en,
                "translation_hi": doc.translation_hi,
                "shloka_sanskrit": doc.shloka_sanskrit,
                "shloka_iast": doc.shloka_iast,
                "sarga_titles": doc.sarga_titles,
                "search_score": doc.final_score
            }
            sources.append(source)
        
        # Confidence should reflect evidence relevance, not whether an LLM was used
        # or whether citations were mechanically attached. Fused retrieval scores are
        # not probabilities, so keep this conservative and include query/source overlap.
        confidence = 0.15
        if documents:
            top = documents[0]
            try:
                top_score = max(0.0, float(top.final_score or 0.0))
            except (TypeError, ValueError):
                top_score = 0.0
            # A saturating score contribution avoids treating an arbitrary fused score
            # (which can exceed 1 after reranking) as a percentage probability.
            confidence += 0.25 * (top_score / (top_score + 0.5)) if top_score > 0 else 0.0
            query_terms = set(re.findall(r"[a-z0-9]+", (query or '').lower()))
            stopwords = {
                'a', 'an', 'and', 'are', 'about', 'as', 'at', 'be', 'by', 'did',
                'do', 'for', 'from', 'how', 'in', 'is', 'it', 'of', 'on', 'or',
                'tell', 'the', 'to', 'was', 'what', 'when', 'where', 'who', 'why',
                'with', 'me', 'please', 'describe', 'explain', 'story'
            }
            query_terms -= stopwords
            source_text = ' '.join(
                str(getattr(doc, 'translation_en', '') or '') + ' ' +
                ' '.join(getattr(doc, 'sarga_titles', None) or []) + ' ' +
                str(getattr(doc, 'kanda_name', '') or '')
                for doc in documents[:3]
            ).lower()
            if query_terms:
                covered = sum(1 for term in query_terms if re.search(r"\b" + re.escape(term) + r"\b", source_text))
                coverage = covered / len(query_terms)
                confidence += 0.25 * coverage
            if valid_citations:
                confidence += 0.05
            # An LLM may write fluently even when retrieval is wrong; do not reward it.
            confidence = min(0.75, max(0.0, confidence))
        else:
            confidence = 0.0
        
        # Apply output policy sanitization (e.g., remove 'Uttara Kanda' mentions)
        response_text = self._apply_policy_sanitization(response_text)
        
        processing_time = time.time() - start_time
        
        return SynthesizedResponse(
            answer=response_text,
            sources=sources,
            citations=valid_citations,
            confidence=confidence,
            query_type=query_type,
            llm_used=llm_used,
            processing_time=processing_time
        )
    
    def _create_no_results_response(
        self, query: str, start_time: float, response_language: str = "en"
    ) -> SynthesizedResponse:
        """
        Create response when no documents are found
        """
        
        return SynthesizedResponse(
            answer=(
                "इस प्रश्न का उत्तर देने के लिए कोई प्रासंगिक श्लोक नहीं मिला। प्रश्न को दूसरे शब्दों में पूछकर देखें।"
                if response_language == "hi" else
                "Is sawaal ka jawab dene ke liye relevant shlok nahi mile. Sawal ko alag shabdon mein pooch kar dekhein."
                if response_language == "hinglish" else
                "No relevant verses were found to answer this question. You may want to rephrase your query or try different terms."
            ),
            sources=[],
            citations=[],
            confidence=0.0,
            query_type="no_results",
            llm_used=False,
            processing_time=time.time() - start_time
        )

    def _create_insufficient_evidence_response(
        self,
        query: str,
        documents: List[SearchResult],
        start_time: float,
        response_language: str = "en",
    ) -> SynthesizedResponse:
        """Return a cautious response when the top fused score is too low.
        Include the top 1–2 citations as suggestions rather than a definitive answer."""
        try:
            top = documents[:2]
            suggestions = ", ".join(denormalize_verse_id(d.verse_id) for d in top if getattr(d, 'verse_id', None))
        except Exception:
            suggestions = ""
        if response_language == "hi":
            text = ("विश्वसनीय उत्तर देने के लिए पर्याप्त प्रमाण नहीं मिले। संबंधित संदर्भ: " + suggestions) if suggestions else "विश्वसनीय उत्तर देने के लिए पर्याप्त प्रमाण नहीं मिले।"
        elif response_language == "hinglish":
            text = ("Confident jawab dene ke liye kaafi evidence nahi mila. Relevant references: " + suggestions) if suggestions else "Confident jawab dene ke liye kaafi evidence nahi mila."
        else:
            text = "I don't have enough evidence to answer confidently. You may refer to these verses: " + suggestions if suggestions else "I don't have enough evidence to answer confidently."
        return SynthesizedResponse(
            answer=text,
            sources=[{
                "verse_id": d.verse_id,
                "citation_format": denormalize_verse_id(d.verse_id),
                "kanda_name": d.kanda_name,
                "sarga_num": d.sarga_num,
                "verse_index": d.verse_index,
                "translation_en": d.translation_en,
                "translation_hi": d.translation_hi,
                "shloka_sanskrit": d.shloka_sanskrit,
                "shloka_iast": d.shloka_iast,
                "sarga_titles": d.sarga_titles,
                "search_score": d.final_score,
            } for d in top] if documents else [],
            citations=[denormalize_verse_id(d.verse_id) for d in top if getattr(d, 'verse_id', None)],
            confidence=0.2,
            query_type="insufficient_evidence",
            llm_used=False,
            processing_time=time.time() - start_time,
        )
    
    def synthesize_count_response(
        self, count_result: Dict, response_language: str = "en"
    ) -> SynthesizedResponse:
        """Format a verified database count without asking an LLM to recalculate it."""
        answer = count_result.get("answer", "Count query processed.")
        target = count_result.get("target")
        count = count_result.get("count")
        kanda = count_result.get("kanda") or ""
        sarga = count_result.get("sarga")
        hindi_kanda = KANDA_NAMES_HI.get(str(kanda).lower(), str(kanda))

        # A zero returned for a missing Sarga is not a legitimate verse count.
        # Keep invalid-entity / invalid-Sarga responses distinct from true counts.
        unresolved_kanda = count_result.get("entity_resolved") is False
        missing_sarga = count_result.get("sarga_resolved") is False
        if unresolved_kanda or missing_sarga:
            if response_language == "hi":
                if missing_sarga and kanda and sarga:
                    answer = f"{hindi_kanda} के सर्ग {sarga} की इस डेटासेट में कोई प्रविष्टि नहीं मिली। कृपया सर्ग संख्या जाँचें।"
                else:
                    answer = "इस कांड का नाम डेटासेट में नहीं मिला। कृपया उपलब्ध कांडों के नाम जाँचें।"
            elif response_language == "hinglish":
                if missing_sarga and kanda and sarga:
                    answer = f"Is dataset mein {kanda} ke Sarga {sarga} ki entry nahi mili. Sarga number check karein."
                else:
                    answer = "Yeh Kanda is dataset mein nahi mila. Available Kanda names check karein."
            else:
                answer = count_result.get("answer", "The requested entity could not be resolved in the dataset.")
            return SynthesizedResponse(
                answer=answer,
                sources=[],
                citations=[],
                confidence=0.0,
                query_type="clarification",
                llm_used=False,
                processing_time=0.001,
            )

        if response_language == "hi":
            breakdown = count_result.get("breakdown", []) or []
            breakdown_hi = "; ".join(
                f"{KANDA_NAMES_HI.get(str(item.get('kanda_name', '')).lower(), item.get('kanda_name', ''))}: "
                f"{item.get('sarga_count', 0)} सर्ग, {int(item.get('verse_count_in_kanda', 0)):,} श्लोक"
                for item in breakdown
            )
            occurrence_name = (count_result.get("parameters") or {}).get("name", "दिए गए नाम")
            templates = {
                "kandas": lambda: f"इस डेटासेट में रामायण के {count} काण्ड हैं।",
                "verses": lambda: f"इस डेटासेट में रामायण के कुल {count:,} श्लोक हैं।",
                "sargas_total": lambda: f"इस डेटासेट में रामायण के कुल {count} सर्ग हैं।",
                "sargas": lambda: f"{hindi_kanda} में कुल {count} सर्ग हैं।",
                "verses_in_kanda": lambda: f"{hindi_kanda} में कुल {count:,} श्लोक हैं।",
                "verses_in_sarga": lambda: f"{hindi_kanda} के सर्ग {sarga} में {count} श्लोक हैं।",
                "characters_total": lambda: f"इस डेटासेट में कुल {count} पात्र दर्ज हैं।",
                "locations_total": lambda: f"इस डेटासेट में कुल {count} स्थान दर्ज हैं।",
                "occurrences": lambda: f"'{occurrence_name}' का उल्लेख {count} मेल खाते पाठ-प्रविष्टियों में मिला। एक प्रविष्टि में एक से अधिक व्यक्तिगत श्लोक हो सकते हैं।",
                "kanda_breakdown": lambda: "काण्ड-वार विवरण: " + breakdown_hi,
            }
            answer = templates[target]() if target in templates else "डेटासेट से इस गणना की पुष्टि नहीं हो सकी।"
            if count_result.get("entity_resolved") is False:
                answer = "इस कांड का नाम डेटासेट में नहीं मिला। कृपया उपलब्ध कांडों के नाम जाँचें।"
        elif response_language == "hinglish":
            templates = {
                "kandas": lambda: f"Is dataset mein Ramayana ke {count} kanda hain.",
                "verses": lambda: f"Is dataset mein Ramayana ke kul {count:,} shlok hain.",
                "sargas_total": lambda: f"Is dataset mein Ramayana ke kul {count} sarga hain.",
                "sargas": lambda: f"{kanda} mein kul {count} sarga hain.",
                "verses_in_kanda": lambda: f"{kanda} mein kul {count:,} shlok hain.",
                "verses_in_sarga": lambda: f"{kanda} ke Sarga {sarga} mein {count} shlok hain.",
                "kanda_breakdown": lambda: "Kanda-wise breakdown: " + answer,
            }
            answer = templates[target]() if target in templates else "Is count ko dataset se verify nahi kar paaya."
            if count_result.get("entity_resolved") is False:
                answer = "Yeh Kanda is dataset mein nahi mila. Available Kanda names check karein."

        return SynthesizedResponse(
            answer=answer,
            sources=[],
            citations=[],
            confidence=1.0,
            query_type="count",
            llm_used=False,
            processing_time=0.001,
        )
    
    def summarize_citation_context(
        self, citation_result: Dict[str, Any], response_language: str = "en"
    ) -> str:
        """
        Create a concise 2-3 sentence context summary for a direct verse lookup
        using the verse itself and its surrounding verses (from `citation_result['sources']`).

        Uses LLM when available; otherwise falls back to a deterministic heuristic
        based on English translations and sarga titles.
        """
        self.last_context_summary_llm_used = False
        try:
            sources: List[Dict[str, Any]] = citation_result.get("sources", []) or []
            if not sources:
                return "Context summary unavailable."

            # Main verse is first, neighbors follow
            main = sources[0]
            neighbors = sources[1:]

            # Build compact source snippets for prompt (cap to 5 total)
            selected = sources[:5]
            lines: List[str] = []
            for s in selected:
                citation = denormalize_verse_id(str(s.get("verse_id", "")))
                en = (s.get("translation_en") or "").strip()
                iast = (s.get("shloka_iast") or "").strip()

                # Prefer English for summarization, fall back to IAST if missing
                text_for_summary = en or iast
                if text_for_summary:
                    # Keep each line short
                    snippet = text_for_summary.replace("\n", " ").strip()
                    if len(snippet) > 300:
                        snippet = snippet[:297].rstrip() + "..."
                    lines.append(f"- {citation}: {snippet}")

            # LLM path
            if self.llm_client and lines:
                try:
                    main_citation = denormalize_verse_id(str(main.get("verse_id", "")))
                    kanda_name = main.get("kanda_name", "")
                    sarga_num = main.get("sarga_num", "")
                    sources_block = "\n".join(lines)
                    policy_tail = ""
                    try:
                        if getattr(settings, 'restrict_to_valmiki', False):
                            policy_tail += " Use only Valmiki Ramayana content."
                        if getattr(settings, 'disallow_uttara_kanda_mentions', False):
                            policy_tail += " Do not mention 'Uttara Kanda'."
                    except Exception:
                        pass
                    language_instruction = {
                        "hi": "Respond only in natural Hindi written in Devanagari script.",
                        "hinglish": "Respond in natural Hinglish using Roman script.",
                        "en": "Respond in clear English.",
                    }.get(response_language, "Respond in clear English.")
                    system_prompt = (
                        "You are a precise assistant for the Ramayana. "
                        "Given a verse and its immediate surrounding verses, write a brief, neutral summary "
                        "(2-3 sentences) of what is happening in this place in the narrative. "
                        "Use ONLY the provided content. Do not invent details. Do not include citations in the text. "
                        "Do not include any analysis, steps, chain-of-thought, or <think> blocks. "
                        + language_instruction + " Output ONLY the final 2-3 sentences." + policy_tail
                    )
                    user_prompt = (
                        f"Summarize the immediate context of verse {main_citation} "
                        f"({kanda_name}, Sarga {sarga_num}).\n\n"
                        f"Context verses:\n{sources_block}\n\n"
                        "Write 2-3 concise sentences."
                    )
                    summary = self.llm_client.generate(
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        query_type="narrative"
                    )
                    if summary and summary.strip():
                        clean = self._sanitize_summary_text(summary.strip())
                        clean = self._limit_sentences(clean, 3)
                        if len(clean) > 480:
                            clean = clean[:477].rstrip() + "..."
                        self.last_context_summary_llm_used = True
                        return self._apply_policy_sanitization(clean)
                except Exception as e:
                    logger.warning("LLM context summary failed (%s)", type(e).__name__)

            # Fallback heuristic summarization
            titles: List[str] = []
            try:
                if isinstance(main.get("sarga_titles"), list):
                    titles = [t for t in main.get("sarga_titles", []) if isinstance(t, str) and t.strip()]
            except Exception:
                titles = []

            title_lead = (titles[0].strip() if titles else "").rstrip('.')
            translation_key = "translation_hi" if response_language in {"hi", "hinglish"} else "translation_en"
            main_en = (main.get(translation_key) or main.get("translation_en") or "").strip()
            prev_en = next((n.get(translation_key, n.get("translation_en", "")).strip() for n in neighbors if (n.get("verse_index", 0) or 0) < (main.get("verse_index", 0) or 0) and n.get(translation_key, n.get("translation_en"))), "")
            next_en = next((n.get(translation_key, n.get("translation_en", "")).strip() for n in neighbors if (n.get("verse_index", 0) or 0) > (main.get("verse_index", 0) or 0) and n.get(translation_key, n.get("translation_en"))), "")

            parts: List[str] = []
            if title_lead:
                parts.append(f"{title_lead}.")
            if main_en:
                parts.append(main_en)
            # Add one neighbor for additional context if available
            neighbor_sentence = prev_en or next_en
            if neighbor_sentence and neighbor_sentence not in parts:
                parts.append(neighbor_sentence)

            if not parts:
                return "Context summary unavailable."

            fallback = ' '.join(p.strip() for p in parts if p).replace("\n", " ")
            if response_language == "hi":
                lead = f"संदर्भ श्लोक {denormalize_verse_id(str(main.get('verse_id', '')))} के आसपास का प्रसंग: "
                fallback = lead + fallback
            elif response_language == "hinglish":
                lead = f"Verse {denormalize_verse_id(str(main.get('verse_id', '')))} ke aas-paas ka context: "
                fallback = lead + fallback
            # Keep fallback concise
            if len(fallback) > 480:
                fallback = fallback[:477].rstrip() + "..."
            return self._apply_policy_sanitization(fallback)
        except Exception:
            return "Context summary unavailable."
    
    def get_synthesizer_stats(self) -> Dict[str, Any]:
        """
        Get synthesizer statistics
        """
        
        return {
            "llm_available": self.llm_client is not None,
            "llm_models": self.llm_client.get_stats() if self.llm_client else None,
            "supported_query_types": ["general", "relationship", "narrative", "analysis"],
            "citation_format": "[Kanda-Sarga-Verse]",
            "fallback_enabled": True
        }
