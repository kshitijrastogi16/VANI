"""
VANI V3 Citations - Citation Validation and Source Formatting
Enforcing: Comprehensive citation management with validation
"""

import re
import json
from typing import List, Dict, Set, Optional, Any


def extract_citations(text: str) -> List[str]:
    """
    Extract all verse citations from text in K.S.V or K.S.V1-V2 format
    """
    
    # Match dot-separated citations like 2.72.50 or 6.129.5-9
    citation_pattern = r'\b(\d+\.\d+\.\d+(?:-\d+)?)\b'
    return re.findall(citation_pattern, text)


def validate_citations(text: str, available_verse_ids: List[str]) -> bool:
    """
    Validate that all citations in text exist in available sources
    """
    
    found_citations = extract_citations(text)
    
    if not found_citations:
        return True  # No citations to validate
    
    # Normalize available verse IDs (convert dots to hyphens)
    normalized_available = {vid.replace('.', '-') for vid in available_verse_ids}
    
    # Check if all found citations are valid
    for citation in found_citations:
        # Convert dot-format to internal hyphen-format for comparison
        norm = normalize_verse_id(citation)
        if norm not in normalized_available:
            return False
    
    return True


def format_sources(search_results: List[Dict]) -> str:
    """
    Format search results into comprehensive source listing
    """
    
    if not search_results:
        return "No sources available."
    
    source_lines = []
    
    for i, result in enumerate(search_results, 1):
        verse_id = result.get('verse_id', 'Unknown')
        # Display citations in K.S.V[-V2] format
        citation_format = denormalize_verse_id(str(verse_id))
        
        # Basic reference (no brackets)
        source_line = f"{i}. {citation_format}"
        
        # Add location information
        kanda_name = result.get('kanda_name', 'Unknown Kanda')
        sarga_num = result.get('sarga_num', 'Unknown')
        
        source_line += f" {kanda_name}, Sarga {sarga_num}"
        
        # Add contextual information if available
        sarga_titles = result.get('sarga_titles', [])
        if sarga_titles and isinstance(sarga_titles, list):
            context = sarga_titles[0] if sarga_titles else ""
            if context:
                source_line += f" - {context}"
        
        source_lines.append(source_line)
    
    return '\n'.join(source_lines)


def create_detailed_citation(verse_data: Dict) -> Dict[str, Any]:
    """
    Create detailed citation object with all verse information
    """
    
    citation = {
        "verse_id": verse_data.get('verse_id'),
        # Human-friendly display format
        "citation_format": denormalize_verse_id(verse_data.get('verse_id', '')),
        "location": {
            "kanda_name": verse_data.get('kanda_name'),
            "sarga_num": verse_data.get('sarga_num'),
            "verse_index": verse_data.get('verse_index')
        },
        "content": {
            "sanskrit": verse_data.get('shloka_sanskrit'),
            "iast": verse_data.get('shloka_iast'), 
            "english": verse_data.get('translation_en'),
            "hindi": verse_data.get('translation_hi')
        },
        "context": {
            "sarga_titles": verse_data.get('sarga_titles', [])
        }
    }
    
    return citation


def format_citation_for_display(citation: Dict) -> str:
    """
    Format a citation for user display
    """
    
    lines = []
    
    # Header with citation and location (no brackets)
    header = f"**{citation['citation_format']}** {citation['location']['kanda_name']}, "
    header += f"Sarga {citation['location']['sarga_num']}, Verse {citation['location']['verse_index']}"
    lines.append(header)
    
    # Content
    content = citation.get('content', {})
    
    if content.get('sanskrit'):
        lines.append(f"**Sanskrit:** {content['sanskrit']}")
    
    if content.get('iast'):
        lines.append(f"**IAST:** {content['iast']}")
    
    if content.get('english'):
        lines.append(f"**English:** {content['english']}")
    
    if content.get('hindi'):
        lines.append(f"**Hindi:** {content['hindi']}")
    
    # Context
    context_titles = citation.get('context', {}).get('sarga_titles', [])
    if context_titles:
        lines.append(f"**Context:** {context_titles[0]}")
    
    return '\n\n'.join(lines)


def remove_invalid_citations(text: str, valid_verse_ids: Set[str]) -> str:
    """
    Remove invalid citations from text
    """
    
    def citation_replacer(match):
        citation = match.group(1)
        # Convert found dot-format to internal hyphen-format for validation
        norm = normalize_verse_id(citation)
        if norm in valid_verse_ids:
            return match.group(0)  # Keep valid citation
        else:
            return ""  # Remove invalid citation
    
    citation_pattern = r'\b(\d+\.\d+\.\d+(?:-\d+)?)\b'
    return re.sub(citation_pattern, citation_replacer, text)


def normalize_verse_id(verse_id: str) -> str:
    """
    Normalize verse ID format (convert dots to hyphens for citations)
    """
    return verse_id.replace('.', '-')


def denormalize_verse_id(citation: str) -> str:
    """
    Convert internal hyphenated verse_id to human display format K.S.V or K.S.V1-V2
    """
    if not citation:
        return citation
    parts = str(citation).split('-')
    try:
        if len(parts) == 3:
            # K-S-V -> K.S.V
            return f"{parts[0]}.{parts[1]}.{parts[2]}"
        if len(parts) == 4:
            # K-S-V1-V2 -> K.S.V1-V2
            return f"{parts[0]}.{parts[1]}.{parts[2]}-{parts[3]}"
        # If already dot-format or unexpected, attempt best-effort
        if '.' in citation and '-' in citation:
            return citation
        return citation.replace('-', '.')
    except Exception:
        return citation.replace('-', '.')


def validate_citation_format(citation: str) -> bool:
    """
    Validate that citation follows proper K.S.V[-V2] format
    """
    pattern = r'^\d+\.\d+\.\d+(?:-\d+)?$'
    return bool(re.match(pattern, citation))


def extract_verse_references(text: str) -> List[Dict[str, str]]:
    """
    Extract all verse references and their context from text
    """
    
    references = []
    # Match dot-format K.S.V or K.S.V1-V2
    citation_pattern = r'\b(\d+\.\d+\.\d+(?:-\d+)?)\b'
    
    for match in re.finditer(citation_pattern, text):
        citation = match.group(1)
        start_pos = match.start()
        end_pos = match.end()
        
        # Extract surrounding context (30 characters before and after)
        context_start = max(0, start_pos - 30)
        context_end = min(len(text), end_pos + 30)
        context = text[context_start:context_end].strip()
        
        references.append({
            "citation": citation,
            "position": start_pos,
            "context": context
        })
    
    return references
