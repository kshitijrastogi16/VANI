"""
VANI V3 Entity Resolver - V1 Advanced Intelligence RESTORED
Enforcing: Bidirectional relationship intelligence and cultural normalization
"""

import os
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from rapidfuzz import fuzz, process
import logging

logger = logging.getLogger(__name__)

@dataclass
class ResolvedEntity:
    canonical_name: str
    entity_id: str
    entity_type: str
    confidence: float
    match_type: str
    original_query_term: str
    aliases: List[str]
    context: List[str]


class EntityResolver:
    """
    V1 Advanced Entity Resolution with Cultural Intelligence
    Features: Bidirectional patterns, Sanskrit normalization, relationship awareness
    """
    
    def __init__(self, characters_path: str, locations_path: str, confidence_threshold: float = 0.7):
        self.characters_path = characters_path
        self.locations_path = locations_path
        self.confidence_threshold = confidence_threshold
        
        # Core data structures (V1 approach)
        self.canonical_lookup: Dict[str, Dict] = {}
        self.alias_lookup: Dict[str, str] = {}
        self.context_index: Dict[str, List[str]] = {}
        self.entity_data: Dict[str, Dict] = {}
        self.relationship_patterns: Dict[str, List[str]] = {}
        
        # Priority boosting for main characters
        self.main_characters = {
            'rama', 'sita', 'hanuman', 'ravana', 'lakshmana', 'bharata', 
            'shatrughna', 'dasaratha', 'vali', 'sugriva', 'vibhishana'
        }
        
        self._load_entities()
        self._build_relationship_patterns()
    
    def _normalize_text(self, text: str) -> str:
        """Advanced normalization with Sanskrit/Hindi support (V1 feature)"""
        if not text:
            return ""
        
        text = text.lower().strip()
        
        # Sanskrit diacritic normalization (V1 cultural intelligence)
        diacritic_map = {
            'ā': 'a', 'ī': 'i', 'ū': 'u', 'ṛ': 'r', 'ḷ': 'l',
            'ṅ': 'n', 'ñ': 'n', 'ṇ': 'n', 'ṃ': 'm', 'ḥ': 'h',
            'ṭ': 't', 'ḍ': 'd', 'ś': 's', 'ṣ': 's'
        }
        
        for diacritic, base in diacritic_map.items():
            text = text.replace(diacritic, base)
        
        # Clean and normalize
        text = re.sub(r"[^\w\s\-']", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def _load_entities(self):
        """Load character and location data with enhanced indexing"""
        
        # Helper to normalize a raw record's keys to expected names
        def _normalize_record(rec: dict) -> dict:
            # Support both snake_case and camel/case-insensitive keys
            def pick(*names, default=''):
                for n in names:
                    if n in rec and rec[n] is not None:
                        return rec[n]
                    # try case-insensitive
                    for k in rec.keys():
                        if k.lower() == n.lower():
                            return rec[k]
                return default

            normalized = {
                'id': pick('id'),
                'canonicalname': pick('canonical_name', 'canonicalname', default='').strip(),
                'entitytype': pick('entity_type', 'entitytype', default='').strip(),
                'aliases': pick('aliases', default=[]) or [],
                'disambiguationcontext': pick('disambiguation_context', 'disambiguationcontext', default=[]) or [],
                'relations': pick('relations', default={}) or {}
            }
            # Coerce types
            if not isinstance(normalized['aliases'], list):
                normalized['aliases'] = [str(normalized['aliases'])] if normalized['aliases'] else []
            if not isinstance(normalized['disambiguationcontext'], list):
                normalized['disambiguationcontext'] = [str(normalized['disambiguationcontext'])] if normalized['disambiguationcontext'] else []
            if not isinstance(normalized['relations'], dict):
                normalized['relations'] = {}
            return normalized

        # Load characters
        try:
            with open(self.characters_path, 'r', encoding='utf-8') as f:
                raw_chars = json.load(f)

            # Accept dict-of-records or list-of-records
            if isinstance(raw_chars, dict):
                records = list(raw_chars.values())
            else:
                records = list(raw_chars)

            for raw in records:
                char = _normalize_record(raw)
                canonical_name = char.get('canonicalname', '').strip()
                if not canonical_name:
                    continue

                norm_name = self._normalize_text(canonical_name)

                # Store entity data
                self.entity_data[norm_name] = char
                self.canonical_lookup[norm_name] = char

                # Index aliases
                for alias in char.get('aliases', []):
                    if isinstance(alias, str) and alias.strip():
                        norm_alias = self._normalize_text(alias.strip())
                        self.alias_lookup[norm_alias] = norm_name

                # Index context phrases
                for context in char.get('disambiguationcontext', []):
                    if isinstance(context, str) and context.strip():
                        norm_context = self._normalize_text(context.strip())
                        if norm_context not in self.context_index:
                            self.context_index[norm_context] = []
                        self.context_index[norm_context].append(norm_name)

                logger.debug(f"Loaded character: {canonical_name}")

        except Exception as e:
            logger.error(f"Error loading characters: {e}")
            raise
        
        # Load locations
        try:
            with open(self.locations_path, 'r', encoding='utf-8') as f:
                raw_locs = json.load(f)

            if isinstance(raw_locs, dict):
                records = list(raw_locs.values())
            else:
                records = list(raw_locs)

            for raw in records:
                loc = _normalize_record(raw)
                canonical_name = loc.get('canonicalname', '').strip()
                if not canonical_name:
                    continue

                norm_name = self._normalize_text(canonical_name)

                # Store entity data
                self.entity_data[norm_name] = loc
                self.canonical_lookup[norm_name] = loc

                # Index aliases
                for alias in loc.get('aliases', []):
                    if isinstance(alias, str) and alias.strip():
                        norm_alias = self._normalize_text(alias.strip())
                        self.alias_lookup[norm_alias] = norm_name

                # Index context phrases
                for context in loc.get('disambiguationcontext', []):
                    if isinstance(context, str) and context.strip():
                        norm_context = self._normalize_text(context.strip())
                        if norm_context not in self.context_index:
                            self.context_index[norm_context] = []
                        self.context_index[norm_context].append(norm_name)

                logger.debug(f"Loaded location: {canonical_name}")

        except Exception as e:
            logger.error(f"Error loading locations: {e}")
            raise
        
        logger.info(f"✅ Loaded {len(self.canonical_lookup)} entities total")
    
    def _build_relationship_patterns(self):
        """Build bidirectional relationship patterns (V1 intelligence)"""
        
        # Normalize relation labels to a consistent vocabulary
        def _norm_label(label: str) -> str:
            m = (label or '').strip().lower()
            m = {
                'sons': 'son', 'son': 'son',
                'daughters': 'daughter', 'daughter': 'daughter',
                'children': 'child', 'child': 'child',
                'parents': 'parent', 'parent': 'parent',
                'fathers': 'father', 'father': 'father',
                'mothers': 'mother', 'mother': 'mother',
                'spouses': 'spouse', 'spouse': 'spouse',
                'brothers': 'brother', 'brother': 'brother',
                'sisters': 'sister', 'sister': 'sister',
                'siblings': 'sibling', 'sibling': 'sibling',
            }.get(m, m)
            return m

        # Inverse mapping for common family labels
        inverse_map = {
            'father': 'child',
            'mother': 'child',
            'parent': 'child',
            'son': 'parent',
            'daughter': 'parent',
            'child': 'parent',
            'husband': 'wife',
            'wife': 'husband',
            'spouse': 'spouse',
            'brother': 'brother',
            'sister': 'sister',
            'sibling': 'sibling',
        }

        for norm_name, entity in self.entity_data.items():
            relations = entity.get('relations', {})
            # Iterate through relation categories with normalization
            for rel_cat, rel_value in relations.items():
                if rel_cat in {'parents', 'children', 'siblings'} and isinstance(rel_value, dict):
                    # Nested dict case e.g., parents: {father: X}
                    for sub_rel, target in rel_value.items():
                        label = _norm_label(sub_rel)
                        names: List[str] = target if isinstance(target, list) else [target]
                        for relative_name in [n for n in names if isinstance(n, str) and n.strip()]:
                            norm_relative = self._normalize_text(relative_name)
                            # Subject-based patterns: "<subject>'s <label>" -> target (relative)
                            subject_patterns = [
                                f"{norm_name}'s {label}",
                                f"{label} of {norm_name}",
                                f"{norm_name} {label}",
                                f"{label} {norm_name}",
                            ]
                            for pattern in subject_patterns:
                                if pattern not in self.relationship_patterns:
                                    self.relationship_patterns[pattern] = []
                                # Map to the relative entity (answer)
                                self.relationship_patterns[pattern].append(norm_relative)

                            # Inverse-based patterns: "<inverse> of <relative>" -> subject (norm_name)
                            inv = inverse_map.get(label)
                            if inv:
                                inverse_patterns = [
                                    f"{norm_relative}'s {inv}",
                                    f"{inv} of {norm_relative}",
                                    f"{norm_relative} {inv}",
                                    f"{inv} {norm_relative}",
                                ]
                                for pattern in inverse_patterns:
                                    if pattern not in self.relationship_patterns:
                                        self.relationship_patterns[pattern] = []
                                    # Map to the subject entity (answer)
                                    self.relationship_patterns[pattern].append(norm_name)
                else:
                    # Simple list or string case e.g., spouses: [Y]
                    label = _norm_label(rel_cat)
                    names: List[str] = []
                    if isinstance(rel_value, list):
                        names = [n for n in rel_value if isinstance(n, str) and n.strip()]
                    elif isinstance(rel_value, str) and rel_value:
                        names = [rel_value]
                    elif isinstance(rel_value, dict):
                        # Unexpected dict; flatten conservatively
                        for v in rel_value.values():
                            if isinstance(v, list):
                                names.extend([n for n in v if isinstance(n, str) and n.strip()])
                            elif isinstance(v, str) and v:
                                names.append(v)

                    for relative_name in names:
                        norm_relative = self._normalize_text(relative_name)
                        # Subject-based patterns map to the relative
                        subject_patterns = [
                            f"{norm_name}'s {label}",
                            f"{label} of {norm_name}",
                            f"{norm_name} {label}",
                            f"{label} {norm_name}",
                        ]
                        for pattern in subject_patterns:
                            if pattern not in self.relationship_patterns:
                                self.relationship_patterns[pattern] = []
                            self.relationship_patterns[pattern].append(norm_relative)

                        # Inverse patterns map to the subject
                        inv = inverse_map.get(label)
                        if inv:
                            inverse_patterns = [
                                f"{norm_relative}'s {inv}",
                                f"{inv} of {norm_relative}",
                                f"{norm_relative} {inv}",
                                f"{inv} {norm_relative}",
                            ]
                            for pattern in inverse_patterns:
                                if pattern not in self.relationship_patterns:
                                    self.relationship_patterns[pattern] = []
                                self.relationship_patterns[pattern].append(norm_name)

        logger.info(f"✅ Built {len(self.relationship_patterns)} relationship patterns")
    
    def resolve(self, query_term: str) -> Optional[ResolvedEntity]:
        """
        Main resolution method with V1 intelligence
        Priority: Exact → Alias → Relationship → Fuzzy
        """
        
        if not query_term or not query_term.strip():
            return None
        
        norm_term = self._normalize_text(query_term)
        
        # 1. Exact canonical match (highest confidence)
        if norm_term in self.canonical_lookup:
            entity = self.canonical_lookup[norm_term]
            confidence = 1.0
            
            # Boost main characters
            if norm_term in self.main_characters:
                confidence = 1.0
            
            return ResolvedEntity(
                canonical_name=entity['canonicalname'],
                entity_id=entity.get('id', ''),
                entity_type=entity.get('entitytype', ''),
                confidence=confidence,
                match_type='exact_canonical',
                original_query_term=query_term,
                aliases=entity.get('aliases', []),
                context=entity.get('disambiguationcontext', [])
            )
        
        # 2. Exact alias match
        if norm_term in self.alias_lookup:
            canonical = self.alias_lookup[norm_term]
            entity = self.canonical_lookup.get(canonical)
            
            if entity:
                confidence = 0.95
                
                # Boost main characters
                if canonical in self.main_characters:
                    confidence = 0.98
                
                return ResolvedEntity(
                    canonical_name=entity['canonicalname'],
                    entity_id=entity.get('id', ''),
                    entity_type=entity.get('entitytype', ''),
                    confidence=confidence,
                    match_type='exact_alias',
                    original_query_term=query_term,
                    aliases=entity.get('aliases', []),
                    context=entity.get('disambiguationcontext', [])
                )
        
        # 3. Relationship pattern match (V1 intelligence)
        for pattern, entity_names in self.relationship_patterns.items():
            if norm_term in pattern or pattern in norm_term:
                # Use best matching entity
                for entity_name in entity_names:
                    entity = self.canonical_lookup.get(entity_name)
                    if entity:
                        confidence = 0.85
                        
                        # Boost main characters
                        if entity_name in self.main_characters:
                            confidence = 0.90
                        
                        return ResolvedEntity(
                            canonical_name=entity['canonicalname'],
                            entity_id=entity.get('id', ''),
                            entity_type=entity.get('entitytype', ''),
                            confidence=confidence,
                            match_type='relationship_pattern',
                            original_query_term=query_term,
                            aliases=entity.get('aliases', []),
                            context=entity.get('disambiguationcontext', [])
                        )
        
        # 4. Context-based match
        for context, entity_names in self.context_index.items():
            if norm_term in context or context in norm_term:
                for entity_name in entity_names:
                    entity = self.canonical_lookup.get(entity_name)
                    if entity:
                        return ResolvedEntity(
                            canonical_name=entity['canonicalname'],
                            entity_id=entity.get('id', ''),
                            entity_type=entity.get('entitytype', ''),
                            confidence=0.80,
                            match_type='context_match',
                            original_query_term=query_term,
                            aliases=entity.get('aliases', []),
                            context=entity.get('disambiguationcontext', [])
                        )
        
        # 5. Fuzzy match with prioritization
        candidates = []
        candidates.extend(list(self.canonical_lookup.keys()))
        candidates.extend(list(self.alias_lookup.keys()))
        
        matches = process.extract(norm_term, candidates, scorer=fuzz.ratio, limit=5)
        
        # RapidFuzz returns (choice, score, index); unpack safely
        for match, score, *_ in matches:
            if score < 75:  # Minimum threshold
                continue
            
            # Get entity
            canonical = self.alias_lookup.get(match, match)
            entity = self.canonical_lookup.get(canonical)
            
            if not entity:
                continue
            
            confidence = score / 100.0
            
            # Apply entity type and importance boosting
            entity_type = entity.get('entitytype', '')
            
            # Boost main characters
            if canonical in self.main_characters:
                confidence = min(confidence + 0.1, 1.0)
            
            # Boost humans over other entity types
            if entity_type == 'human':
                confidence = min(confidence + 0.05, 1.0)
            
            # Only return if above threshold
            if confidence >= self.confidence_threshold:
                return ResolvedEntity(
                    canonical_name=entity['canonicalname'],
                    entity_id=entity.get('id', ''),
                    entity_type=entity.get('entitytype', ''),
                    confidence=confidence,
                    match_type='fuzzy',
                    original_query_term=query_term,
                    aliases=entity.get('aliases', []),
                    context=entity.get('disambiguationcontext', [])
                )
        
        # No match found
        logger.debug(f"No entity resolved for: {query_term}")
        return None
    
    def resolve_multiple(self, text: str) -> List[ResolvedEntity]:
        """Resolve multiple entities from a text string"""
        
        entities: List[ResolvedEntity] = []
        words = text.split()

        # Collect candidates across n-grams
        for n in range(3, 0, -1):  # 3-gram, 2-gram, 1-gram
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i + n])
                resolved = self.resolve(phrase)
                if resolved and resolved.confidence >= self.confidence_threshold:
                    entities.append(resolved)

        if not entities:
            return []

        # Deduplicate by canonical name keeping the best (by match priority then confidence)
        # Higher is better
        priority_order = {
            'exact_canonical': 5,
            'exact_alias': 4,
            'relationship_pattern': 3,
            'context_match': 2,
            'fuzzy': 1,
        }

        best_by_canon: Dict[str, ResolvedEntity] = {}
        for ent in entities:
            key = ent.canonical_name
            cur = best_by_canon.get(key)
            if cur is None:
                best_by_canon[key] = ent
                continue
            cur_rank = (priority_order.get(cur.match_type, 0), float(cur.confidence or 0.0))
            new_rank = (priority_order.get(ent.match_type, 0), float(ent.confidence or 0.0))
            if new_rank > cur_rank:
                best_by_canon[key] = ent

        # Sort by match priority desc, then confidence desc, with a small boost for main characters
        def sort_key(ent: ResolvedEntity):
            pri = priority_order.get(ent.match_type, 0)
            conf = float(ent.confidence or 0.0)
            # Small deterministic boost for main characters to break ties
            main_boost = 0.05 if self._normalize_text(ent.canonical_name) in self.main_characters else 0.0
            return (-pri, -(conf + main_boost))

        ordered = sorted(best_by_canon.values(), key=sort_key)
        return ordered
    
    def get_entity_info(self, entity_id: str) -> Optional[Dict]:
        """Get complete entity information by ID"""
        
        for entity in self.entity_data.values():
            if entity.get('id') == entity_id:
                return entity
        return None
