"""
VANI V3 Relationship Intelligence - Character and Location Relationships
Enforcing: V1 relationship intelligence for family trees and connections
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RelationshipType(Enum):
    FAMILY = "family"
    DEVOTION = "devotion"
    ENMITY = "enmity"
    FRIENDSHIP = "friendship"
    RULER_SUBJECT = "ruler_subject"
    LOCATION = "location"


@dataclass
class Relationship:
    source_entity: str
    target_entity: str
    relationship_type: RelationshipType
    relationship_label: str
    bidirectional: bool = False
    context: List[str] = None


class RelationshipGraph:
    """
    Graph-based relationship intelligence for Ramayana entities
    Supports complex relationship queries and family tree navigation
    """
    
    def __init__(self, characters_path: str, locations_path: str):
        self.characters_path = characters_path
        self.locations_path = locations_path
        self.relationships: Dict[str, List[Relationship]] = {}
        self.entities: Dict[str, Dict] = {}
        
        self._load_relationships()
    
    def _load_relationships(self):
        """Load and process relationship data from JSON files"""
        
        # Load characters
        try:
            with open(self.characters_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)

            # Support both dict keyed by slug and list of character dicts
            if isinstance(raw, dict):
                iterable = raw.items()
            elif isinstance(raw, list):
                iterable = [(None, c) for c in raw]
            else:
                iterable = []

            count_chars = 0
            for slug, char in iterable:
                if not isinstance(char, dict):
                    continue

                canonical_name = (
                    char.get('canonical_name')
                    or char.get('canonicalname')
                    or (slug if isinstance(slug, str) else '')
                )

                if not canonical_name:
                    continue

                canonical_key = canonical_name.lower()
                self.entities[canonical_key] = char
                # Also index by dataset slug (if present) to support relation targets like 'bharata_1'
                if isinstance(slug, str) and slug.strip():
                    self.entities[slug.strip().lower()] = char
                count_chars += 1
                
                # Process family relationships
                relations = char.get('relations', {})
                if isinstance(relations, dict):
                    self._process_entity_relations(canonical_key, relations, RelationshipType.FAMILY)
                
                # Optional: Process character-linked locations if present
                locations = char.get('locations', [])
                if isinstance(locations, list):
                    for location in locations:
                        if isinstance(location, str) and location:
                            self._add_relationship(
                                canonical_key, 
                                location.lower(),
                                RelationshipType.LOCATION,
                                "associated_with"
                            )
            
            logger.info(f"Loaded relationships for {count_chars} characters")
            
        except Exception as e:
            logger.error(f"Error loading character relationships: {e}")
        
        # Load locations
        try:
            with open(self.locations_path, 'r', encoding='utf-8') as f:
                raw_locs = json.load(f)
            
            if isinstance(raw_locs, dict):
                loc_iterable = raw_locs.items()
            elif isinstance(raw_locs, list):
                loc_iterable = [(None, l) for l in raw_locs]
            else:
                loc_iterable = []

            count_locs = 0
            for slug, loc in loc_iterable:
                if not isinstance(loc, dict):
                    continue
                canonical_name = (
                    loc.get('canonical_name')
                    or loc.get('canonicalname')
                    or (slug if isinstance(slug, str) else '')
                )
                if not canonical_name:
                    continue
                self.entities[canonical_name.lower()] = loc
                # Also index by dataset slug (if present)
                if isinstance(slug, str) and slug.strip():
                    self.entities[slug.strip().lower()] = loc
                count_locs += 1
            
            logger.info(f"Loaded {count_locs} locations")
            
        except Exception as e:
            logger.error(f"Error loading locations: {e}")
    
    def _process_entity_relations(self, entity_name: str, relations: Dict, rel_type: RelationshipType):
        """Process relationship data for a single entity"""
        
        # Map relation labels to their inverse for non-bidirectional relations
        family_mappings = {
            'father': ('father', 'child'),
            'mother': ('mother', 'child'),
            'parent': ('parent', 'child'),
            'son': ('child', 'parent'),
            'daughter': ('child', 'parent'),
            'child': ('child', 'parent'),
            'husband': ('husband', 'wife'),
            'wife': ('wife', 'husband'),
            'guru': ('guru', 'disciple'),
            'disciple': ('disciple', 'guru'),
            'friend': ('friend', 'friend'),
        }

        bidirectional_labels = {'brother', 'sister', 'sibling', 'spouse', 'friend'}

        def _add(entity_target: str, label: str, bidi: bool = False):
            if not entity_target or not isinstance(entity_target, str):
                return
            target_clean = entity_target.strip().lower()
            if not target_clean:
                return
            if bidi:
                self._add_relationship(
                    entity_name,
                    target_clean,
                    rel_type,
                    label,
                    bidirectional=True
                )
            else:
                self._add_relationship(
                    entity_name,
                    target_clean,
                    rel_type,
                    label,
                    bidirectional=False
                )
                # Add inverse if mapping available
                inverse_rel = family_mappings.get(label, (None, None))[1]
                if inverse_rel:
                    self._add_relationship(
                        target_clean,
                        entity_name,
                        rel_type,
                        inverse_rel,
                        bidirectional=False
                    )

        # Handle parents
        parents = relations.get('parents', {})
        if isinstance(parents, dict):
            for p_label in ('father', 'mother', 'parent'):
                val = parents.get(p_label)
                if not val:
                    continue
                if isinstance(val, list):
                    for v in val:
                        _add(v, p_label)
                elif isinstance(val, str):
                    _add(val, p_label)

        # Handle children
        children = relations.get('children', {})
        if isinstance(children, dict):
            for c_label, vals in children.items():
                label_norm = {'sons': 'son', 'daughters': 'daughter', 'children': 'child'}.get(c_label, c_label)
                if isinstance(vals, list):
                    for v in vals:
                        _add(v, label_norm)
                elif isinstance(vals, str):
                    _add(vals, label_norm)

        # Handle siblings
        siblings = relations.get('siblings', {})
        if isinstance(siblings, dict):
            for s_label, vals in siblings.items():
                label_norm = {'brothers': 'brother', 'sisters': 'sister', 'siblings': 'sibling'}.get(s_label, s_label)
                if isinstance(vals, list):
                    for v in vals:
                        _add(v, label_norm, bidi=True)
                elif isinstance(vals, str):
                    _add(vals, label_norm, bidi=True)

        # Handle spouses
        spouses = relations.get('spouses') or relations.get('spouse')
        if isinstance(spouses, list):
            for v in spouses:
                _add(v, 'spouse', bidi=True)
        elif isinstance(spouses, str):
            _add(spouses, 'spouse', bidi=True)

        # Generic pass-through for other simple relation categories (e.g., guru, disciple)
        for rel_key, rel_data in relations.items():
            if rel_key in {'parents', 'children', 'siblings', 'spouses', 'spouse'}:
                continue
            if isinstance(rel_data, list):
                for v in rel_data:
                    _add(v, rel_key, bidi=rel_key in bidirectional_labels)
            elif isinstance(rel_data, dict):
                for sub_rel, target in rel_data.items():
                    if isinstance(target, list):
                        for v in target:
                            _add(v, sub_rel, bidi=sub_rel in bidirectional_labels)
                    else:
                        _add(target, sub_rel, bidi=sub_rel in bidirectional_labels)
            elif isinstance(rel_data, str):
                _add(rel_data, rel_key, bidi=rel_key in bidirectional_labels)
    
    def _add_relationship(self, source: str, target: str, rel_type: RelationshipType, label: str, bidirectional: bool = False):
        """Add a relationship to the graph with normalized entity keys.
        - Source/target keys are normalized to canonical-name keys when known.
        - Prevents leakage of dataset slugs (e.g., 'bharata_1') in downstream answers.
        """

        def _norm_key(name: str) -> str:
            try:
                k = (name or '').strip().lower()
                if not k:
                    return k
                info = self.entities.get(k)
                if isinstance(info, dict):
                    can = (info.get('canonical_name') or info.get('canonicalname') or '').strip().lower()
                    return can if can else k
                return k
            except Exception:
                return (name or '').strip().lower()

        src = _norm_key(source)
        tgt = _norm_key(target)

        if src not in self.relationships:
            self.relationships[src] = []

        relationship = Relationship(
            source_entity=src,
            target_entity=tgt,
            relationship_type=rel_type,
            relationship_label=label,
            bidirectional=bidirectional
        )

        self.relationships[src].append(relationship)

        # Add bidirectional relationship
        if bidirectional:
            if tgt not in self.relationships:
                self.relationships[tgt] = []

            reverse_relationship = Relationship(
                source_entity=tgt,
                target_entity=src,
                relationship_type=rel_type,
                relationship_label=label,
                bidirectional=True
            )

            self.relationships[tgt].append(reverse_relationship)
    
    def get_relationships(self, entity_name: str, rel_type: Optional[RelationshipType] = None) -> List[Relationship]:
        """Get all relationships for an entity"""
        
        entity_key = entity_name.lower()
        if entity_key not in self.relationships:
            return []
        
        relationships = self.relationships[entity_key]
        
        if rel_type:
            relationships = [r for r in relationships if r.relationship_type == rel_type]
        
        return relationships
    
    def find_family_members(self, entity_name: str) -> Dict[str, List[str]]:
        """Get family tree for an entity"""
        
        family_rels = self.get_relationships(entity_name, RelationshipType.FAMILY)
        
        family_tree = {}
        for rel in family_rels:
            label = rel.relationship_label
            if label not in family_tree:
                family_tree[label] = []
            family_tree[label].append(rel.target_entity)
        
        return family_tree
    
    def find_path(self, source: str, target: str, max_depth: int = 3) -> Optional[List[str]]:
        """Find relationship path between two entities"""
        
        if source.lower() == target.lower():
            return [source]
        
        visited = set()
        queue = [(source.lower(), [source.lower()])]
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            if current in visited:
                continue
                
            visited.add(current)
            
            if current == target.lower():
                return path
            
            # Get all relationships for current entity
            for rel in self.get_relationships(current):
                next_entity = rel.target_entity
                if next_entity not in visited:
                    new_path = path + [next_entity]
                    queue.append((next_entity, new_path))
        
        return None
    
    def get_entity_info(self, entity_name: str) -> Optional[Dict]:
        """Get complete entity information"""
        
        entity_key = entity_name.lower()
        return self.entities.get(entity_key)
    
    # ----------------------------
    # Deterministic family inference
    # ----------------------------
    def _canonicalize_names(self, names: List[str]) -> List[str]:
        """Map internal lowercase keys to canonical_name for display/search."""
        canonical: List[str] = []
        for n in names:
            info = self.get_entity_info(n)
            if isinstance(info, dict) and info.get('canonical_name'):
                canonical.append(info['canonical_name'])
            else:
                # Fallback to best-effort capitalization
                canonical.append(n.title())
        # Deduplicate while preserving order
        seen: Set[str] = set()
        uniq: List[str] = []
        for c in canonical:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq
    
    def _direct_relatives(self, entity_name: str, label: str) -> List[str]:
        """Return list of direct relatives for a given label (internal keys)."""
        fam = self.find_family_members(entity_name)
        # Normalize label variants
        normalize = {
            'sons': 'son', 'daughters': 'daughter', 'children': 'child',
            'brothers': 'brother', 'sisters': 'sister', 'siblings': 'sibling',
            'parents': 'parent', 'spouses': 'spouse'
        }
        lab = normalize.get(label.lower(), label.lower())
        vals = fam.get(lab, [])
        return list(dict.fromkeys([v.lower() for v in vals if isinstance(v, str) and v]))
    
    def infer_family(self, entity_name: str, relation: str) -> List[str]:
        """Infer family members for direct and grand- relations.
        Supported relations: father, mother, parent(s), son, daughter, child(ren),
        brother, sister, sibling(s), husband, wife, spouse,
        grandfather, grandmother, grandparent(s), grandson, granddaughter, grandchild(ren).
        Returns a list of canonical names.
        """
        if not entity_name or not relation:
            return []
        subject = entity_name.lower()
        rel = relation.strip().lower()
        # Direct relations
        direct_labels = {
            'father', 'mother', 'parent', 'parents',
            'son', 'daughter', 'child', 'children',
            'brother', 'sister', 'sibling', 'siblings',
            'husband', 'wife', 'spouse', 'spouses'
        }
        if rel in direct_labels:
            # Aggregate across gendered variants to ensure completeness
            results: List[str] = []
            if rel in {'parent', 'parents'}:
                results.extend(self._direct_relatives(subject, 'father'))
                results.extend(self._direct_relatives(subject, 'mother'))
                results.extend(self._direct_relatives(subject, 'parent'))
            elif rel in {'child', 'children'}:
                results.extend(self._direct_relatives(subject, 'son'))
                results.extend(self._direct_relatives(subject, 'daughter'))
                results.extend(self._direct_relatives(subject, 'child'))
            elif rel in {'sibling', 'siblings'}:
                results.extend(self._direct_relatives(subject, 'brother'))
                results.extend(self._direct_relatives(subject, 'sister'))
                results.extend(self._direct_relatives(subject, 'sibling'))
            elif rel in {'spouse', 'spouses'}:
                # Include gendered spouse relations when present
                results.extend(self._direct_relatives(subject, 'spouse'))
                results.extend(self._direct_relatives(subject, 'husband'))
                results.extend(self._direct_relatives(subject, 'wife'))
            else:
                results.extend(self._direct_relatives(subject, rel))
            return self._canonicalize_names(results)
        # Grand relations
        if rel in {'grandfather', 'grandmother', 'grandparent', 'grandparents'}:
            # parents' parents
            parents = self._direct_relatives(subject, 'father') + self._direct_relatives(subject, 'mother')
            if not parents:
                parents = self._direct_relatives(subject, 'parent')
            grands: List[str] = []
            for p in parents:
                grands.extend(self._direct_relatives(p, 'father'))
                grands.extend(self._direct_relatives(p, 'mother'))
                # Some datasets may only have 'parent'
                grands.extend(self._direct_relatives(p, 'parent'))
            # If specific gender requested, filter
            if rel == 'grandfather':
                # best-effort: keep those who are listed as father/son somewhere (no gender in schema)
                # We cannot reliably disambiguate gender for grandparents; keep as is
                pass
            if rel == 'grandmother':
                pass
            return self._canonicalize_names(grands)
        if rel in {'grandson', 'granddaughter', 'grandchild', 'grandchildren'}:
            # children's children
            children = self._direct_relatives(subject, 'son') + self._direct_relatives(subject, 'daughter')
            if not children:
                children = self._direct_relatives(subject, 'child')
            grands: List[str] = []
            for c in children:
                grands.extend(self._direct_relatives(c, 'son'))
                grands.extend(self._direct_relatives(c, 'daughter'))
                grands.extend(self._direct_relatives(c, 'child'))
            return self._canonicalize_names(grands)
        return []
    
    def search_relationships(self, query: str) -> List[Dict]:
        """Search for relationships matching a query"""
        
        results = []
        query_lower = query.lower()
        
        # Search through relationship labels and entity names
        for source, relationships in self.relationships.items():
            for rel in relationships:
                if (query_lower in rel.relationship_label.lower() or 
                    query_lower in rel.source_entity.lower() or
                    query_lower in rel.target_entity.lower()):
                    
                    results.append({
                        'source': rel.source_entity,
                        'target': rel.target_entity,
                        'relationship': rel.relationship_label,
                        'type': rel.relationship_type.value,
                        'bidirectional': rel.bidirectional
                    })
        
        return results
