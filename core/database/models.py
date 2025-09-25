"""
VANI V3 Database Models - Pydantic Models for Data Validation
Enforcing: Type safety and data validation for all database operations
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import json
from app.settings import settings


class VerseModel(BaseModel):
    """Pydantic model for verse data validation"""
    
    id: Optional[int] = None
    verse_id: str = Field(..., pattern=r'^\d+-\d+-\d+(?:-\d+)?$')
    kanda_num: int = Field(..., ge=1, le=7)
    sarga_num: int = Field(..., ge=1)
    verse_index: int = Field(..., ge=1)
    kanda_name: str = Field(..., min_length=1)
    
    # Verse content
    shloka_sanskrit: str = Field(default="")
    shloka_iast: str = Field(default="")
    translation_en: str = Field(default="")
    translation_hi: str = Field(default="")
    
    # Metadata
    sarga_titles_json: str = Field(default="[]")
    
    @validator('sarga_titles_json')
    def validate_json_field(cls, v):
        """Ensure sarga_titles_json is valid JSON"""
        try:
            json.loads(v)
            return v
        except json.JSONDecodeError:
            return "[]"
    
    class Config:
        from_attributes = True


class KandaModel(BaseModel):
    """Kanda (book) information model"""
    
    kanda_num: int = Field(..., ge=1, le=7)
    kanda_name: str = Field(..., min_length=1)
    sarga_count: int = Field(..., ge=1)
    verse_count_in_kanda: int = Field(..., ge=1)
    
    class Config:
        from_attributes = True


class SargaModel(BaseModel):
    """Sarga (chapter) information model"""
    
    kanda_num: int = Field(..., ge=1, le=7)
    sarga_num: int = Field(..., ge=1)
    verse_count_in_sarga: int = Field(..., ge=1)
    titles: List[str] = Field(default_factory=list)
    
    class Config:
        from_attributes = True


class EntityModel(BaseModel):
    """Character/Location entity model"""
    
    id: str = Field(..., min_length=1)
    canonical_name: str = Field(..., min_length=1)
    entity_type: str = Field(..., pattern=r'^(human|divine|rakshasa|vanara|asura|bird|bear|kingdom|mountain|forest|river|village|garden)$')
    aliases: List[str] = Field(default_factory=list)
    description: str = Field(default="")
    disambiguation_context: List[str] = Field(default_factory=list, alias='disambiguationcontext')
    relations: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        from_attributes = True
        populate_by_name = True


class SearchResultModel(BaseModel):
    """Search result model for API responses"""
    
    verse_id: str
    kanda_name: str
    sarga_num: int
    verse_index: int
    translation_en: str
    translation_hi: str
    shloka_sanskrit: str
    shloka_iast: str
    sarga_titles: List[str]
    score: float = Field(..., ge=0.0, le=1.0)
    search_type: str
    
    class Config:
        from_attributes = True


class QueryRequestModel(BaseModel):
    """API query request model"""
    
    question: str = Field(..., min_length=1, max_length=1000)
    include_context: bool = Field(default=True)
    max_results: int = Field(default=10, ge=1, le=50)
    query_type: Optional[str] = Field(default=None, pattern=r'^(general|count|citation|relationship)$')
    
    @validator('question')
    def validate_question(cls, v):
        """Basic question validation"""
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class QueryResponseModel(BaseModel):
    """API query response model"""
    
    answer: str
    sources: List[Dict[str, Any]]
    citations: List[str]
    query_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    response_time: float = Field(..., ge=0.0)
    llm_used: bool = Field(default=False)
    
    class Config:
        from_attributes = True


class HealthStatusModel(BaseModel):
    """Health check response model"""
    
    status: str = Field(..., pattern=r'^(healthy|degraded|unhealthy)$')
    version: str
    timestamp: float
    components: Dict[str, bool]
    details: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True


# Database table creation helpers
def get_create_table_sql() -> Dict[str, str]:
    """Get SQL statements for table creation"""
    
    tables = {
        'kandas': """
        CREATE TABLE IF NOT EXISTS kandas (
            kanda_num INTEGER PRIMARY KEY,
            kanda_name TEXT NOT NULL,
            sarga_count INTEGER NOT NULL,
            verse_count_in_kanda INTEGER NOT NULL
        )""",
        
        'sargas': """
        CREATE TABLE IF NOT EXISTS sargas (
            kanda_num INTEGER NOT NULL,
            sarga_num INTEGER NOT NULL,
            verse_count_in_sarga INTEGER NOT NULL,
            titles_json TEXT DEFAULT '[]',
            PRIMARY KEY (kanda_num, sarga_num),
            FOREIGN KEY (kanda_num) REFERENCES kandas (kanda_num)
        )""",
        
        'verses': """
        CREATE TABLE IF NOT EXISTS verses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            verse_id TEXT UNIQUE NOT NULL,
            kanda_num INTEGER NOT NULL,
            sarga_num INTEGER NOT NULL,
            verse_index INTEGER NOT NULL,
            kanda_name TEXT NOT NULL,
            shloka_sanskrit TEXT DEFAULT '',
            shloka_iast TEXT DEFAULT '',
            translation_en TEXT DEFAULT '',
            translation_hi TEXT DEFAULT '',
            sarga_titles_json TEXT DEFAULT '[]',
            FOREIGN KEY (kanda_num, sarga_num) REFERENCES sargas (kanda_num, sarga_num)
        )""",
        
        'verses_fts': """
        CREATE VIRTUAL TABLE IF NOT EXISTS verses_fts USING fts5(
            verse_id UNINDEXED,
            kanda_name UNINDEXED,
            shloka_iast,
            translation_en,
            translation_hi,
            sarga_titles_json UNINDEXED,
            content='verses',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics=2'
        )"""
    }
    
    return tables


def get_create_indexes_sql() -> List[str]:
    """Get SQL statements for index creation"""
    
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_verses_kanda_sarga ON verses (kanda_num, sarga_num)",
        "CREATE INDEX IF NOT EXISTS idx_verses_verse_id ON verses (verse_id)",
        "CREATE INDEX IF NOT EXISTS idx_verses_kanda_name ON verses (kanda_name)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_verses_coordinates ON verses (kanda_num, sarga_num, verse_index)"
    ]
    
    return indexes


def get_create_triggers_sql() -> List[str]:
    """Get SQL statements for FTS trigger creation, honoring settings for Hindi column."""
    remove_hi = getattr(settings, 'fts_remove_hi', False)
    if remove_hi:
        triggers = [
            """
            CREATE TRIGGER IF NOT EXISTS verses_fts_insert AFTER INSERT ON verses BEGIN
                INSERT INTO verses_fts(
                    rowid, verse_id, kanda_name, shloka_iast, 
                    translation_en, sarga_titles_json
                ) VALUES (
                    new.id, new.verse_id, new.kanda_name, 
                    new.shloka_iast, new.translation_en, 
                    new.sarga_titles_json
                );
            END""",
            
            """
            CREATE TRIGGER IF NOT EXISTS verses_fts_delete AFTER DELETE ON verses BEGIN
                DELETE FROM verses_fts WHERE rowid = old.id;
            END""",
            
            """
            CREATE TRIGGER IF NOT EXISTS verses_fts_update AFTER UPDATE ON verses BEGIN
                DELETE FROM verses_fts WHERE rowid = old.id;
                INSERT INTO verses_fts(
                    rowid, verse_id, kanda_name, shloka_iast, 
                    translation_en, sarga_titles_json
                ) VALUES (
                    new.id, new.verse_id, new.kanda_name, 
                    new.shloka_iast, new.translation_en,
                    new.sarga_titles_json
                );
            END"""
        ]
    else:
        triggers = [
            """
            CREATE TRIGGER IF NOT EXISTS verses_fts_insert AFTER INSERT ON verses BEGIN
                INSERT INTO verses_fts(
                    rowid, verse_id, kanda_name, shloka_iast, 
                    translation_en, translation_hi, sarga_titles_json
                ) VALUES (
                    new.id, new.verse_id, new.kanda_name, 
                    new.shloka_iast, new.translation_en, new.translation_hi, 
                    new.sarga_titles_json
                );
            END""",
            
            """
            CREATE TRIGGER IF NOT EXISTS verses_fts_delete AFTER DELETE ON verses BEGIN
                DELETE FROM verses_fts WHERE rowid = old.id;
            END""",
            
            """
            CREATE TRIGGER IF NOT EXISTS verses_fts_update AFTER UPDATE ON verses BEGIN
                DELETE FROM verses_fts WHERE rowid = old.id;
                INSERT INTO verses_fts(
                    rowid, verse_id, kanda_name, shloka_iast, 
                    translation_en, translation_hi, sarga_titles_json
                ) VALUES (
                    new.id, new.verse_id, new.kanda_name, 
                    new.shloka_iast, new.translation_en, new.translation_hi,
                    new.sarga_titles_json
                );
            END"""
        ]
    return triggers


# Coordinate parsing utilities
def parse_verse_coordinates(coordinate_string: str) -> Optional[Dict[str, int]]:
    """Parse K-S-V coordinate string into components"""
    
    import re
    match = re.match(r'^(\d+)-(\d+)-(\d+)(?:-(\d+))?$', coordinate_string)
    
    if match:
        return {
            'kanda': int(match.group(1)),
            'sarga': int(match.group(2)),
            'verse': int(match.group(3)),
            'verse_end': int(match.group(4)) if match.group(4) else None
        }
    
    return None


def format_verse_id(kanda: int, sarga: int, verse: int, verse_end: Optional[int] = None) -> str:
    """Format verse coordinates into standard verse_id"""
    
    if verse_end and verse_end != verse:
        return f"{kanda}-{sarga}-{verse}-{verse_end}"
    else:
        return f"{kanda}-{sarga}-{verse}"
