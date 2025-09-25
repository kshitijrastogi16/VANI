"""
VANI V3 Data Import Script - Production Data Pipeline
Enforcing: V1 advanced multi-verse handling + V2 robust validation
"""

import os
import json
import sqlite3
import logging
from typing import Dict, List, Any, Tuple
import argparse
from pathlib import Path

from core.database.schema import ensure_database_schema
from core.database.models import VerseModel, KandaModel, SargaModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VANIDataImporter:
    """
    Production-grade data importer with validation and error recovery
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.stats = {
            "kandas_imported": 0,
            "sargas_imported": 0,
            "verses_imported": 0,
            "multi_verses": 0,
            "errors": 0
        }
    
    def import_ramayana_data(self, json_file_path: str) -> bool:
        """
        Import complete Ramayana data from JSON file
        """
        
        logger.info(f"🚀 Starting Ramayana data import from {json_file_path}")
        
        # Ensure database schema exists
        if not ensure_database_schema(self.db_path):
            logger.error("Failed to setup database schema")
            return False
        
        # Load and validate JSON data
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                ramayana_data = json.load(f)
            
            logger.info(f"Loaded JSON data: {len(ramayana_data)} kandas")
            
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            return False
        
        # Import data
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            # Begin transaction for atomicity
            conn.execute("BEGIN TRANSACTION")
            
            for kanda_data in ramayana_data:
                self._import_kanda(conn, kanda_data)
            
            # Commit transaction
            conn.commit()
            
            # Rebuild FTS index for optimal search performance
            self._rebuild_fts_index(conn)
            
            conn.close()
            
            self._log_import_stats()
            logger.info("✅ Ramayana data import completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Import failed: {e}")
            if 'conn' in locals():
                conn.rollback()
                conn.close()
            return False
    
    def _import_kanda(self, conn: sqlite3.Connection, kanda_data: Dict[str, Any]):
        """
        Import single kanda with all sargas and verses
        """
        
        kanda_name = kanda_data.get('kanda', 'Unknown Kanda')
        kanda_num = self._extract_kanda_number(kanda_name)
        sarga_count = kanda_data.get('sarga_count', 0)
        verse_count = kanda_data.get('verse_count_in_kanda', 0)
        
        logger.info(f"📖 Importing {kanda_name} (Kanda {kanda_num})")
        
        # Import kanda metadata
        try:
            kanda_model = KandaModel(
                kanda_num=kanda_num,
                kanda_name=kanda_name,
                sarga_count=sarga_count,
                verse_count_in_kanda=verse_count
            )
            
            conn.execute("""
            INSERT OR REPLACE INTO kandas (kanda_num, kanda_name, sarga_count, verse_count_in_kanda)
            VALUES (?, ?, ?, ?)
            """, (kanda_model.kanda_num, kanda_model.kanda_name, 
                  kanda_model.sarga_count, kanda_model.verse_count_in_kanda))
            
            self.stats["kandas_imported"] += 1
            
        except Exception as e:
            logger.error(f"Failed to import kanda {kanda_name}: {e}")
            self.stats["errors"] += 1
            raise
        
        # Import sargas
        sargas = kanda_data.get('sargas', [])
        for sarga_data in sargas:
            self._import_sarga(conn, kanda_num, kanda_name, sarga_data)
    
    def _import_sarga(self, conn: sqlite3.Connection, kanda_num: int, kanda_name: str, sarga_data: Dict[str, Any]):
        """
        Import single sarga with all verses
        """
        
        sarga_index = sarga_data.get('sarga_index', 1)
        verse_count = sarga_data.get('verse_count_in_sarga', 0)
        titles = sarga_data.get('titles', [])
        
        # Import sarga metadata
        try:
            sarga_model = SargaModel(
                kanda_num=kanda_num,
                sarga_num=sarga_index,
                verse_count_in_sarga=verse_count,
                titles=titles
            )
            
            conn.execute("""
            INSERT OR REPLACE INTO sargas (kanda_num, sarga_num, verse_count_in_sarga, titles_json)
            VALUES (?, ?, ?, ?)
            """, (sarga_model.kanda_num, sarga_model.sarga_num,
                  sarga_model.verse_count_in_sarga, json.dumps(sarga_model.titles)))
            
            self.stats["sargas_imported"] += 1
            
        except Exception as e:
            logger.error(f"Failed to import sarga {kanda_num}-{sarga_index}: {e}")
            self.stats["errors"] += 1
            raise
        
        # Import verses
        verses = sarga_data.get('verses', [])
        for verse_data in verses:
            self._import_verse(conn, kanda_num, kanda_name, sarga_index, titles, verse_data)
    
    def _import_verse(self, conn: sqlite3.Connection, kanda_num: int, kanda_name: str, 
                     sarga_num: int, sarga_titles: List[str], verse_data: Dict[str, Any]):
        """
        Import single verse with V1 multi-verse handling
        """
        
        try:
            # Extract verse information
            verse_indices = verse_data.get('verse_indices', [])
            verse_ids = verse_data.get('verse_ids', [])
            
            if not verse_indices or not verse_ids:
                logger.warning(f"Skipping verse with missing indices/IDs in {kanda_num}-{sarga_num}")
                self.stats["errors"] += 1
                return
            
            # Handle multi-verse entries (V1 advanced feature)
            is_multi_verse = len(verse_indices) > 1
            primary_verse_index = verse_indices[0]
            
            # Create combined verse_id for multi-verse entries
            if is_multi_verse:
                verse_nums = [vid.split('-')[-1] for vid in verse_ids]
                primary_prefix = '-'.join(verse_ids[0].split('-')[:-1])
                combined_verse_id = f"{primary_prefix}-{verse_nums[0]}-{verse_nums[-1]}"
                self.stats["multi_verses"] += 1
            else:
                combined_verse_id = verse_ids[0]
            
            # Create verse model
            verse_model = VerseModel(
                verse_id=combined_verse_id,
                kanda_num=kanda_num,
                sarga_num=sarga_num,
                verse_index=primary_verse_index,
                kanda_name=kanda_name,
                shloka_sanskrit=verse_data.get('shloka_sanskrit', ''),
                shloka_iast=verse_data.get('shloka_iast', ''),
                translation_en=verse_data.get('translation_en', ''),
                translation_hi=verse_data.get('translation_hi', ''),
                sarga_titles_json=json.dumps(sarga_titles)
            )
            
            # Insert verse
            conn.execute("""
            INSERT OR REPLACE INTO verses 
            (verse_id, kanda_num, sarga_num, verse_index, kanda_name,
             shloka_sanskrit, shloka_iast, translation_en, translation_hi, sarga_titles_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                verse_model.verse_id, verse_model.kanda_num, verse_model.sarga_num,
                verse_model.verse_index, verse_model.kanda_name,
                verse_model.shloka_sanskrit, verse_model.shloka_iast,
                verse_model.translation_en, verse_model.translation_hi,
                verse_model.sarga_titles_json
            ))
            
            self.stats["verses_imported"] += 1
            
            # Log progress
            if self.stats["verses_imported"] % 1000 == 0:
                logger.info(f"   Imported {self.stats['verses_imported']} verses...")
                
        except Exception as e:
            logger.error(f"Failed to import verse {kanda_num}-{sarga_num}-{verse_indices}: {e}")
            self.stats["errors"] += 1
            raise
    
    def _extract_kanda_number(self, kanda_name: str) -> int:
        """
        Extract kanda number from kanda name
        """
        
        kanda_mapping = {
            'baala kanda': 1, 'bala kanda': 1,
            'ayodhya kanda': 2, 'ayodhaya kanda': 2,
            'aranya kanda': 3,
            'kishkindha kanda': 4, 'kiskindha kanda': 4,
            'sundara kanda': 5, 'sundar kanda': 5,
            'yuddha kanda': 6, 'yudha kanda': 6, 'lanka kanda': 6,
            'uttara kanda': 7, 'uttar kanda': 7
        }
        
        normalized_name = kanda_name.lower().strip()
        return kanda_mapping.get(normalized_name, 1)  # Default to 1 if not found
    
    def _rebuild_fts_index(self, conn: sqlite3.Connection):
        """
        Rebuild FTS5 index after data import
        """
        
        logger.info("🔄 Rebuilding FTS5 search index...")
        
        try:
            # Verify FTS5 availability
            try:
                cursor = conn.execute("PRAGMA compile_options")
                compile_options = [row[0] for row in cursor.fetchall()]
                if 'ENABLE_FTS5' not in compile_options:
                    logger.info("FTS5 is not enabled; skipping FTS index rebuild.")
                    return
            except Exception:
                logger.info("Unable to determine FTS5 support; skipping FTS index rebuild.")
                return

            # Delegate to robust rebuild function
            from core.database.schema import rebuild_fts_index
            # Ensure previous work committed
            try:
                conn.commit()
            except Exception:
                pass
            if not rebuild_fts_index(self.db_path):
                logger.warning("FTS index rebuild reported failure; continuing without FTS.")
            else:
                logger.info("✅ FTS index rebuilt successfully")
        
        except Exception as e:
            logger.error(f"FTS rebuild failed: {e}")
            # Do not raise; continue gracefully without FTS
    
    def _log_import_stats(self):
        """
        Log comprehensive import statistics
        """
        
        logger.info("📊 Import Statistics:")
        logger.info(f"   Kandas imported: {self.stats['kandas_imported']}")
        logger.info(f"   Sargas imported: {self.stats['sargas_imported']}")
        logger.info(f"   Verses imported: {self.stats['verses_imported']}")
        logger.info(f"   Multi-verse entries: {self.stats['multi_verses']}")
        logger.info(f"   Errors encountered: {self.stats['errors']}")


def main():
    """
    Command-line interface for data import
    """
    
    parser = argparse.ArgumentParser(description="Import Ramayana data into VANI database")
    parser.add_argument("json_file", help="Path to Ramayana JSON data file")
    parser.add_argument("--db-path", default="data/processed/ramayana.db", 
                       help="Path to SQLite database file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.json_file):
        logger.error(f"JSON file not found: {args.json_file}")
        return 1
    
    # Create output directory
    os.makedirs(os.path.dirname(args.db_path), exist_ok=True)
    
    # Import data
    importer = VANIDataImporter(args.db_path)
    success = importer.import_ramayana_data(args.json_file)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
