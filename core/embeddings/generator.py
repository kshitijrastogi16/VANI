"""
VANI V3 Embeddings Generator - FIXED BUGS + V1 Fallback Strategy
Enforcing: V2 E5-Large primary + V1 cultural fallback + CRITICAL BUG FIXES
"""

import os
import json
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingsGenerator:
    """Production-grade embeddings generation with cultural fallback"""
    
    def __init__(self, 
                 primary_model: str = "intfloat/multilingual-e5-large",
                 fallback_model: str = "l3cube-pune/indic-sentence-similarity-sbert",
                 batch_size: int = None):  # Changed to optional
        
        self.primary_model_id = primary_model
        self.fallback_model_id = fallback_model
        self.batch_size = batch_size or self._determine_optimal_batch_size()
        self.model: Optional[SentenceTransformer] = None
        self.current_model_id: Optional[str] = None
        
        self._load_optimal_model()
    
    def _load_optimal_model(self):
        """Load embedding model with V1 fallback strategy"""
        
        # Try primary model (V2's choice for performance)
        try:
            logger.info(f"🚀 Loading primary embedding model: {self.primary_model_id}")
            self.model = SentenceTransformer(self.primary_model_id)
            self.current_model_id = self.primary_model_id
            logger.info("✅ Primary model loaded successfully")
            return
        except Exception as e:
            logger.warning(f"⚠️ Primary model failed: {e}")
        
        # Fallback to cultural model (V1's cultural optimization)
        try:
            logger.info(f"🔄 Loading fallback model: {self.fallback_model_id}")
            self.model = SentenceTransformer(self.fallback_model_id)
            self.current_model_id = self.fallback_model_id
            logger.info("✅ Fallback model loaded successfully")
            return
        except Exception as e:
            logger.warning(f"⚠️ Fallback model failed: {e}")
        
        # Final fallback (V1 strategy)
        try:
            logger.info("🔄 Loading final fallback: all-MiniLM-L6-v2")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.current_model_id = "all-MiniLM-L6-v2"
            logger.info("✅ Final fallback model loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load any embedding model: {e}")
    
    def _determine_optimal_batch_size(self) -> int:
        """
        Determine optimal batch size based on available GPU memory
        Defaults to 32 if GPU not available or detection fails
        """
        try:
            import torch
            if torch.cuda.is_available():
                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                # Approximate memory per item (conservative estimate for E5-large)
                per_item = 1024 * 1024 * 50  # 50MB per item
                return min(256, max(8, int(free_mem / per_item)))
        except:
            pass
        return 32  # Default fallback
    
    def _get_tokenizer(self):
        """Return underlying HF tokenizer if accessible via sentence-transformers; else None."""
        try:
            first = getattr(self.model, "_first_module", None)
            if callable(first):
                first = first()
            if first is not None and hasattr(first, "tokenizer"):
                return first.tokenizer
        except Exception:
            pass
        try:
            # Some SentenceTransformer versions expose .tokenizer directly
            if hasattr(self.model, "tokenizer"):
                return self.model.tokenizer
        except Exception:
            pass
        return None

    def _get_max_seq_length(self) -> int:
        """Best-effort retrieval of model max seq length (defaults to 512)."""
        try:
            if hasattr(self.model, "get_max_seq_length"):
                return int(self.model.get_max_seq_length())
            if hasattr(self.model, "max_seq_length"):
                return int(self.model.max_seq_length)
        except Exception:
            pass
        return 512

    def _count_tokens(self, text: str) -> int:
        """Count tokens using HF tokenizer if available; fallback to whitespace token count."""
        tok = self._get_tokenizer()
        try:
            if tok is not None:
                return int(len(tok(text, truncation=False, add_special_tokens=True)["input_ids"]))
        except Exception:
            pass
        return int(len((text or "").split()))
    
    def build_corpus(self, db_path: str) -> Tuple[List[str], List[Dict]]:
        """Extract and prepare text corpus from database"""
        
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        query = """
        SELECT id, verse_id, kanda_num, sarga_num, verse_index, 
               translation_en, shloka_iast, translation_hi, sarga_titles_json
        FROM verses 
        ORDER BY kanda_num, sarga_num, verse_index
        """
        
        rows = conn.execute(query).fetchall()
        conn.close()
        
        texts, mappings = [], []
        
        logger.info(f"📚 Processing {len(rows)} verses for embedding...")
        
        for row in tqdm(rows, desc="Building corpus"):
            # Extract and JOIN ALL sarga titles for richer semantic context
            try:
                raw_titles = row["sarga_titles_json"]
                titles = json.loads(raw_titles) if raw_titles else []
                if not isinstance(titles, list):
                    titles = []
                joined_titles = " | ".join([t for t in titles if isinstance(t, str) and t.strip()])
            except (json.JSONDecodeError, TypeError, KeyError):
                joined_titles = ""

            # Structured, field-tagged passage (multi-line) for clarity (EN + IAST only)
            en = (row['translation_en'] or "").strip()
            iast = (row['shloka_iast'] or "").strip()
            body = f"EN: {en}\nIAST: {iast}\nTITLES: {joined_titles}".strip()

            # E5 asymmetric prefix handling
            if "e5" in self.current_model_id.lower():
                texts.append(f"passage: {body}")
            else:
                texts.append(body)
            
            mappings.append({
                "rowid": row["id"],
                "verse_id": row["verse_id"],
                "kanda": row["kanda_num"],
                "sarga": row["sarga_num"],
                "verse": row["verse_index"]
            })
        
        logger.info(f"✅ Corpus built: {len(texts)} passages")
        return texts, mappings
    
    def build_windowed_corpus_from_json(
        self,
        json_path: str,
        window_size_units: int = 5,
        stride_units: int = 4,
        min_window_size_units: int = 1,
    ) -> Tuple[List[str], List[Dict]]:
        """
        Build a windowed corpus directly from the raw Ramayana JSON, preserving multi-verse blocks
        as atomic units. Each window contains 3–4 consecutive units (configurable) within the same
        (kanda, sarga). Embeds only English + IAST in a structured, multi-line chunk.

        Returns:
            texts: List of passage strings (with optional E5 prefix) ready for embedding
            mappings: List of per-window metadata dicts for mappings.json
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts: List[str] = []
        mappings: List[Dict] = []

        def join_titles(titles: List[str]) -> str:
            try:
                return " | ".join([t.strip() for t in titles if isinstance(t, str) and t.strip()])
            except Exception:
                return ""

        def unit_display_label(kanda_num: int, sarga_num: int, verse_indices: List[int]) -> str:
            if not verse_indices:
                return f"{kanda_num}.{sarga_num}.?"
            start = verse_indices[0]
            end = verse_indices[-1]
            if start == end:
                return f"{kanda_num}.{sarga_num}.{start}"
            # Use an en dash for ranges
            return f"{kanda_num}.{sarga_num}.{start}–{end}"

        # Iterate kandas and sargas
        for kanda_obj in data:
            kanda_name = kanda_obj.get("kanda", "")
            sargas = kanda_obj.get("sargas", [])

            for sarga_obj in sargas:
                sarga_num = int(sarga_obj.get("sarga_index", 0))
                titles = sarga_obj.get("titles", [])
                joined_titles = join_titles(titles)
                verses = sarga_obj.get("verses", [])

                # Build atomic units (each item in verses is an atomic unit; may include multi-verse indices)
                units = []
                for v in verses:
                    verse_indices: List[int] = v.get("verse_indices", [])
                    verse_ids: List[str] = v.get("verse_ids", [])
                    en: str = (v.get("translation_en") or "").strip()
                    iast: str = (v.get("shloka_iast") or "").strip()

                    # Derive kanda_num from verse_ids when possible; fallback to parse from first id
                    if verse_ids:
                        try:
                            k_num = int(verse_ids[0].split("-")[0])
                        except Exception:
                            k_num = 0
                    else:
                        k_num = 0

                    units.append({
                        "kanda_num": k_num,
                        "sarga_num": sarga_num,
                        "kanda_name": kanda_name,
                        "verse_indices": verse_indices,
                        "verse_ids": verse_ids,
                        "en": en,
                        "iast": iast,
                        "titles_joined": joined_titles,
                    })

                if not units:
                    continue

                # Sliding window over units within this sarga
                W = max(1, int(window_size_units))
                S = max(1, int(stride_units))
                MINW = max(1, int(min_window_size_units))
                n = len(units)
                i = 0
                last_end = 0  # track last covered end (exclusive) in unit indices
                while i + W <= n:
                    window = units[i:i+W]

                    # Compute start/end verse indices for display
                    start_idx = window[0]["verse_indices"][0] if window[0]["verse_indices"] else None
                    end_idx = window[-1]["verse_indices"][-1] if window[-1]["verse_indices"] else None

                    # Build English section
                    english_lines = [f"English: Verse {start_idx} to Verse {end_idx} :"]
                    for u in window:
                        label = unit_display_label(u["kanda_num"], u["sarga_num"], u["verse_indices"]) \
                                 if u["verse_indices"] else f"{u['kanda_num']}.{u['sarga_num']}.?"
                        english_lines.append(f"{label} {u['en']}")

                    # Build IAST section
                    iast_lines = [f"IAST: Verse {start_idx} to Verse {end_idx} :"]
                    for u in window:
                        label = unit_display_label(u["kanda_num"], u["sarga_num"], u["verse_indices"]) \
                                 if u["verse_indices"] else f"{u['kanda_num']}.{u['sarga_num']}.?"
                        iast_lines.append(f"{label} {u['iast']}")

                    chunk = (
                        f"Kanda: {kanda_name}\n"
                        f"Sarga: {sarga_num}\n"
                        f"Titles: {joined_titles}\n\n"
                        + "\n".join(english_lines) + "\n\n"
                        + "\n".join(iast_lines)
                    ).strip()

                    # E5 asymmetric prefix when applicable
                    text_for_embed = f"passage: {chunk}" if "e5" in (self.current_model_id or "").lower() else chunk
                    texts.append(text_for_embed)

                    # Build mapping entry
                    flat_ids: List[str] = []
                    for u in window:
                        flat_ids.extend(list(u.get("verse_ids", [])))

                    # Choose a central anchor verse_id for better representativeness
                    try:
                        mid_unit = window[len(window)//2]
                        vid_list = list(mid_unit.get("verse_ids", []))
                        anchor_id = vid_list[len(vid_list)//2] if vid_list else window[0].get("verse_ids", [None])[0]
                    except Exception:
                        anchor_id = window[0].get("verse_ids", [None])[0]

                    # Monitoring: token counts and potential truncation
                    approx_tokens = self._count_tokens(text_for_embed)
                    max_len = self._get_max_seq_length()

                    mappings.append({
                        "kanda_num": window[0]["kanda_num"],
                        "sarga_num": window[0]["sarga_num"],
                        "start_verse_index": start_idx,
                        "end_verse_index": end_idx,
                        "verse_ids": flat_ids,
                        "anchor_verse_id": anchor_id,
                        "kanda_name": kanda_name,
                        "titles": titles,
                        "approx_token_count": approx_tokens,
                        "model_max_seq_length": max_len,
                        "maybe_truncated": bool(approx_tokens > max_len),
                    })

                    # Advance and update covered end
                    last_end = i + W
                    i += S

                # Tail window handling:
                # - If stride==1 and n>=W, last_end == n so no tail needed.
                # - If stride>1, ensure we end-align one final window to cover the end.
                # - If n < W, include a small tail window if it meets MINW.
                if last_end < n:
                    if n >= W:
                        # End-aligned tail of size W
                        tail_start = n - W
                        window = units[tail_start:n]
                        start_idx = window[0]["verse_indices"][0] if window[0]["verse_indices"] else None
                        end_idx = window[-1]["verse_indices"][-1] if window[-1]["verse_indices"] else None

                        english_lines = [f"English: Verse {start_idx} to Verse {end_idx} :"]
                        for u in window:
                            label = unit_display_label(u["kanda_num"], u["sarga_num"], u["verse_indices"]) \
                                     if u["verse_indices"] else f"{u['kanda_num']}.{u['sarga_num']}.?"
                            english_lines.append(f"{label} {u['en']}")

                        iast_lines = [f"IAST: Verse {start_idx} to Verse {end_idx} :"]
                        for u in window:
                            label = unit_display_label(u["kanda_num"], u["sarga_num"], u["verse_indices"]) \
                                     if u["verse_indices"] else f"{u['kanda_num']}.{u['sarga_num']}.?"
                            iast_lines.append(f"{label} {u['iast']}")

                        chunk = (
                            f"Kanda: {kanda_name}\n"
                            f"Sarga: {sarga_num}\n"
                            f"Titles: {joined_titles}\n\n"
                            + "\n".join(english_lines) + "\n\n"
                            + "\n".join(iast_lines)
                        ).strip()

                        text_for_embed = f"passage: {chunk}" if "e5" in (self.current_model_id or "").lower() else chunk
                        texts.append(text_for_embed)

                        flat_ids = []
                        for u in window:
                            flat_ids.extend(list(u.get("verse_ids", [])))

                        # Central anchor for tail as well
                        try:
                            mid_unit = window[len(window)//2]
                            vid_list = list(mid_unit.get("verse_ids", []))
                            anchor_id = vid_list[len(vid_list)//2] if vid_list else window[0].get("verse_ids", [None])[0]
                        except Exception:
                            anchor_id = window[0].get("verse_ids", [None])[0]

                        approx_tokens = self._count_tokens(text_for_embed)
                        max_len = self._get_max_seq_length()

                        mappings.append({
                            "kanda_num": window[0]["kanda_num"],
                            "sarga_num": window[0]["sarga_num"],
                            "start_verse_index": start_idx,
                            "end_verse_index": end_idx,
                            "verse_ids": flat_ids,
                            "anchor_verse_id": anchor_id,
                            "kanda_name": kanda_name,
                            "titles": titles,
                            "approx_token_count": approx_tokens,
                            "model_max_seq_length": max_len,
                            "maybe_truncated": bool(approx_tokens > max_len),
                        })
                    else:
                        # Very short sarga (n < W): include as a single small window if it meets MINW
                        if n >= MINW:
                            window = units[0:n]
                            start_idx = window[0]["verse_indices"][0] if window[0]["verse_indices"] else None
                            end_idx = window[-1]["verse_indices"][-1] if window[-1]["verse_indices"] else None

                            english_lines = [f"English: Verse {start_idx} to Verse {end_idx} :"]
                            for u in window:
                                label = unit_display_label(u["kanda_num"], u["sarga_num"], u["verse_indices"]) \
                                         if u["verse_indices"] else f"{u['kanda_num']}.{u['sarga_num']}.?"
                                english_lines.append(f"{label} {u['en']}")

                            iast_lines = [f"IAST: Verse {start_idx} to Verse {end_idx} :"]
                            for u in window:
                                label = unit_display_label(u["kanda_num"], u["sarga_num"], u["verse_indices"]) \
                                         if u["verse_indices"] else f"{u['kanda_num']}.{u['sarga_num']}.?"
                                iast_lines.append(f"{label} {u['iast']}")

                            chunk = (
                                f"Kanda: {kanda_name}\n"
                                f"Sarga: {sarga_num}\n"
                                f"Titles: {joined_titles}\n\n"
                                + "\n".join(english_lines) + "\n\n"
                                + "\n".join(iast_lines)
                            ).strip()

                            text_for_embed = f"passage: {chunk}" if "e5" in (self.current_model_id or "").lower() else chunk
                            texts.append(text_for_embed)

                            flat_ids = []
                            for u in window:
                                flat_ids.extend(list(u.get("verse_ids", [])))

                            try:
                                mid_unit = window[len(window)//2]
                                vid_list = list(mid_unit.get("verse_ids", []))
                                anchor_id = vid_list[len(vid_list)//2] if vid_list else window[0].get("verse_ids", [None])[0]
                            except Exception:
                                anchor_id = window[0].get("verse_ids", [None])[0]

                            approx_tokens = self._count_tokens(text_for_embed)
                            max_len = self._get_max_seq_length()

                            mappings.append({
                                "kanda_num": window[0]["kanda_num"],
                                "sarga_num": window[0]["sarga_num"],
                                "start_verse_index": start_idx,
                                "end_verse_index": end_idx,
                                "verse_ids": flat_ids,
                                "anchor_verse_id": anchor_id,
                                "kanda_name": kanda_name,
                                "titles": titles,
                                "approx_token_count": approx_tokens,
                                "model_max_seq_length": max_len,
                                "maybe_truncated": bool(approx_tokens > max_len),
                            })

        logger.info(f"✅ Windowed corpus built from JSON: {len(texts)} passages (W={window_size_units}, S={stride_units})")
        return texts, mappings

    def generate_embeddings(self, db_path: str, output_dir: str) -> Dict[str, any]:
        """Generate embeddings with FIXED V2 bugs"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Build corpus
        texts, mappings = self.build_corpus(db_path)
        
        # Generate embeddings in batches
        logger.info(f"🧠 Generating embeddings with {self.current_model_id}...")
        
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,  # Critical for cosine similarity
                show_progress_bar=False
            )
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings).astype(np.float32)
        
        # CRITICAL BUG FIX: Correct array indexing
        metadata = {
            "model_id": self.current_model_id,
            "model_version": self._get_model_version(),
            "model_card": f"https://huggingface.co/{self.current_model_id}",
            "dim": int(embeddings.shape[1]),      # FIX: was embeddings.shape[19]
            "count": int(embeddings.shape[0]),    # FIX: was embeddings.shape
            "normalization": "L2",
            "creation_time": np.datetime64('now').astype(str),
            "dependencies": {
                "sentence_transformers": self._get_package_version('sentence_transformers'),
                "numpy": self._get_package_version('numpy'),
                "faiss": self._get_package_version('faiss')
            }
        }
        
        # Save all files
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
        
        with open(os.path.join(output_dir, "mappings.json"), "w", encoding="utf-8") as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Embeddings saved to {output_dir}")
        logger.info(f"📊 Shape: {embeddings.shape}, Model: {self.current_model_id}")
        
        return {
            "embeddings_shape": embeddings.shape,
            "model_id": self.current_model_id,
            "output_dir": output_dir,
            "metadata": metadata
        }

    def generate_embeddings_from_json(
        self,
        json_path: str,
        output_dir: str,
        window_size_units: int = 5,
        stride_units: int = 4,
        min_window_size_units: int = 1,
    ) -> Dict[str, any]:
        """
        Generate embeddings using JSON-based sliding windows over atomic verse units.
        Embeds only English + IAST. Writes per-window mappings and enriched metadata.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Build windowed corpus from JSON
        texts, mappings = self.build_windowed_corpus_from_json(
            json_path=json_path,
            window_size_units=window_size_units,
            stride_units=stride_units,
            min_window_size_units=min_window_size_units,
        )

        logger.info(f"🧠 Generating windowed embeddings with {self.current_model_id}...")

        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.append(batch_embeddings)

        embeddings = np.vstack(all_embeddings).astype(np.float32)

        # Metadata: enriched for windowing
        metadata = {
            "model_id": self.current_model_id,
            "model_version": self._get_model_version(),
            "model_card": f"https://huggingface.co/{self.current_model_id}",
            "dim": int(embeddings.shape[1]),
            "count": int(embeddings.shape[0]),
            "normalization": "L2",
            "creation_time": np.datetime64('now').astype(str),
            "dependencies": {
                "sentence_transformers": self._get_package_version('sentence_transformers'),
                "numpy": self._get_package_version('numpy'),
                "faiss": self._get_package_version('faiss'),
            },
            "languages_included": ["en", "iast"],
            "windowing": {
                "window_size_units": int(window_size_units),
                "stride_units": int(stride_units),
                "min_window_size_units": int(min_window_size_units),
                "unit_type": "atomic_verse",
                "multi_verse_policy": "atomic",
            },
            "source": "json",
        }

        # Save files
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
        with open(os.path.join(output_dir, "mappings.json"), "w", encoding="utf-8") as f:
            json.dump(mappings, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Windowed embeddings saved to {output_dir}")
        logger.info(f"📊 Shape: {embeddings.shape}, Model: {self.current_model_id}")

        return {
            "embeddings_shape": embeddings.shape,
            "model_id": self.current_model_id,
            "output_dir": output_dir,
            "metadata": metadata,
        }
    
    def _get_package_version(self, package_name: str) -> str:
        """Helper to get package version"""
        try:
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except:
            return "unknown"
            
    def _get_model_version(self) -> str:
        """Get the model version from HuggingFace"""
        try:
            if hasattr(self.model, '__version__'):
                return self.model.__version__
            return "unknown"
        except:
            return "unknown"


def build_embeddings(db_path: str, output_dir: str, 
                    primary_model: str = "intfloat/multilingual-e5-large",
                    fallback_model: str = "l3cube-pune/indic-sentence-similarity-sbert") -> Dict:
    """Main function to build embeddings with V3 hybrid strategy"""
    
    generator = EmbeddingsGenerator(
        primary_model=primary_model,
        fallback_model=fallback_model
    )
    
    return generator.generate_embeddings(db_path, output_dir)


if __name__ == "__main__":
    # Test embeddings generation
    result = build_embeddings(
        db_path="data/processed/ramayana.db",
        output_dir="data/processed/embeddings"
    )
    print(f"✅ Embeddings generation complete: {result}")
