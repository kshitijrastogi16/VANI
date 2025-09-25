# VANI - Voice of Ancient Narrative Intelligence

**Production-Ready RAG System for Ramayana Question Answering**

VANI is a sophisticated Retrieval-Augmented Generation (RAG) system that combines advanced entity resolution, hybrid search, and multi-model LLM integration to provide accurate, culturally-sensitive answers about the Ramayana. The retrieval and synthesis stack is focused on IAST transliteration and English. Sanskrit (Devanagari) and Hindi are preserved in the data model but are not used for ranking or LLM context.

## 🎯 What VANI Can Do

### Query Types Supported
- **Count Queries**: "How many kandas are in the Ramayana?" (direct database, no LLM needed)
- **Citation Lookups**: "1.1.1" → Full verse with IAST and English (Sanskrit/Hindi stored in DB but not used in LLM context)
- **Entity-Enhanced Search**: "Hanuman's devotion" → Resolves entities and expands with aliases
- **Semantic Search**: Natural language understanding with cultural context
- **General Questions**: "What is dharma according to the Ramayana?"

### Example Queries
```
Count Queries (Direct Database):
- "How many kandas are in the Ramayana?"
- "Total verses in Sundara Kanda?"

Citation Lookups:
- "1.1.1" → Shows verse 1 of sarga 1 of kanda 1
- "Show me verse 5.2.10"

Entity-Enhanced Queries:
- "Hanuman's devotion to Rama"
- "Sita's exile"
- "Battle between Rama and Ravana"

General Questions:
- "What is dharma according to the Ramayana?"
- "Why did Rama go to the forest?"
```

## ✨ Key Features

### 🧠 Advanced Intelligence Layer
- **V1 Entity Resolution**: Bidirectional relationship intelligence with culturally-aware normalization
- **Sophisticated Hybrid Search**: BM25 + Vector similarity with weighted fusion and entity enhancement (EN + IAST focused)
- **Multi-Model LLM Fallback**: 3-tier Groq model cascade with intelligent query routing

### 🔍 Search Capabilities
- **Cultural Intelligence**: IAST diacritic handling (retrieval uses EN + IAST)
- **Relationship Recognition**: Cultural context preservation and pattern recognition
- **Performance Optimized**: FAISS IndexFlatIP for fast similarity search

### 📚 Citations
- **Verse Details**: IAST transliteration and English translation (Sanskrit/Hindi are present in raw data but not used in ranking/LLM context)
- **Context Information**: Kanda name, Sarga number, chapter titles
- **Citation Format**: Kanda.Sarga.Verse (e.g., 2.72.50); for ranges within a sarga use K.S.V1-V2 (e.g., 6.129.5-9)
- **Source Validation**: Only verified citations from database

### 🗼 Production Architecture
- **Clean V2 Architecture**: Modular layers with proper separation of concerns
- **Production Validation**: Comprehensive startup checks and health monitoring
- **Error Resilience**: Graceful fallbacks and component isolation
- **Performance Optimization**: WAL mode SQLite, FTS5 search, connection pooling

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- SQLite with FTS5 support
- Groq API key (for LLM functionality)

### 1. Installation
```bash
# Clone repository
git clone <repository-url>
cd VANI

# Create virtual environment
python -m venv vani_env
source vani_env/bin/activate  # On Windows: vani_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
# REQUIRED: Add your Groq API key
GROQ_API_KEY=your_actual_groq_api_key_here
```

#### Config: Citation Context Window
You can control how many neighboring verses (context) are included around a cited verse.

- Add this to your `.env` (already present in `.env.example`):

```bash
# Number of neighboring verses to include around a cited verse (±N)
# Set to 0 to disable neighbor context globally
CITATION_CONTEXT_WINDOW=2
```

- This affects both endpoints when context is enabled:
  - `/query` (when `include_context=true`): the first source is the cited verse, followed by ±N neighbors from the same kanda/sarga.
  - `/search/verse/{verse_id}` (when `include_context=true`): returns a `sources` array with the cited verse and its neighbors.

- Notes:
  - Setting `CITATION_CONTEXT_WINDOW=0` disables neighbor context globally.
  - For `/query`, if you set `include_context=false`, only the main verse will be returned for citation queries regardless of the global window.
  - Neighbor selection is scoped by `kanda_num` and `sarga_num` using `verse_index` as the anchor.

### 3. Data Setup
```bash
# Create directories
mkdir -p data/raw data/processed/embeddings logs

# Place your data files:
# - data/raw/ramayana.json (main text data)
# - data/raw/characters.json (character entities)
# - data/raw/locations.json (location entities)

# Import data to database
python -m data.scripts.import_data data/raw/ramayana.json --db-path data/processed/ramayana.db

# Generate embeddings
python -m data.scripts.build_embeddings
```

### 4. Run VANI
```bash
# Start the server (recommended)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
> Tip: To enable API docs in development, set `DEBUG=true` in your `.env`.

### 5. Access VANI
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (requires `DEBUG=true`)
- **Health Check**: http://localhost:8000/health

## 📊 Project Structure

```
VANI/
├── app/                    # FastAPI application layer
│   ├── main.py            # Application entry point
│   ├── settings.py        # Centralized configuration
│   └── dependencies.py    # Dependency injection
├── core/                  # Business logic layer
│   ├── database/          # Database management
│   ├── embeddings/        # Vector embeddings (FIXED V2 bugs)
│   ├── search/            # Hybrid search engine
│   └── entities/          # Entity resolution (V1 RESTORED)
├── rag/                   # RAG pipeline components
│   ├── retriever.py       # Retrieval orchestration
│   ├── synthesizer.py     # Response synthesis
│   ├── citations.py       # Citation management
│   └── query_router.py    # Query classification
├── llm/                   # LLM integration
│   ├── client.py          # Multi-model Groq client (V1 RESTORED)
│   └── prompts.py         # Structured prompt templates
├── api/                   # API routes
│   └── routes/
│       ├── health.py      # Health monitoring
│       └── search.py      # Search endpoints
├── data/                  # Data management
│   ├── raw/               # Original data files
│   ├── processed/         # Database and embeddings
│   └── scripts/           # Data processing scripts
├── web/                   # Web interface
│   ├── static/css/        # Styles
│   └── templates/         # HTML templates
├── utils/                 # Utilities
│   └── logging.py         # Logging configuration
├── requirements.txt       # Dependencies
├── .env.example          # Configuration template
└── README.md             # Documentation
```

## 🔧 API Usage

### Query Endpoint
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Who is Hanuman?",
    "include_context": true,
    "max_results": 10
  }'
```

### Response Format (example)
```json
{
  "answer": "Hanuman is the devoted follower of Rama...",
  "sources": [
    {
      "verse_id": "5-1-15",
      "kanda_name": "Sundara Kanda",
      "sarga_num": 1,
      "translation_en": "...",
      "shloka_iast": "..."
    }
  ],
  "citations": ["5.1.15", "4.3.22"],
  "confidence": 0.92,
  "response_time": 1.45,
  "query_type": "general"
}
```
Note: Depending on your data import, additional fields such as `translation_hi` or `shloka_sanskrit` may be present in raw sources. The retrieval and LLM context use EN + IAST.

## 🔥 Advanced Features

### Multi-Model LLM Fallback
VANI uses intelligent model selection and automatic fallback:

1. **llama-3.3-70b-versatile** (Production)
2. **deepseek-r1-distill-llama-70b** (Complex)
3. **llama-3.1-8b-instant** (Speed)

### Hybrid Embedding Strategy
- **Primary**: `BAAI/bge-m3` (Production)
- **Fallback**: `intfloat/multilingual-e5-large` (E5 asymmetric support)
- **Tertiary**: `l3cube-pune/indic-sentence-similarity-sbert`

### Cultural Intelligence
- IAST diacritic handling (EN + IAST focus)
- Cultural context preservation
- Relationship pattern recognition

## 📈 Performance Tuning

### Database Optimization
- WAL mode enabled for concurrency
- Optimized FTS5 configuration
- Proper indexing on verse_id, kanda, sarga
- Connection pooling and reuse

### Search Performance
- Hybrid search with 40% BM25 + 60% vector weighting
- Entity resolution caching
- Batch embedding generation
- FAISS IndexFlatIP for fast similarity

### LLM Optimization
- Intelligent model routing by query type
- Automatic fallback cascade
- Response caching (configurable)
- Streaming support for real-time responses

## 🛠️ Development

### Running Tests
Note: This repository does not currently include a `tests/` directory by default. The commands below are provided for when you add tests.

```bash
pytest tests/ -v
```

### Code Formatting
```bash
black app/ core/ rag/ llm/
flake8 app/ core/ rag/ llm/
```

### Optional Modules (not used in default runtime)
The following modules are included for future expansion or reference and are not wired into the default runtime flow:

- `core/entities/relationship.py` (relationship graph utilities)
- `core/search/vector_search.py` (alternative vector search helper)
- `llm/prompts.py` (prompt templates; synthesizer currently inlines its prompts)
- `rag/citations.py` (citation utilities; synthesizer handles basic citation logic)
- `tree.py` (project tree printing utility)

You can remove or integrate these depending on your needs.

### Database Management
```bash
# Rebuild FTS index
python -c "from core.database.schema import rebuild_fts_index; rebuild_fts_index('data/processed/ramayana.db')"

# Check database integrity
python -c "from core.database.schema import check_database_integrity; print(check_database_integrity('data/processed/ramayana.db'))"
```

## 🔒 Security & Privacy

- **No PII Collection**: Only processes Ramayana content
- **API Key Security**: Environment variable management
- **Input Validation**: Comprehensive Pydantic validation
- **Rate Limiting**: Configurable request limits
- **CORS Protection**: Configurable origins

## 📝 Citation & Academic Use

VANI provides academically rigorous citations in `Kanda.Sarga.Verse` format (e.g., `2.72.50`). Multi-verse ranges within a sarga are shown as `Kanda.Sarga.VerseStart-VerseEnd` (e.g., `6.129.5-9`). All citations are verified against the source database and include:

- **Sanskrit Original**: Devanagari script
- **IAST Transliteration**: International scholarly standard
- **English Translation**: Scholarly translation
- **Hindi Translation**: Cultural context
- **Source Context**: Kanda name, Sarga title, narrative context

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the established patterns
4. Add tests for new functionality  
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. A copy is included below for convenience. For distribution compliance, it is recommended to also include a dedicated `LICENSE` file at the repository root.

MIT License

Copyright (c) 2025 VANI Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 🙏 Acknowledgments
- Sanskrit texts and translations from traditional sources
- Groq for LLM API services
- Sentence Transformers community
- FastAPI and Pydantic frameworks

---
**VANI** - Bridging ancient wisdom with modern AI technology.