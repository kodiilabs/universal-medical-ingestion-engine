# Universal Medical Ingestion Engine - Setup Guide

## Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Backend API and processing pipeline |
| Node.js | 18+ | React frontend |
| Ollama | latest | Local LLM inference (MedGemma, VLM) |
| Tesseract | 5+ | Fallback OCR (optional but recommended) |

## Quick Start

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd universal-medical-ingestion-engine

# 2. Set up Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Set up environment config
cp .env.example .env
# Edit .env if needed (defaults work for local development)

# 4. Install and start Ollama, then pull the models
ollama pull MedAIBase/MedGemma1.5:4b-it-q8_0
ollama pull minicpm-v

# 5. Set up reference databases (LOINC + RxNorm)
# See "Reference Data Setup" section below

# 6. Install frontend dependencies
cd frontend && npm install && cd ..

# 7. Run the application
./scripts/run_all.sh
```

The app will be available at:
- **React Frontend**: http://localhost:3000
- **FastAPI Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## Step-by-Step Setup

### 1. Python Backend

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
cp .env.example .env
```

The defaults in `.env.example` work for local development with Ollama. Key settings:

| Setting | Default | Description |
|---|---|---|
| `BACKEND` | `ollama` | LLM backend (`ollama` or `transformers`) |
| `OLLAMA_MODEL` | `MedAIBase/MedGemma1.5:4b-it-q8_0` | MedGemma model for extraction |
| `VLM_MODEL` | `minicpm-v` | Vision model for image extraction |
| `USE_CLOUD` | `false` | Set `true` to use Azure GPT-4o instead of local models |

### 3. Ollama Models

Install Ollama from https://ollama.ai, then pull the required models:

```bash
# Required: MedGemma for text extraction and classification
ollama pull MedAIBase/MedGemma1.5:4b-it-q8_0

# Required: Vision model for image/scanned document extraction
ollama pull minicpm-v

# Optional: Embedding model (only if using Ollama for embeddings)
# By default, the app uses sentence-transformers (no Ollama needed)
# ollama pull mxbai-embed-large
```

**Performance tip**: For faster processing, start Ollama with parallel request support:
```bash
OLLAMA_NUM_PARALLEL=2 ollama serve
```

### 4. Reference Data Setup

The app uses LOINC (lab tests) and RxNorm (medications) databases for enrichment. These are large reference datasets that need to be downloaded separately.

#### Option A: Automated Setup (Recommended)

```bash
# Download LOINC data and build lab tests database
python data/lab_tests/populate_labtests_db.py

# Download RxNorm data and build medications database
python data/medications/populate_medications_db.py
```

#### Option B: Manual Setup

**LOINC (Lab Tests)**:
1. Download LOINC from https://loinc.org/downloads/ (free account required)
2. Extract to `data/lab_tests/Loinc_2.81/`
3. Run: `python data/lab_tests/populate_labtests_db.py`

**RxNorm (Medications)**:
1. Download RxNorm Full Release from https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html
2. Extract to `data/medications/RxNorm_full_01052026/`
3. Run: `python data/medications/populate_medications_db.py`

The populate scripts build SQLite databases (`lab_tests.db`, `medications.db`) used at runtime.

### 5. Optional: Tesseract OCR

Tesseract provides fallback OCR. PaddleOCR is the primary OCR engine (installed via pip), but Tesseract helps with edge cases.

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### 6. Frontend

```bash
cd frontend
npm install
cd ..
```

---

## Running the Application

### All services at once

```bash
./scripts/run_all.sh
```

This starts the FastAPI backend (port 8000), React frontend (port 3000), and Streamlit UI (port 8501).

### Individual services

```bash
# Backend only
PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000

# Frontend only
cd frontend && npm start

# Streamlit UI only (legacy)
PYTHONPATH=src streamlit run ui/app.py --server.port 8501
```

---

## Docker Deployment

See [deployment/docker/README.md](deployment/docker/README.md) for Docker setup instructions.

```bash
cd deployment/docker
docker-compose up --build
```

---

## Cloud Extraction (Azure GPT-4o)

For higher accuracy without local GPU, you can use Azure OpenAI GPT-4o:

1. Create a GPT-4o deployment in Azure OpenAI Studio
2. Update `.env`:
   ```
   USE_CLOUD=true
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT=gpt-4o
   ```

---

## Project Structure

```
universal-medical-ingestion-engine/
├── api/                    # FastAPI backend
│   └── main.py             # API endpoints
├── frontend/               # React frontend
│   └── src/
├── src/medical_ingestion/  # Core processing pipeline
│   ├── classifiers/        # Document classification (MedGemma)
│   ├── core/               # Pipeline orchestration, config
│   ├── enrichers/          # LOINC/RxNorm enrichment
│   ├── extractors/         # OCR, VLM, LLM extraction
│   ├── fhir_utils/         # FHIR R4 bundle generation
│   ├── medgemma/           # Ollama/Transformers LLM clients
│   └── processors/         # Document-type-specific processing
├── data/
│   ├── samples/            # Test documents (labs, prescriptions, radiology)
│   ├── lab_tests/          # LOINC database (generated)
│   └── medications/        # RxNorm database (generated)
├── scripts/                # Setup and run scripts
├── models/                 # Model configuration files
│   ├── Modelfile.medgemma  # Ollama model definition
│   └── download_models.py  # Model download script
├── .env.example            # Environment configuration template
├── requirements.txt        # Python dependencies
└── SETUP.MD                # This file
```

---

## Troubleshooting

**Ollama connection refused**: Make sure Ollama is running (`ollama serve` or check the system tray).

**Model not found**: Run `ollama list` to see installed models. Pull missing ones with `ollama pull <model>`.

**PaddleOCR import errors**: PaddleOCR requires `paddlepaddle`. If pip install fails, try: `pip install paddlepaddle paddleocr`.

**Slow processing**: Each document makes 2-3 LLM calls. On CPU, expect 30-90 seconds per document. Use `OLLAMA_NUM_PARALLEL=2 ollama serve` for faster throughput.

**Out of memory**: MedGemma 4B needs ~3 GB RAM. If running both MedGemma and a VLM, ensure 8+ GB free RAM. Reduce to `VLM_MODEL=moondream` (1.8B) for lower memory usage.
