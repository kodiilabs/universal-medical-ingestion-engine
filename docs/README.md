# Universal Medical Ingestion Engine - Architecture

## Overview

The Universal Medical Ingestion Engine processes **any medical document** using AI-powered extraction with an extraction-first architecture. It automatically extracts all medical content, classifies documents, enriches with type-specific metadata, and outputs structured data including **FHIR R4 bundles**. The system learns from past extractions using a vector store and supports an interactive **chat-with-document** feature.

**Key principle:** Extract first, classify optionally. No document is rejected - unknown formats are processed with best-effort extraction and flagged for review.

---

## Architecture: Extraction-First Pipeline (V2)

The engine uses the **Extraction-First** architecture approach. This is the only supported processing pipeline.

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     EXTRACTION-FIRST PIPELINE (V2)                          │
└─────────────────────────────────────────────────────────────────────────────┘

    DOCUMENT UPLOAD
          │
          ├── HEIC/HEIF → JPEG conversion (iPhone photos)
          ├── EXIF orientation correction (rotated camera images)
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   STAGE 1: UNIVERSAL TEXT EXTRACTION                                        │
│   (universal_text_extractor.py)                                             │
│                                                                             │
│   Any format (PDF, Image) → Clean, structured text + layout info            │
│   • Auto-detects: digital PDF vs scanned PDF vs image                       │
│   • Standard: PaddleOCR (primary), pdfplumber (digital), VLM fallback      │
│   • VLM Unified: VLM replaces PaddleOCR, pdfplumber unchanged             │
│   • Preserves: word bboxes, tables, sections, regions                       │
│   • Per-page progress callback → Frontend shows "Page X of Y"             │
│   • EXIF transpose + HEIC conversion via load_image_for_ocr()              │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   STAGE 1.5: RAW FIELD EXTRACTION (Before Classification!)                  │
│   (_extract_raw_fields in extraction_first_pipeline.py)                     │
│                                                                             │
│   Extract ALL key-value pairs from raw text FIRST                           │
│   • Pattern matching: "Label: Value" pairs                                  │
│   • Invoice/reference numbers, amounts, dates                               │
│   • Medical/insurance-specific fields                                       │
│   • Ensures NO data is lost regardless of classification result            │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   STAGE 2: SIMILAR DOCUMENT LOOKUP (Vector Store)                           │
│   (adaptive_retrieval.py + vector_store.py)                                │
│                                                                             │
│   • Find similar documents using embeddings                                │
│   • Get extraction hints from successful past extractions                   │
│   • Supports strategies: simple, router, fusion, chunked                   │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   STAGE 3: EXTRACTION + CLASSIFICATION                                      │
│                                                                             │
│   ┌───────────────────────────────┐    ┌───────────────────────────────┐   │
│   │ CONTENT-AGNOSTIC EXTRACTOR    │    │ DOCUMENT CLASSIFIER            │   │
│   │ (content_agnostic_extractor)  │    │ (document_classifier.py)       │   │
│   │                               │    │                                │   │
│   │ SINGLE comprehensive LLM call │    │ • MedGemma + Fingerprint       │   │
│   │ (replaces 5 parallel calls):  │    │ • Returns: type, confidence    │   │
│   │ • Patient demographics        │    │ • Uses raw_fields for hints    │   │
│   │ • Test results/values         │    │                                │   │
│   │ • Medications + validation    │    │ (Parallel with extraction      │   │
│   │ • Clinical findings           │    │  when using cloud backends)    │   │
│   │ • Dates/providers             │    │                                │   │
│   │                               │    │                                │   │
│   │ + OCR correction pipeline:    │    │                                │   │
│   │   • correct_lab_value_ocr()   │    │                                │   │
│   │   • fix_spurious_spaces()     │    │                                │   │
│   │   • DNR value filtering       │    │                                │   │
│   └───────────────────────────────┘    └───────────────────────────────┘   │
│              │                                       │                      │
│              └──────────────┬────────────────────────┘                      │
│                             ▼                                               │
│              MERGE: Classification enriches extraction                      │
│              (raw_fields merged into GenericMedicalExtraction)             │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   STAGE 4: TYPE-SPECIFIC ENRICHMENT (if classification confident >= 0.7)   │
│   (enrichers/)                                                              │
│                                                                             │
│   Lab → LabEnricher: LOINC codes, reference ranges, abnormal flags,        │
│                      critical value detection, validation_status            │
│   Prescription → PrescriptionEnricher: RxNorm codes, drug interactions,    │
│                      drug class info, validation_status                     │
│   Radiology → RadiologyEnricher: Critical finding detection                 │
│   Pathology → PathologyEnricher: TNM staging, margin status                 │
│   Insurance → (raw_fields passed through for formatting)                    │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   STAGE 5: FHIR R4 BUNDLE GENERATION                                       │
│   (fhir/fhir_builder.py)                                                   │
│                                                                             │
│   Converts extraction results into FHIR R4-compliant resources:            │
│   • Patient, Observation, MedicationRequest, DiagnosticReport              │
│   • LOINC/RxNorm codes mapped to FHIR coding systems                      │
│   • Bundle type: "collection"                                              │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   STAGE 6: STORE SUCCESSFUL EXTRACTION (Learning)                           │
│                                                                             │
│   If confidence >= 0.7: Store to vector store for future hints             │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   RESULT TRANSFORMATION + OUTPUT                                            │
│   (_transform_v2_result_for_frontend in api/main.py)                       │
│                                                                             │
│   Transforms ExtractionFirstResult into frontend-compatible format:        │
│   • extracted_values: test results + medications as array                  │
│   • sections: findings, impressions, patient_info, provider_info           │
│   • raw_fields: all key-value pairs for tabular display                    │
│   • fhir_bundle: FHIR R4 resources                                        │
│   • classification: type, confidence                                        │
│   • display_name: human-readable document title                            │
│   • universal_extraction: complete extraction data for JSON view           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Processing Modes

The frontend provides three processing modes that control the extraction strategy:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│   PROCESSING MODE SELECTOR (Upload page)                                    │
│                                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐                  │
│   │    Auto      │  │    Fast     │  │  Most Accurate    │                  │
│   │  (default)   │  │             │  │                   │                  │
│   └─────────────┘  └─────────────┘  └──────────────────┘                  │
│                                                                             │
│   Auto:     App decides — OCR first, VLM fallback for low-quality scans    │
│   Fast:     OCR only — fastest, best for clear digital PDFs                │
│   Accurate: VLM for every page — slowest, best for handwritten/complex     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Mode Configuration Mapping

| Mode | `use_vlm` | `force_vlm_all_pages` | `use_consensus_extraction` | Behavior |
|------|-----------|----------------------|---------------------------|----------|
| **Auto** | `true` | `false` | unchanged | PaddleOCR primary, VLM fallback if OCR < threshold |
| **Fast** | `false` | `false` | `false` | PaddleOCR only, no VLM at all |
| **Accurate** | `true` | `true` | unchanged | VLM processes every page (overrides OCR results) |

### Flow: Frontend → API → Config → Extractor

```
Upload.js (processingMode state)
    │
    ▼
api.js: startProcessing(fileId, strategy, processingMode)
    │
    ▼
api/main.py: POST /api/v2/process?processing_mode=auto|fast|accurate
    │
    ▼
Config overrides applied to extraction pipeline config dict
    │
    ▼
UniversalTextExtractor reads: use_vlm, force_vlm_all_pages
```

---

## Performance Optimizations

### Single Comprehensive LLM Call

The content-agnostic extractor uses a **single comprehensive LLM call** instead of the previous 5 parallel calls (one each for patient info, dates, medications/tests, provider info, and raw fields). This reduces Ollama round-trips and improves consistency.

### LLM Response Caching

The Ollama client integrates a `PromptCache` (LRU + TTL + disk persistence) to avoid re-running identical prompts:

```
┌──────────────────────────────────────────────────┐
│  LLM REQUEST                                      │
│       │                                           │
│       ▼                                           │
│  PromptCache.get_response(prompt, params)         │
│       │                                           │
│       ├── HIT → Return cached response (0ms)      │
│       │                                           │
│       └── MISS → Call Ollama → Store in cache      │
│                                                    │
│  Cache Config:                                     │
│  • Max size: 500 entries (LRU eviction)           │
│  • TTL: 3600s (1 hour)                            │
│  • Auto-persist to disk every 5 operations        │
│  • Cache dir: ./ollama_cache                       │
└──────────────────────────────────────────────────┘
```

### Configurable Parallel Chunk Extraction

For long documents split into chunks, the extractor supports semaphore-controlled parallelism:

```python
# Sequential (default for Ollama — safe, avoids request queuing)
MAX_CONCURRENT_CHUNKS=1

# Parallel (opt-in for multi-GPU or cloud backends)
MAX_CONCURRENT_CHUNKS=4  # Uses asyncio.Semaphore
```

**Important:** Ollama serializes requests by default (`OLLAMA_NUM_PARALLEL=1`). Enabling parallel extraction without configuring Ollama causes request queuing and slower performance. Only enable when:
- Using cloud backends (GPT-4o, etc.)
- Running Ollama with `OLLAMA_NUM_PARALLEL` > 1

---

## Real-Time Progress Updates

The pipeline provides **real-time progress updates** via a callback mechanism:

```python
# In ExtractionFirstPipeline.process():
async def process(
    self,
    document_path: Path,
    progress_callback: Optional[callable] = None
) -> ExtractionFirstResult:

    def notify_progress(stage: PipelineStage):
        if progress_callback:
            progress_callback(stage, all_stages)

    stage = PipelineStage(name="text_extraction", status="running")
    notify_progress(stage)
    # ... do work ...
    stage.status = "completed"
    notify_progress(stage)
```

### Progress Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PIPELINE                           API                       FRONTEND   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Stage starts          ──→  progress_callback()                          │
│                              └──→ job["workflow_steps"] updated           │
│                                                                          │
│  Per-page progress     ──→  stage.details updated with                   │
│  (text_extraction)           { current_page: 3, total_pages: 7 }        │
│                              └──→ job["workflow_steps"] updated           │
│                                                                          │
│  (processing...)            GET /api/v2/jobs/{id}  ←─── Poll every 2s   │
│                              └──→ Returns current steps                   │
│                                                                          │
│                              ProcessFlow component shows:                │
│                              ✓ text_extraction (completed)              │
│                              ● raw_field_extraction (running)           │
│                                  "Extracting page 3 of 7..."           │
│                              ○ parallel_extraction (pending)            │
│                                                                          │
│  Stage completes       ──→  progress_callback()                          │
│                              └──→ job["workflow_steps"] updated           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Per-Page Progress

During Stage 1 (text extraction), a `page_progress` callback updates `stage.details` with the current page number. This is called from within the OCR/VLM per-page loop:

```python
def page_progress(current_page, total_pages):
    stage.details = { "current_page": current_page, "total_pages": total_pages }
    notify_progress(stage)
```

The frontend `ProcessFlow.js` displays this as "Extracting page X of Y..." during the running state. The callback runs from `asyncio.to_thread()` background threads; CPython's GIL makes the dict mutations thread-safe.

### Pipeline Stages

| Stage | Name | Description |
|-------|------|-------------|
| 1 | `text_extraction` | Universal text extraction from any format |
| 1.5 | `raw_field_extraction` | Extract ALL key-value pairs before classification |
| 2 | `similar_doc_lookup` | Find similar documents in vector store (optional) |
| 3 | `parallel_extraction` | Content-agnostic extraction + classification |
| 4 | `{type}_enrichment` | Type-specific enrichment (e.g., `lab_enrichment`) |
| 5 | `fhir_generation` | FHIR R4 bundle creation |
| 6 | `vector_store` | Store successful extraction for future learning |

---

## Core Components

### Pipeline Components

| Component | File | Purpose |
|-----------|------|---------|
| **ExtractionFirstPipeline** | `core/extraction_first_pipeline.py` | Main orchestrator with progress callbacks |
| **Orchestrator** | `core/orchestrator.py` | V1 pipeline entry point, display_name generation, FHIR output |
| **UniversalTextExtractor** | `extractors/universal_text_extractor.py` | X2Text pattern - any format → structured text |
| **ContentAgnosticExtractor** | `extractors/content_agnostic_extractor.py` | Single-call generic medical extraction |
| **DocumentClassifier** | `classifiers/document_classifier.py` | MedGemma + fingerprint-based classification |
| **AdaptiveRetrieval** | `core/adaptive_retrieval.py` | Multiple retrieval strategies + vector store |
| **PromptManager** | `core/prompt_manager.py` | Configurable prompts (Prompt Studio pattern) |
| **VectorStore** | `core/vector_store.py` | Document similarity + extraction hints |

### Extractors

| Extractor | File | Purpose |
|-----------|------|---------|
| **UniversalTextExtractor** | `extractors/universal_text_extractor.py` | Routes to PaddleOCR/VLM/pdfplumber; supports VLM Unified mode |
| **VLMClient** | `extractors/vlm_client.py` | VLM via Ollama — fallback (standard) or primary OCR (unified) |
| **OCRRouter** | `extractors/ocr_router.py` | Routes regions to appropriate OCR engine |
| **RegionDetector** | `extractors/region_detector.py` | Detects text, tables, stamps, signatures |
| **ContentAgnosticExtractor** | `extractors/content_agnostic_extractor.py` | LLM-based generic extraction with OCR correction |

### Enrichers

| Enricher | Document Type | Adds |
|----------|---------------|------|
| **LabEnricher** | Lab reports | LOINC codes, reference ranges, abnormal flags, critical values, validation_status |
| **PrescriptionEnricher** | Prescriptions | RxNorm codes, drug interactions, drug class, validation_status |
| **RadiologyEnricher** | Radiology | Critical findings, anatomical locations |
| **PathologyEnricher** | Pathology | TNM staging, tumor grade, margins |

### Databases

| Database | File | Purpose |
|----------|------|---------|
| **LabTestDB** | `constants/lab_test_db.py` | 96K+ LOINC codes, OCR-tolerant fuzzy matching, region-aware |
| **MedicationDB** | `constants/medication_db.py` | RxNorm lookup, OCR-tolerant fuzzy matching, drug class patterns |

---

## Validation Status Tracking

Both medications and test results carry a `validation_status` field for audit trail compliance:

### Medication Validation (`MedicationInfo.validation_status`)

| Status | Meaning |
|--------|---------|
| `verified` | Found exact match in RxNorm database |
| `ocr_corrected` | Found via edit-distance OCR correction |
| `medgemma_verified` | Confirmed by MedGemma LLM |
| `strength_mismatch` | Drug found but strength doesn't match known strengths |
| `unverified` | Not found in any database (needs human review) |

### Test Result Validation (`TestResult` enrichment)

| Status | Meaning |
|--------|---------|
| `verified` | Found in LOINC (exact/fuzzy/alias/prefix match) |
| `ocr_corrected` | Found via edit-distance correction |
| `unverified` | Not found in LOINC (needs human review) |

---

## Medical Database Architecture

### LOINC Database (`lab_test_db.py`)

```
┌──────────────────────────────────────────────────────────────┐
│  LOINC LOOKUP PIPELINE                                        │
│                                                                │
│  Input: "Hemoglobin" or "HGB" or "Hgb" or "hemoglbin"        │
│       │                                                        │
│       ▼                                                        │
│  1. CLINICAL_NAME_OVERRIDES (hardcoded known mismatches)      │
│     "hemoglobin" → 718-7 (not MCHC 786-4)                    │
│     "MPV" → 32623-1 (not MVV)                                │
│       │                                                        │
│       ▼ (if no override)                                       │
│  2. Exact match on component/shortname/consumer_name           │
│       │                                                        │
│       ▼ (if no exact match)                                    │
│  3. Alias match (semicolon-separated relatednames)            │
│     Whole-word matching to avoid substring false positives     │
│       │                                                        │
│       ▼ (if no alias match)                                    │
│  4. Prefix match + common_test_rank scoring                   │
│     rank=0 means UNRANKED (use 999999), not highest priority  │
│       │                                                        │
│       ▼ (if no prefix match)                                   │
│  5. OCR-tolerant fuzzy match (Levenshtein distance)           │
│                                                                │
│  Region-aware: USA (default), Canada, India                   │
│  Singleton pattern: one instance per process                   │
└──────────────────────────────────────────────────────────────┘
```

### RxNorm Database (`medication_db.py`)

```
┌──────────────────────────────────────────────────────────────┐
│  RxNorm LOOKUP PIPELINE                                       │
│                                                                │
│  Input: "AMOXICILIN 500MG" (with OCR typo)                   │
│       │                                                        │
│       ▼                                                        │
│  1. Drug name cleaning                                         │
│     Strip prefixes: TAB., CAP., INJ., SYR., SUSP., OINT.    │
│     Strip schedules: Morning, Evening, After Food, SOS        │
│     Strip trailing numbers: "ABCIXIMAB 1" → "ABCIXIMAB"      │
│       │                                                        │
│       ▼                                                        │
│  2. Exact match on drug name                                   │
│       │                                                        │
│       ▼ (if no exact match)                                    │
│  3. OCR variant generation                                     │
│     Character confusions: t↔l, n↔u, o↔a, rn↔m, etc.         │
│     Generates candidate spellings and checks each              │
│       │                                                        │
│       ▼ (if no OCR match)                                      │
│  4. Fuzzy match: Levenshtein distance + Soundex               │
│     Context scoring uses other medications for disambiguation  │
│       │                                                        │
│       ▼                                                        │
│  5. Drug class pattern matching (fallback)                     │
│     Beta blockers, ACE inhibitors, statins, SSRIs, etc.       │
│                                                                │
│  Region-aware: USA (default), Canada, India                   │
│  Singleton pattern: one instance per process                   │
└──────────────────────────────────────────────────────────────┘
```

---

## OCR Correction Pipeline

Medical OCR produces frequent errors that are corrected at multiple stages:

### Lab Value OCR Correction (`correct_lab_value_ocr()`)

Fixes common numeric OCR errors in lab values:
- `E` → `.` (decimal point)
- `O` → `0` (zero)
- `l` → `1` (one)

### Spurious Space Correction (`fix_spurious_spaces()`)

Fixes OCR artifacts that split words: `"Septe mber"` → `"September"`

### DNR Value Filtering

Filters out non-result values during structuring:
- DNR, Did Not Report, TNP, QNS

### Application Order

```
LLM Extraction Output
      │
      ▼
correct_lab_value_ocr()     ← Fix numeric OCR errors
      │
      ▼
fix_spurious_spaces()       ← Fix split words
      │
      ▼
DNR value filtering          ← Remove non-results
      │
      ▼
Structured TestResult / MedicationInfo objects
```

---

## iPhone Camera Image Handling

iPhone photos require special handling before OCR:

```
┌──────────────────────────────────────────────────────────────┐
│  iPhone Photo Processing (load_image_for_ocr)                 │
│                                                                │
│  1. HEIC/HEIF → JPEG conversion                              │
│     • Primary: pillow-heif library                            │
│     • Fallback: macOS sips command                            │
│     (Done at upload time in api/main.py)                      │
│                                                                │
│  2. EXIF orientation correction                               │
│     • ImageOps.exif_transpose()                               │
│     • Critical: 50% of iPhone photos have EXIF tag 6          │
│       (90° CW rotation)                                       │
│     • Without this: OCR sees sideways text → garbled output   │
│                                                                │
│  3. Downscale large images                                    │
│     • Max dimension: 2500px (iPhone 15 Pro: 4032x3024)       │
│     • Preserves aspect ratio                                  │
│                                                                │
│  4. RGB conversion                                             │
│     • Ensures consistent color space for OCR                  │
└──────────────────────────────────────────────────────────────┘
```

### PaddleOCR Configuration for Rotated Images

```python
# Must enable orientation detection for camera photos
paddleocr.PaddleOCR(
    use_doc_orientation_classify=True,
    use_textline_orientation=True
)
```

---

## Chat with Document

Users can ask questions about any processed document via an AI chat interface:

```
┌──────────────────────────────────────────────────────────────┐
│  CHAT FLOW                                                    │
│                                                                │
│  User sends question                                          │
│       │                                                        │
│       ▼                                                        │
│  POST /api/documents/{id}/chat                                │
│       │                                                        │
│       ▼                                                        │
│  Build context from:                                          │
│  • Extracted text (raw_text)                                  │
│  • Structured extraction results                              │
│  • Classification info                                         │
│  • Chat history (previous Q&A in session)                     │
│       │                                                        │
│       ▼                                                        │
│  LLM generates answer grounded in document context            │
│       │                                                        │
│       ▼                                                        │
│  Response returned to frontend Chat tab                       │
│                                                                │
│  Features:                                                     │
│  • Suggested questions (GET /api/documents/{id}/suggestions)  │
│  • Chat history per document                                  │
│  • Clear history (DELETE /api/documents/{id}/chat)            │
└──────────────────────────────────────────────────────────────┘
```

---

## FHIR R4 Output

The pipeline generates **FHIR R4-compliant bundles** from extraction results:

### Supported FHIR Resources

| Resource Type | Generated From | FHIR Codes |
|---------------|----------------|------------|
| `Patient` | patient_info extraction | - |
| `Observation` | Lab test results | LOINC codes |
| `MedicationRequest` | Prescription medications | RxNorm codes |
| `DiagnosticReport` | Overall lab report | LOINC panel codes |
| `Condition` | Clinical findings | - |

### Bundle Structure

```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "name": [{"given": ["John"], "family": "Doe"}]
      }
    },
    {
      "resource": {
        "resourceType": "Observation",
        "code": {
          "coding": [{"system": "http://loinc.org", "code": "718-7", "display": "Hemoglobin"}]
        },
        "valueQuantity": {"value": 14.5, "unit": "g/dL"},
        "referenceRange": [{"low": {"value": 12.0}, "high": {"value": 17.5}}]
      }
    }
  ]
}
```

---

## VLM (Vision Language Model) Integration

### Standard Mode (VLM_UNIFIED=false)

VLM is used as a **fallback** when PaddleOCR produces insufficient text:

```
    IMAGE/SCANNED PDF
          │
          ▼
    PaddleOCR Extraction
          │
          ├── Confidence >= 0.5 → Use OCR result
          │
          └── Confidence < 0.5 → VLM Fallback
                    │
                    ▼
              ┌─────────────────────────────────────┐
              │   VLM CLIENT (vlm_client.py)        │
              │                                      │
              │   Models (CPU-friendly, by size):   │
              │   1. moondream (1.8B) - fastest     │
              │   2. minicpm-v (3B) - balanced      │
              │   3. llava-phi3 (3.8B) - good       │
              │   4. llava:7b (7B) - best quality   │
              │   5. qwen3-vl:latest - all-rounder  │
              │                                      │
              │   Per-page rendering + extraction   │
              │   Timeout per page: 120s default    │
              └─────────────────────────────────────┘
                    │
                    ▼
              Structured text output
```

### VLM Unified Mode (VLM_UNIFIED=true)

Replaces PaddleOCR with VLM for **all** text extraction from images and scanned PDFs. Reduces the model stack from 3 to 2 (no PaddleOCR, no 4 OCR sub-models).

```
    IMAGE/SCANNED PDF                       DIGITAL PDF
          │                                      │
          ▼                                      ▼
    VLM (qwen3-vl) directly              pdfplumber (unchanged)
    Per-page: render → image → VLM OCR        │
    Prompt: "Transcribe ALL text exactly"      │
    num_predict: 4096 tokens                   │
          │                                      │
          ▼                                      ▼
    Raw text ──────────────────────────────→ MedGemma (classification + extraction)
```

**Key differences from standard mode:**
- PaddleOCR never loads (saves ~2GB RAM + startup time)
- VLM uses a simple OCR prompt ("transcribe faithfully") not the structured extraction prompt
- `num_predict` increased to 4096 (full medical page can be 3000+ tokens)
- Response used as raw text directly (no EXTRACTED TEXT/KEY-VALUE parsing)
- Digital PDFs still use pdfplumber (no model needed)

### VLM Configuration

```env
# Standard mode (PaddleOCR + VLM fallback)
USE_VLM=true
VLM_MODEL=qwen3-vl:latest
VLM_FALLBACK_THRESHOLD=0.5
VLM_TIMEOUT=180
VLM_UNIFIED=false

# Unified mode (VLM replaces PaddleOCR)
VLM_UNIFIED=true
VLM_MODEL=qwen3-vl:latest         # reads images
OLLAMA_MODEL=medgemma-4b-local     # classification, extraction, chat (unchanged)
```

### Performance Comparison (per page)

| | PaddleOCR (standard) | VLM (unified) |
|---|---|---|
| Clear scan | **~3-5s** | ~15-25s |
| Handwritten | Poor accuracy | **Better accuracy** |
| RAM usage | +2GB (4 sub-models) | **No extra models** |
| Dependencies | PaddleOCR + OpenCV | Just Ollama |

---

## Vector Store & Learning

The pipeline uses a vector store (SQLite + embeddings) to learn from past extractions:

```
NEW DOCUMENT → Embedding
                       │
                       ▼
              Find Similar Documents (cosine similarity)
                       │
                       ▼
              ┌─────────────────────────────────────┐
              │   EXTRACTION HINTS                   │
              │                                      │
              │   From similar documents:            │
              │   • Field examples: "Hemoglobin: 14" │
              │   • Expected fields: name, dob, mrn  │
              │   • Common patterns                  │
              └─────────────────────────────────────┘
                       │
                       ▼
              CONTENT-AGNOSTIC EXTRACTION
              (uses hints to improve accuracy)
                       │
                       ▼
              HIGH CONFIDENCE? (>= 0.7)
                       │
                       ├── YES → Store to vector store (for future learning)
                       └── NO  → Skip storage
```

### Embedding Models

| Backend | Model | Dimensions | Notes |
|---------|-------|------------|-------|
| `sentence_transformers` | `all-MiniLM-L6-v2` | 384 | **Default for Docker** — downloads automatically, no Ollama needed |
| `ollama` | `mxbai-embed-large` | 1024 | **Best quality** — top MTEB performer, requires Ollama |

### Retrieval Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `simple` | Full document text as context | Short documents (<8000 chars) |
| `chunked` | Keyword-based chunk selection | Long documents with clear sections |
| `vector` | Embedding-based semantic chunk selection | Long documents with vector store |
| `router` | Auto-select best strategy based on document | **Default - recommended** |
| `fusion` | Combine multiple strategies | Complex documents |

---

## Frontend Architecture

### React Frontend Structure

```
frontend/
├── src/
│   ├── pages/
│   │   ├── DocumentDetail/
│   │   │   ├── DocumentDetail.js    # Main document view with tabs
│   │   │   └── DocumentDetail.css
│   │   ├── ProcessFlow/
│   │   │   └── ProcessFlow.js       # Upload + real-time workflow display
│   │   └── ...
│   ├── components/
│   │   ├── ExtractedData.js         # Tabbed data display
│   │   ├── ExtractedData.css
│   │   └── shared/                   # StatusBadge, ConfidenceBadge
│   └── services/
│       └── api.js                    # API client (V2 endpoints)
```

### Document Detail Tabs

| Tab | Component | Data Source | Purpose |
|-----|-----------|-------------|---------|
| **Structured Data** | `StructuredDataView` | `result.raw_fields`, `result.extracted_values` | Tabular display of extracted fields |
| **Formatted** | `FormattedView` | `result.raw_fields`, `result.classification` | Template-based rendering by document type |
| **JSON** | `JsonView` | `result.universal_extraction` | Raw extraction data |
| **FHIR** | `FhirView` | `result.fhir_bundle` | FHIR R4 resources |
| **Bounding Boxes** | `BoundingBoxesView` | `result.bounding_boxes` | Visual field locations on PDF |
| **Chat** | Chat interface | LLM + document context | Ask questions about the document |
| **Workflow Log** | `WorkflowLogView` | `job.workflow_steps` | Real-time processing stages |

### Field Color Coding

| Panel Type | Color | Examples |
|------------|-------|----------|
| CBC | Blue | WBC, RBC, Hemoglobin, Platelets |
| Metabolic | Green | Glucose, BUN, Creatinine, Sodium |
| Lipid | Orange | Cholesterol, HDL, LDL, Triglycerides |
| Liver | Purple | ALT, AST, Bilirubin, Albumin |
| Thyroid | Teal | TSH, T3, T4, Free T4 |
| Renal | Indigo | eGFR, BUN/Creatinine ratio |
| Cardiac | Red | Troponin, BNP, CK-MB |

---

## Docker Deployment

The application ships as a Docker Compose stack for one-click deployment on Windows:

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Docker Compose Network                                      │
│                                                              │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────┐   │
│  │ frontend │    │   backend   │    │      ollama      │   │
│  │ (nginx)  │    │  (FastAPI)  │    │  (LLM server)    │   │
│  │ :80→3000 │    │    :8000    │    │     :11434       │   │
│  └──────────┘    └─────────────┘    └──────────────────┘   │
│                        ↕                     ↕              │
│                  app-data volume     ollama-data volume      │
│                                     models/ bind mount      │
└─────────────────────────────────────────────────────────────┘

Browser → http://localhost:3000 (frontend nginx)
Frontend JS → http://localhost:8000 (backend API, direct)
Backend → http://ollama:11434 (Ollama, Docker internal network)
```

### Three Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| **frontend** | Multi-stage: Node 18 build → nginx:alpine | 3000 → 80 | Serves React SPA |
| **backend** | Python 3.11 + FastAPI + PaddleOCR + ML deps | 8000 | REST API + ML pipeline |
| **ollama** | ollama/ollama:latest | 11434 | Local LLM inference server |

### Volumes

| Volume | Purpose |
|--------|---------|
| `medical-ollama-data` | Persists downloaded Ollama models between restarts |
| `medical-app-data` | Uploads, SQLite DBs (documents.db, vector_store.db, audit.db) |
| `../../models:/models:ro` | Bind mount for local GGUF model files |

### Model Setup (First Start)

Models are pulled/created at first startup via `start.bat`:

1. **MedGemma** — Created from local GGUF file: `ollama create medgemma-4b-local -f Modelfile.medgemma` (~2.5GB Q4_K_M quantized)
2. **MiniCPM-V** — Downloaded from Ollama Hub: `ollama pull minicpm-v` (~5.5GB vision model)
3. **Sentence Transformers** — Auto-downloads inside Python container on first use (~90MB)

### Windows One-Click Launcher (`start.bat`)

```
1. Check Docker Desktop installed → opens download page if missing
2. Check Docker daemon running → starts Docker Desktop if not
3. Wait for Docker to be ready (polls docker info, up to 2 min)
4. docker compose up --build -d
5. Create MedGemma model from local GGUF (first time)
6. Pull MiniCPM-V vision model (first time)
7. Wait for backend health check (polls /api/health)
8. Open http://localhost:3000 in default browser
```

### Docker Environment (`.env.docker`)

```env
BACKEND=ollama
OLLAMA_HOST=http://ollama:11434
OLLAMA_MODEL=medgemma-4b-local
USE_VLM=true
VLM_MODEL=minicpm-v
VLM_UNIFIED=false
EMBEDDING_BACKEND=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
USE_CLOUD=false
USE_CACHE=true
MAX_CONCURRENT_CHUNKS=1
```

### System Requirements (Docker)

- **Windows 10/11** (64-bit) or macOS
- **16 GB RAM** minimum
- **15 GB free disk space** (AI models + container images)
- **Internet** (first time only, for Docker + model downloads)

---

## API Endpoints

**Base URL:** `http://localhost:8000`

### Document Upload

```http
POST /api/upload
Content-Type: multipart/form-data

Body: file (PDF/image, including HEIC)

Response:
{
  "file_id": "uuid",
  "file_name": "report.pdf",
  "file_path": "/data/uploads/uuid.pdf",
  "size": 12345
}
```

Note: HEIC/HEIF files are automatically converted to JPEG on upload.

### Upload with Quality Analysis

```http
POST /api/upload-and-analyze
Content-Type: multipart/form-data

Body: file (PDF/image)
Query: force=false

Response:
{
  "file_id": "uuid",
  "file_name": "report.pdf",
  "quality": {
    "quality_level": "good",
    "quality_score": 85,
    "can_process": true,
    "issues": []
  },
  "recommendation": "proceed"
}
```

### Start Processing (V2)

```http
POST /api/v2/process?file_id={file_id}&strategy={strategy}&processing_mode={mode}

Parameters:
- file_id: ID from /api/upload
- strategy: "simple" | "router" | "fusion" | "chunked" (default: "router")
- processing_mode: "auto" | "fast" | "accurate" (default: "auto")
- skip_classification: true | false (default: false)

Response:
{
  "job_id": "uuid",
  "status": "pending",
  "pipeline_version": "v2",
  "retrieval_strategy": "router",
  "processing_mode": "auto"
}
```

### Get Job Status

```http
GET /api/v2/jobs/{job_id}

Response:
{
  "job_id": "uuid",
  "status": "completed",
  "document_type": "lab",
  "classification_confidence": 0.92,
  "total_time": 3.45,
  "confidence": 0.87,
  "requires_review": false,
  "result": {
    "extracted_values": [...],
    "raw_fields": {...},
    "sections": {...},
    "fhir_bundle": {...},
    "display_name": "CBC Panel - John Doe",
    "classification": { "type": "lab", "confidence": 0.92 },
    "universal_extraction": {...}
  },
  "workflow_steps": [...]
}
```

### Chat with Document

```http
POST /api/documents/{document_id}/chat
Content-Type: application/json

Body: { "message": "What is the hemoglobin level?" }

Response:
{
  "response": "The hemoglobin level is 14.5 g/dL, which is within the normal range...",
  "sources": [...]
}
```

### Chat Suggestions

```http
GET /api/documents/{document_id}/suggestions

Response:
{
  "suggestions": [
    "What are the abnormal values?",
    "Summarize the key findings",
    "Are there any critical values?"
  ]
}
```

### Clear Chat History

```http
DELETE /api/documents/{document_id}/chat
```

### Get Workflow Steps (Real-Time)

```http
GET /api/v2/jobs/{job_id}/workflow

Response:
{
  "steps": [
    {
      "id": "0",
      "name": "text_extraction",
      "status": "completed",
      "duration_seconds": 1.2,
      "details": {"chars_extracted": 5000, "pages": 1}
    },
    ...
  ]
}
```

### Other Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v2/samples/process` | POST | Process a sample document |
| `/api/v2/jobs` | GET | List all jobs |
| `/api/v2/jobs/{job_id}` | DELETE | Delete a job |
| `/api/classify?file_id={id}` | POST | Classify document (testing) |
| `/api/document/{file_id}/quality` | GET | Document quality analysis |
| `/api/health` | GET | Backend health check |

---

## Extraction Result Structure

### Complete Result

```json
{
  "extracted_values": [
    {
      "field_name": "hemoglobin",
      "value": "14.5",
      "unit": "g/dL",
      "confidence": 0.98,
      "reference_min": "12.0",
      "reference_max": "17.5",
      "abnormal_flag": null,
      "loinc_code": "718-7",
      "validation_status": "verified",
      "category": "lab"
    }
  ],
  "raw_fields": {
    "patient_name": "John Doe",
    "date_of_birth": "1985-03-15",
    "collection_date": "2024-01-15",
    "ordering_physician": "Dr. Smith"
  },
  "sections": {
    "findings": ["..."],
    "impressions": ["..."],
    "patient_info": {...},
    "provider_info": {...}
  },
  "clinical_summary": "Complete blood count within normal limits...",
  "critical_findings": [],
  "classification": {
    "type": "lab",
    "confidence": 0.92,
    "method": "llm_fingerprint"
  },
  "document_type": "lab",
  "display_name": "CBC Panel - John Doe",
  "confidence": 0.95,
  "requires_review": false,
  "review_reasons": [],
  "warnings": [],
  "fhir_bundle": {
    "resourceType": "Bundle",
    "type": "collection",
    "entry": [...]
  },
  "universal_extraction": {
    "test_results": [...],
    "medications": [...],
    "findings": [...],
    "patient_info": {...},
    "raw_fields": {...}
  },
  "enriched_extraction": {
    "enrichment_type": "lab",
    "enrichments": {
      "loinc_codes": [...],
      "critical_values": []
    }
  },
  "pipeline_stages": [...],
  "total_time": 3.45
}
```

### Document Types

| Type | Classification | Processing |
|------|---------------|------------|
| `lab` | Lab report detected | LabEnricher: LOINC codes, reference ranges, validation |
| `radiology` | Radiology report detected | RadiologyEnricher: critical findings |
| `prescription` | Prescription detected | PrescriptionEnricher: RxNorm codes, drug interactions |
| `pathology` | Pathology report detected | PathologyEnricher: TNM staging |
| `insurance` | Insurance document detected | Raw fields formatted for claims |
| `unknown` | Not classified | Best-effort extraction, flagged for review |

---

## Configuration

### Environment Variables (.env)

```env
# ============================================================================
# LLM BACKEND
# ============================================================================
BACKEND=ollama                      # ollama | openai | local
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=medgemma-4b-local      # Classification, extraction, chat

# ============================================================================
# VLM (VISION LANGUAGE MODEL)
# ============================================================================
USE_VLM=true
VLM_MODEL=qwen3-vl:latest          # moondream | minicpm-v | llava-phi3 | qwen3-vl
VLM_TIMEOUT=180
VLM_FALLBACK_THRESHOLD=0.5

# VLM Unified: VLM replaces PaddleOCR for text extraction
# MedGemma still handles classification, extraction, and chat
VLM_UNIFIED=false

# ============================================================================
# EXTRACTION SETTINGS
# ============================================================================
USE_OCR=true
USE_VISION=true
USE_CONSENSUS_EXTRACTION=false      # Disable for Ollama (causes timeouts)
MAX_TEXT_LENGTH=4000                 # Increased from 2500 for lab reports
MAX_CONCURRENT_CHUNKS=1             # 1=sequential (safe for Ollama), >1=parallel

# ============================================================================
# EMBEDDING / VECTOR STORE
# ============================================================================
EMBEDDING_BACKEND=sentence_transformers  # sentence_transformers | ollama
EMBEDDING_MODEL=all-MiniLM-L6-v2        # or mxbai-embed-large for Ollama
EMBEDDING_DIM=384                        # 384 for MiniLM, 1024 for mxbai

# ============================================================================
# CACHING
# ============================================================================
USE_CACHE=true
CACHE_DIR=./ollama_cache
CACHE_MAX_SIZE=500
CACHE_TTL=3600

# ============================================================================
# CLOUD (disabled for local-only deployments)
# ============================================================================
USE_CLOUD=false

# ============================================================================
# GENERAL
# ============================================================================
DATA_DIR=data
LOG_LEVEL=INFO
TEMPERATURE=0.1
MAX_TOKENS=1000
TIMEOUT=120
```

---

## Data Storage

| Path | Purpose |
|------|---------|
| `data/uploads/` | Uploaded documents (JPEG, PDF) |
| `data/documents.db` | Document metadata and job results |
| `data/vector_store.db` | SQLite vector store for similar document matching |
| `data/audit.db` | Processing audit trail |
| `data/samples/` | Sample documents for testing |
| `data/lab_tests/lab_tests.db` | LOINC database (96K+ codes, ~59MB) |
| `data/medications/medications.db` | RxNorm database (~30MB) |
| `ollama_cache/` | LLM response cache (disk-persisted) |

---

## Ports

| Service | Port | Description |
|---------|------|-------------|
| FastAPI | 8000 | REST API |
| React | 3000 | Frontend (nginx in Docker) |
| Ollama | 11434 | LLM inference |

---

## Model Requirements

### Standard Mode (VLM_UNIFIED=false) — 3 model stack

| Model | Size | Purpose |
|-------|------|---------|
| MedGemma 4B | ~2.5GB (Q4_K_M GGUF) | Classification + structured extraction + chat |
| qwen3-vl (or minicpm-v) | ~4-5GB | VLM fallback for images with poor OCR |
| PaddleOCR | ~100MB (4 sub-models) | Primary OCR engine (auto-downloads) |
| all-MiniLM-L6-v2 | ~90MB | Document embeddings (sentence_transformers) |

### Unified Mode (VLM_UNIFIED=true) — 2 model stack

| Model | Size | Purpose |
|-------|------|---------|
| MedGemma 4B | ~2.5GB (Q4_K_M GGUF) | Classification + structured extraction + chat |
| qwen3-vl | ~4-5GB | Text extraction from images/scans (replaces PaddleOCR) |
| all-MiniLM-L6-v2 | ~90MB | Document embeddings (sentence_transformers) |

PaddleOCR never loads in unified mode — no 4 sub-models, ~2GB less RAM.

### Optional Models

| Model | Size | Purpose |
|-------|------|---------|
| mxbai-embed-large | ~700MB | Higher quality embeddings (via Ollama) |

---

## Two Pipeline Paths

The system supports two processing paths:

| Path | Entry Point | LLM Backend | Enrichment |
|------|-------------|-------------|------------|
| **V1 (Orchestrator)** | `core/orchestrator.py` | Ollama/local | Inline with processor |
| **V2 (Extraction-First)** | `core/extraction_first_pipeline.py` | Any (Ollama/OpenAI) | Post-extraction enrichers |

Both paths:
- Set `validation_status` on medications and test results
- Generate `display_name` for UI
- Produce FHIR R4 bundles
- Support the chat-with-document feature

---

## Confidence Levels

| Level | Score | Meaning |
|-------|-------|---------|
| HIGH | > 0.85 | Reliable extraction, no review needed |
| MEDIUM | 0.70 - 0.85 | Mostly reliable, spot check recommended |
| LOW | 0.50 - 0.70 | Uncertain, human review required |
| VERY LOW | < 0.50 | Extraction methods struggled, must review |

---

## Review Escalation Triggers

Documents are flagged for human review (`requires_review: true`) when:

1. **Low confidence** — Overall confidence < 0.60
2. **Extraction failed** — Content extraction returned empty
3. **Low text content** — Very little text extracted (mostly images)
4. **No content extracted** — No test results, medications, or findings found
5. **Classification failed** — Document type couldn't be determined
6. **VLM fallback used** — OCR confidence was too low
7. **Handwriting detected** — Documents with handwriting flagged for verification
8. **Critical findings** — Urgent abnormalities detected

---

## Architecture Decisions

### Why Extraction-First?

1. **Extraction works without classification** — Generic prompts extract ALL medical content regardless of document type
2. **Better classification accuracy** — Classification uses clean extracted text instead of raw OCR
3. **Simpler architecture** — Clear separation: extraction → classification → enrichment
4. **Universal handling** — Same prompts work for labs, prescriptions, radiology, etc.

### Why Single LLM Call?

1. **Faster** — 1 Ollama round-trip instead of 5 (4 minutes → 1 minute per document)
2. **More consistent** — LLM sees full document context at once
3. **Simpler** — No coordination of parallel calls or result merging

### Why CLINICAL_NAME_OVERRIDES?

1. **LOINC ambiguity** — "Hemoglobin" matches HGB (718-7), MCH (785-6), and MCHC (786-4) by component
2. **Rank isn't enough** — `common_test_rank = 0` means unranked, not highest priority
3. **Hardcoded overrides** — Bypass fuzzy search for known clinical names that always mean one specific test

### Why OCR-Tolerant Database Matching?

1. **Real-world OCR errors** — Medical documents are often scanned, faxed, or photographed
2. **Handwriting** — Prescription drugs written by hand produce character confusions (t↔l, rn↔m)
3. **Progressive fallback** — Exact → alias → prefix → OCR variant → fuzzy (maximizes match rate)

### Why Docker for Deployment?

1. **Zero coding required** — Non-technical users double-click `start.bat`
2. **No source code exposure** — Docker images are opaque binaries
3. **Handles complexity** — Python + Node + Ollama + models in one command
4. **Reproducible** — Same environment everywhere, no "works on my machine"

### Why Real-Time Progress?

1. **User feedback** — Users see what's happening, not just a spinner
2. **Debugging** — Easy to identify which stage failed
3. **Transparency** — Builds trust in the extraction process
