# ============================================================================
# api/main.py
# ============================================================================
"""
FastAPI Backend for Medical Ingestion Engine

Runs on port 8000 (Streamlit runs on 8501).
Provides REST API for document processing and results retrieval.
"""

import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from io import BytesIO

logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contextlib import asynccontextmanager

from medical_ingestion.core import get_config, ProcessingContext
from medical_ingestion.classifiers.document_classifier import DocumentClassifier
from medical_ingestion.utils.image_quality import ImageQualityAnalyzer, QualityLevel
from medical_ingestion.core.document_store import DocumentStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavy models at startup so first request is fast."""
    logger.info("Pre-loading PaddleOCR models...")
    try:
        from medical_ingestion.extractors.paddle_ocr import get_paddle_ocr_extractor
        extractor = get_paddle_ocr_extractor()
        extractor._ensure_initialized()
        logger.info("PaddleOCR models loaded successfully")
    except Exception as e:
        logger.warning(f"PaddleOCR pre-load failed (will retry on first use): {e}")
    yield


app = FastAPI(
    title="Medical Ingestion Engine API",
    description="API for processing medical documents with AI extraction",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for React frontend and Expo mobile app
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        # Expo development
        "http://localhost:8081",
        "http://localhost:19000",
        "http://localhost:19001",
        "http://localhost:19002",
    ],
    # Allow all origins for mobile device testing (Expo Go)
    allow_origin_regex=r"http://192\.168\.\d+\.\d+:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Models
# ============================================================================

class ProcessingRequest(BaseModel):
    document_type: str  # "lab", "radiology", "prescription"
    file_path: Optional[str] = None


class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    document_type: str
    file_name: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WorkflowStep(BaseModel):
    id: str
    name: str
    status: str  # "pending", "running", "completed", "failed"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# ============================================================================
# Background Processing (v1 removed - use v2 endpoints)
# ============================================================================
# V1 processing has been removed. Use /api/v2/process instead.


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Medical Ingestion Engine API"}


@app.get("/api/health")
async def health():
    """Health check for monitoring."""
    return {"status": "healthy"}


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    force: bool = False,  # Allow forcing upload even if quality is poor
    background_tasks: BackgroundTasks = None
):
    """
    Upload a document for processing.

    REJECTS images that don't meet quality requirements (unless force=True).
    """
    import tempfile
    from medical_ingestion.utils.image_utils import is_image_file, load_image_for_ocr

    # Read file content into memory first
    content = await file.read()
    file_ext = Path(file.filename).suffix.lower()
    is_pdf = file_ext == '.pdf'

    # Convert HEIC/HEIF to JPEG immediately on upload so all downstream
    # components get a standard format they can handle.
    heic_converted = False
    if file_ext in ('.heic', '.heif'):
        try:
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
                tmp.write(content)
                heic_tmp = Path(tmp.name)
            img = load_image_for_ocr(heic_tmp, max_dimension=99999, ensure_rgb=True)
            buf = BytesIO()
            img.save(buf, format='JPEG', quality=95)
            content = buf.getvalue()
            file_ext = '.jpg'
            heic_converted = True
            heic_tmp.unlink(missing_ok=True)
            logger.info(f"Converted HEIC upload to JPEG ({len(content)} bytes)")
        except Exception as e:
            logger.warning(f"HEIC conversion failed, saving as-is: {e}")
            if 'heic_tmp' in locals():
                Path(heic_tmp).unlink(missing_ok=True)

    # For images, check quality BEFORE saving permanently
    if not is_pdf:
        # Save to temp file for quality analysis
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            if is_image_file(tmp_path):
                analyzer = ImageQualityAnalyzer()
                report = analyzer.analyze(tmp_path)

                # REJECT if unprocessable (unless forced)
                if report.quality_level == QualityLevel.UNPROCESSABLE and not force:
                    # Clean up temp file
                    tmp_path.unlink(missing_ok=True)

                    # Return error with quality details
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "image_quality_too_low",
                            "message": report.summary,
                            "quality": report.to_dict(),
                            "tips": report.tips if hasattr(report, 'tips') else [
                                "Use better lighting",
                                "Hold camera steady and closer to document",
                                "Ensure document is flat and in focus",
                                "Minimum resolution: 600x450 pixels"
                            ],
                            "can_force": True,
                        }
                    )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"Quality analysis failed, proceeding anyway: {e}")
        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

    # Quality OK (or PDF) - save to permanent location
    file_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / f"{file_id}{file_ext}"

    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "file_id": file_id,
        "file_name": file.filename,
        "file_path": str(file_path),
        "size": len(content),
        "heic_converted": heic_converted,
    }


@app.post("/api/classify-file")
async def classify_file_direct(file_path: str):
    """
    Classify a file directly by path (for testing with sample files).

    This runs classification directly without uploading first.
    Useful for quick testing of the classification pipeline.

    Args:
        file_path: Full path to the file to classify

    Returns:
        Classification result with type, confidence, method, and reasoning
    """
    from medical_ingestion.extractors.text_extractor import TextExtractor

    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    config = get_config()

    try:
        # Extract text
        text_extractor = TextExtractor()
        extraction_result = text_extractor.extract_text_detailed(
            path,
            preserve_layout=False
        )

        # Create context
        context = ProcessingContext(document_path=path)
        context.raw_text = extraction_result.text
        context.total_pages = extraction_result.page_count

        # Run classification
        classifier = DocumentClassifier(config)
        classification = await classifier.execute(context)

        return {
            "file_path": str(path),
            "file_name": path.name,
            "classification": {
                "type": classification.get("type", "unknown"),
                "confidence": classification.get("confidence", 0.0),
                "method": classification.get("method", "unknown"),
                "reasoning": classification.get("reasoning", "")
            },
            "text_length": len(context.raw_text),
            "pages": extraction_result.page_count,
            "layout": context.sections.get('_layout', {})
        }

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.post("/api/classify")
async def classify_document(file_id: str):
    """
    Classify a document synchronously (for testing classification only).

    This runs classification directly without background tasks, returning
    the result immediately. Useful for testing the classification pipeline.

    Args:
        file_id: ID of the uploaded file

    Returns:
        Classification result with type, confidence, method, and reasoning
    """
    from medical_ingestion.extractors.text_extractor import TextExtractor

    # Find the file
    files = list(UPLOADS_DIR.glob(f"{file_id}.*"))
    if not files:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = files[0]
    config = get_config()

    try:
        # Extract text
        text_extractor = TextExtractor()
        extraction_result = text_extractor.extract_text_detailed(
            file_path,
            preserve_layout=False
        )

        # Create context
        context = ProcessingContext(document_path=file_path)
        context.raw_text = extraction_result.text
        context.total_pages = extraction_result.page_count

        # Run classification
        classifier = DocumentClassifier(config)
        classification = await classifier.execute(context)

        return {
            "file_id": file_id,
            "file_name": file_path.name,
            "classification": {
                "type": classification.get("type", "unknown"),
                "confidence": classification.get("confidence", 0.0),
                "method": classification.get("method", "unknown"),
                "reasoning": classification.get("reasoning", "")
            },
            "text_length": len(context.raw_text),
            "pages": extraction_result.page_count,
            "layout": context.sections.get('_layout', {})
        }

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.get("/api/document/{file_id}")
async def get_document(file_id: str, as_pdf: bool = True):
    """
    Get uploaded document for viewing.

    Args:
        file_id: The file ID
        as_pdf: If True, convert images to PDF for viewing (default: True)

    Returns:
        The document file (converted to PDF if needed)
    """
    from medical_ingestion.utils.image_utils import (
        is_image_file,
        convert_image_to_pdf_bytes
    )

    files = list(UPLOADS_DIR.glob(f"{file_id}.*"))
    if not files:
        raise HTTPException(status_code=404, detail="File not found")

    # Prefer the original image over any leftover .pdf conversion
    non_pdf = [f for f in files if f.suffix.lower() != '.pdf']
    file_path = non_pdf[0] if non_pdf else files[0]

    # If it's already a PDF or we don't need conversion, return as-is
    if file_path.suffix.lower() == '.pdf' or not as_pdf:
        return FileResponse(file_path)

    # Check if it's an image that needs conversion to PDF for the viewer
    if is_image_file(file_path):
        try:
            # Convert in memory — no disk write to avoid cluttering uploads/
            pdf_bytes = convert_image_to_pdf_bytes(file_path)
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f"inline; filename={file_path.stem}.pdf"}
            )
        except Exception as e:
            logger.error(f"Failed to convert image to PDF: {e}")
            # Fall back to returning the original image
            return FileResponse(file_path)

    # Unknown file type, return as-is
    return FileResponse(file_path)


@app.get("/api/document/{file_id}/info")
async def get_document_info(file_id: str):
    """
    Get information about an uploaded document.

    Returns:
        Document metadata including type, size, dimensions for images
    """
    from medical_ingestion.utils.image_utils import (
        is_image_file,
        detect_image_type,
        get_image_dimensions
    )

    files = list(UPLOADS_DIR.glob(f"{file_id}.*"))
    if not files:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = files[0]
    file_stat = file_path.stat()

    info = {
        "file_id": file_id,
        "filename": file_path.name,
        "extension": file_path.suffix.lower(),
        "size_bytes": file_stat.st_size,
        "is_pdf": file_path.suffix.lower() == '.pdf',
        "is_image": is_image_file(file_path),
    }

    # Add image-specific info
    if info["is_image"]:
        info["image_type"] = detect_image_type(file_path)
        dimensions = get_image_dimensions(file_path)
        if dimensions:
            info["width"], info["height"] = dimensions

        # Check if PDF conversion exists
        pdf_path = file_path.with_suffix('.pdf')
        info["pdf_available"] = pdf_path.exists()

    return info


@app.get("/api/document/{file_id}/quality")
async def analyze_document_quality(file_id: str):
    """
    Analyze the quality of an uploaded document (image).

    Returns quality metrics and actionable feedback for the user:
    - Quality score (0-100)
    - Whether the document can be processed
    - Specific issues detected (blur, low resolution, poor contrast, etc.)
    - Tips for better image capture
    - Whether handwriting was detected

    Returns:
        QualityReport with metrics, issues, and recommendations
    """
    from medical_ingestion.utils.image_utils import is_image_file

    files = list(UPLOADS_DIR.glob(f"{file_id}.*"))
    if not files:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = files[0]

    # Check if it's an image
    if not is_image_file(file_path) and file_path.suffix.lower() != '.pdf':
        raise HTTPException(
            status_code=400,
            detail="Quality analysis is only available for image files"
        )

    # For PDFs, we could analyze the first page as an image, but for now
    # we'll focus on direct image uploads
    if file_path.suffix.lower() == '.pdf':
        return {
            "quality_level": "unknown",
            "quality_score": None,
            "can_process": True,
            "summary": "PDF documents are processed directly without image quality analysis.",
            "tips": [],
            "issues": [],
            "is_pdf": True
        }

    try:
        analyzer = ImageQualityAnalyzer()
        report = analyzer.analyze(file_path)

        return report.to_dict()

    except Exception as e:
        logger.error(f"Quality analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quality analysis failed: {str(e)}")


@app.post("/api/upload-and-analyze")
async def upload_and_analyze_quality(
    file: UploadFile = File(...),
    force: bool = False,  # Allow forcing upload even if quality is poor
):
    """
    Upload a document and immediately analyze its quality.

    This endpoint combines upload and quality analysis for immediate user feedback.
    REJECTS uploads that don't meet quality requirements (unless force=True).

    Args:
        file: The document to upload
        force: If True, accept the upload even if quality is unprocessable

    Returns:
        - file_id: ID for subsequent operations
        - quality: Full quality analysis report (for images)
        - recommendation: "proceed", "warning", or "retake"

    Raises:
        HTTPException 400: If image quality is too low to process (unless force=True)
    """
    import tempfile
    from medical_ingestion.utils.image_utils import is_image_file

    # Read file content into memory first
    content = await file.read()
    file_ext = Path(file.filename).suffix.lower()
    is_pdf = file_ext == '.pdf'

    # For images, check quality BEFORE saving permanently
    if not is_pdf:
        # Save to temp file for quality analysis
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            if is_image_file(tmp_path):
                analyzer = ImageQualityAnalyzer()
                report = analyzer.analyze(tmp_path)

                # REJECT if unprocessable (unless forced)
                if report.quality_level == QualityLevel.UNPROCESSABLE and not force:
                    # Clean up temp file
                    tmp_path.unlink(missing_ok=True)

                    # Return error with quality details so frontend can show feedback
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "image_quality_too_low",
                            "message": report.summary,
                            "quality": report.to_dict(),
                            "tips": report.tips if hasattr(report, 'tips') else [
                                "Use better lighting",
                                "Hold camera steady and closer to document",
                                "Ensure document is flat and in focus",
                                "Minimum resolution: 600x450 pixels"
                            ],
                            "can_force": True,  # Tell frontend they can retry with force=True
                        }
                    )
        except HTTPException:
            raise  # Re-raise our quality error
        except Exception as e:
            logger.warning(f"Quality analysis failed, proceeding anyway: {e}")
            report = None
        finally:
            # Clean up temp file - we'll save to permanent location next
            tmp_path.unlink(missing_ok=True)
    else:
        report = None

    # Quality OK (or PDF) - save to permanent location
    file_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / f"{file_id}{file_ext}"

    with open(file_path, "wb") as f:
        f.write(content)

    result = {
        "file_id": file_id,
        "file_name": file.filename,
        "file_path": str(file_path),
        "size": len(content),
        "is_image": not is_pdf and is_image_file(file_path),
        "is_pdf": is_pdf,
    }

    # Add quality info to result
    if report:
        result["quality"] = report.to_dict()
        if report.quality_level == QualityLevel.POOR:
            result["recommendation"] = "warning"
            result["message"] = report.summary
        else:
            result["recommendation"] = "proceed"
            result["message"] = report.summary
    else:
        result["quality"] = None
        result["recommendation"] = "proceed"
        result["message"] = "Document uploaded successfully."

    return result


# ============================================================================
# Sample Data Endpoints (for demo)
# ============================================================================

@app.get("/api/samples")
async def list_samples():
    """List sample documents available for processing."""
    samples_dir = DATA_DIR / "samples"
    samples = []

    if samples_dir.exists():
        for category in ["labs", "radiology", "prescriptions"]:
            category_dir = samples_dir / category
            if category_dir.exists():
                for file_path in category_dir.rglob("*"):
                    if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg"]:
                        samples.append({
                            "name": file_path.name,
                            "path": str(file_path),
                            "category": category,
                            "type": "lab" if category == "labs" else category.rstrip("s")
                        })

    return {"samples": samples}


# ============================================================================
# V2 API Endpoints - Extraction-First Pipeline
# ============================================================================

# V2 in-memory job storage (active/in-progress jobs)
v2_processing_jobs: Dict[str, Dict[str, Any]] = {}

# Persistent document store (SQLite — survives restarts)
document_store = DocumentStore()

# Chat history per job (in-memory, cleared on restart)
chat_histories: Dict[str, List[Dict[str, str]]] = {}


def _transform_v2_result_for_frontend(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform v2 ExtractionFirstResult into frontend-compatible format.

    The frontend expects fields like:
    - extracted_values: list of {field_name, value, unit, confidence, ...}
    - sections: dict of section_name -> content
    - clinical_summary: string
    - raw_text: string
    - critical_findings: list
    - raw_fields: dict of all key-value pairs (for Structured Data tab)
    - classification: dict with type, confidence
    - universal_extraction: original extraction data (for JSON tab)
    """
    universal = result_dict.get("universal_extraction", {}) or {}
    if not isinstance(universal, dict):
        universal = {}
    classification = result_dict.get("classification", {}) or {}
    if not isinstance(classification, dict):
        classification = {}
    text_result = result_dict.get("text_result", {}) or {}
    if not isinstance(text_result, dict):
        text_result = {}

    # Build extracted_values from test_results + medications
    extracted_values = []

    # Helper to parse reference range string into min/max
    def parse_reference_range(ref_range: str) -> tuple:
        """Parse reference range string like '3.4-10.8' into (min, max)."""
        if not ref_range:
            return None, None
        import re
        # Handle "10-20", "10.0-20.0" format
        match = re.match(r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)', str(ref_range))
        if match:
            return match.group(1), match.group(2)
        return None, None

    # Add test results
    test_results = universal.get("test_results", [])
    logger.debug(f"Transforming {len(test_results)} test results for frontend")
    for test in test_results:
        test_name = test.get("name", "")
        logger.debug(f"  Adding test: {test_name} = {test.get('value')}")

        # Parse reference_range string into min/max
        ref_range = test.get("reference_range", "")
        ref_min, ref_max = parse_reference_range(ref_range)

        extracted_values.append({
            "field_name": test_name,
            "value": test.get("value", ""),
            "unit": test.get("unit", ""),
            "confidence": test.get("confidence", 0.0),
            "reference_min": ref_min,
            "reference_max": ref_max,
            "abnormal_flag": test.get("abnormal_flag"),
            "category": test.get("category", ""),
            "validation_status": test.get("validation_status"),
            "loinc_code": test.get("loinc_code"),
            "loinc_name": test.get("loinc_name"),
        })

    # Add medications as extracted values too
    for med in universal.get("medications", []):
        extracted_values.append({
            "field_name": med.get("name", "Medication"),
            "value": f"{med.get('name', '')} {med.get('dosage', '')}".strip(),
            "unit": med.get("frequency", ""),
            "confidence": med.get("confidence", 0.0),
            "category": "medication",
        })

    # Build sections from findings and raw_fields
    sections = {}

    # Add findings as a section
    findings = universal.get("findings", [])
    if findings:
        sections["findings"] = [
            f.get("finding", f.get("description", str(f))) if isinstance(f, dict) else str(f)
            for f in findings
        ]

    # Add impressions
    impressions = universal.get("impressions", [])
    if impressions:
        sections["impressions"] = impressions

    # Add raw_fields as a section for display (filter out internal _ keys)
    raw_fields = universal.get("raw_fields", {})
    if not isinstance(raw_fields, dict):
        raw_fields = {}
    display_raw_fields = {k: v for k, v in raw_fields.items() if not k.startswith('_')}
    if display_raw_fields:
        sections["extracted_fields"] = display_raw_fields

    # Add patient info if available (may be list or dict depending on extraction)
    # Extraction produces "patient" key, but some paths use "patient_info"
    patient_info = universal.get("patient_info") or universal.get("patient") or {}
    if not isinstance(patient_info, dict):
        patient_info = {}
    if patient_info:
        sections["patient_info"] = patient_info

    # Add provider info if available (extraction may use "providers" list or "provider_info" dict)
    provider_info = universal.get("provider_info") or {}
    if not isinstance(provider_info, dict):
        provider_info = {}
    providers_list = universal.get("providers", [])
    if not isinstance(providers_list, list):
        providers_list = []
    if provider_info:
        sections["provider_info"] = provider_info
    if providers_list:
        sections["providers"] = providers_list

    # Add dates (may be dict or list depending on extraction)
    dates = universal.get("dates") or {}
    if dates:
        sections["dates"] = dates

    # Build clinical summary from impressions or findings
    clinical_summary = ""
    if impressions:
        clinical_summary = " ".join(impressions[:3])
    elif findings:
        clinical_summary = " ".join([
            f.get("finding", f.get("description", str(f))) if isinstance(f, dict) else str(f)
            for f in findings[:3]
        ])

    # Get raw text
    raw_text = text_result.get("full_text", "")

    # Generate friendly display name from extracted metadata
    doc_type_label = classification.get("type", "unknown").replace("_", " ").title()
    display_parts = [doc_type_label]
    # patient_info and dates may be lists or dicts depending on extraction
    pi = patient_info if isinstance(patient_info, dict) else {}
    dt = dates if isinstance(dates, dict) else {}
    p_name = (pi.get("name") or pi.get("patient_name") or "").strip()
    if p_name:
        display_parts.append(p_name.title() if p_name.isupper() or p_name.islower() else p_name)
    doc_date = (dt.get("collection_date") or dt.get("report_date") or "")
    if doc_date:
        display_parts.append(str(doc_date)[:10])

    # Build the frontend-compatible result
    frontend_result = {
        # Core extraction data (for Structured Data tab)
        "extracted_values": extracted_values,
        "sections": sections,
        "clinical_summary": clinical_summary,
        "raw_text": raw_text,
        "critical_findings": universal.get("critical_findings", []),

        # Raw fields dict (all key-value pairs for tabular display, no internal keys)
        "raw_fields": display_raw_fields,

        # Classification info
        "classification": classification,
        "document_type": classification.get("type", "unknown"),
        "display_name": " - ".join(display_parts),

        # Confidence and review
        "confidence": result_dict.get("confidence", 0.0),
        "requires_review": result_dict.get("requires_review", False),
        "review_reasons": result_dict.get("review_reasons", []),
        "warnings": result_dict.get("warnings", []),

        # Original v2 data (for JSON tab - complete extraction)
        "universal_extraction": universal,
        "enriched_extraction": result_dict.get("enriched_extraction"),

        # Text extraction metadata
        "text_result": text_result,
        "total_pages": text_result.get("page_count", 1),

        # Pipeline metadata
        "pipeline_stages": result_dict.get("pipeline_stages", []),
        "total_time": result_dict.get("total_time", 0.0),

        # FHIR bundle — generated from extracted data
        "fhir_bundle": None,  # populated below

        # Bounding boxes from GPT-4o vision (stored in raw_fields by azure client)
        "bounding_boxes": raw_fields.get('_bounding_boxes', []),
    }

    # Generate FHIR R4 bundle from the extracted data
    try:
        from medical_ingestion.fhir_utils.builder import FHIRBuilder
        fhir_builder = FHIRBuilder()
        frontend_result["fhir_bundle"] = fhir_builder.build_from_v2_result(frontend_result)
    except Exception as e:
        logger.warning(f"FHIR bundle generation failed: {e}")

    return frontend_result


async def process_document_v2(
    job_id: str,
    file_path: Path,
    retrieval_strategy: str = "router",
    skip_classification: bool = False,
    document_type_override: Optional[str] = None,
    processing_mode: str = "auto"
):
    """
    Process document using extraction-first pipeline (v2).

    Flow:
    1. Universal text extraction
    2. Content-agnostic extraction (parallel with classification)
    3. Type-specific enrichment
    """
    from medical_ingestion.core.extraction_first_pipeline import ExtractionFirstPipeline, PipelineStage

    job = v2_processing_jobs[job_id]
    job["status"] = "processing"
    job["started_at"] = datetime.now().isoformat()
    job["document_type"] = "auto"  # Initialize with auto

    config = dict(get_config())  # Copy! get_config() is @lru_cache — never mutate the original
    config["default_retrieval_strategy"] = retrieval_strategy
    config["skip_classification"] = skip_classification

    # Apply processing mode overrides
    if processing_mode == "fast":
        config["use_vlm"] = False
        config["use_consensus_extraction"] = False
    elif processing_mode == "accurate":
        config["use_vlm"] = True
        config["force_vlm_all_pages"] = True
        config["max_pages_for_vlm_fallback"] = 999

    if document_type_override:
        job["document_type_override"] = document_type_override

    def progress_callback(stage: PipelineStage, all_stages: list):
        """
        Real-time progress callback that updates job workflow_steps.

        Called whenever a pipeline stage starts or completes.
        """
        # Convert all stages to workflow_steps format
        job["workflow_steps"] = [
            {
                "id": str(i),
                "name": s.name,
                "status": s.status,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                "duration_seconds": s.duration_seconds,
                "details": s.details
            }
            for i, s in enumerate(all_stages)
        ]
        logger.info(f"Pipeline progress: {stage.name} -> {stage.status}")

    try:
        # Run extraction-first pipeline with progress callback
        pipeline = ExtractionFirstPipeline(config)
        result = await pipeline.process(
            document_path=file_path,
            classification_hint=document_type_override,
            retrieval_strategy=retrieval_strategy,
            progress_callback=progress_callback
        )

        # Convert result to dict for storage
        result_dict = result.to_dict()

        # Extract document type from classification
        doc_type = "unknown"
        classification_confidence = 0.0
        if result.classification:
            doc_type = result.classification.get("type", "unknown")
            classification_confidence = result.classification.get("confidence", 0.0)

        # Transform result to frontend-compatible format
        # The frontend expects: extracted_values, sections, clinical_summary, etc.
        frontend_result = _transform_v2_result_for_frontend(result_dict)

        # Update job with document type
        job["document_type"] = doc_type
        job["classification_confidence"] = classification_confidence

        # Update job
        job["status"] = "completed"
        job["completed_at"] = datetime.now().isoformat()
        job["result"] = frontend_result

        # Add pipeline-specific metadata
        job["pipeline_version"] = "v2"
        job["retrieval_strategy"] = retrieval_strategy
        job["total_time"] = result.total_time
        job["confidence"] = result.confidence
        job["requires_review"] = result.requires_review
        job["review_reasons"] = result.review_reasons

        # Add workflow steps from pipeline stages
        job["workflow_steps"] = [
            {
                "id": str(i),
                "name": stage.name,
                "status": stage.status,
                "started_at": stage.started_at.isoformat() if stage.started_at else None,
                "completed_at": stage.completed_at.isoformat() if stage.completed_at else None,
                "duration_seconds": stage.duration_seconds,
                "details": stage.details
            }
            for i, stage in enumerate(result.pipeline_stages)
        ]

        # Persist to SQLite so results survive restarts
        try:
            document_store.save(job)
        except Exception as db_err:
            logger.error(f"Failed to persist document to store: {db_err}")

    except Exception as e:
        logger.error(f"V2 processing failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()

        # Persist failures too so users can see what went wrong
        try:
            document_store.save(job)
        except Exception as db_err:
            logger.error(f"Failed to persist failed job to store: {db_err}")


@app.post("/api/v2/process")
async def start_processing_v2(
    file_id: str,
    background_tasks: BackgroundTasks,
    strategy: str = "router",
    skip_classification: bool = False,
    document_type: Optional[str] = None,
    processing_mode: str = "auto"
):
    """
    Start processing using extraction-first pipeline (v2).

    This endpoint uses the new Unstract-inspired flow:
    1. Universal text extraction (any format → clean text)
    2. Content-agnostic extraction (parallel with classification)
    3. Type-specific enrichment (LOINC, RxNorm, etc.)

    Args:
        file_id: ID of the uploaded file
        strategy: Retrieval strategy ("simple", "router", "fusion", "chunked")
        skip_classification: If True, skip classification step
        document_type: Optional document type override (skip auto-classification)

    Returns:
        job_id for tracking processing status
    """
    # Find the file
    files = list(UPLOADS_DIR.glob(f"{file_id}.*"))
    if not files:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = files[0]

    # Create job
    job_id = str(uuid.uuid4())
    v2_processing_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "file_name": file_path.name,
        "file_path": str(file_path),
        "created_at": datetime.now().isoformat(),
        "pipeline_version": "v2",
        "retrieval_strategy": strategy,
        "processing_mode": processing_mode,
        "workflow_steps": []
    }

    # Start background processing
    background_tasks.add_task(
        process_document_v2, job_id, file_path, strategy, skip_classification, document_type, processing_mode
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "pipeline_version": "v2",
        "retrieval_strategy": strategy
    }


@app.get("/api/v2/jobs/{job_id}")
async def get_job_status_v2(job_id: str):
    """Get processing job status (v2 pipeline).

    Checks in-memory first (active jobs), then falls back to SQLite.
    """
    # In-memory first (active / in-progress jobs)
    if job_id in v2_processing_jobs:
        return v2_processing_jobs[job_id]

    # Fall back to persistent store
    stored = document_store.get(job_id)
    if stored:
        return stored

    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/v2/jobs/{job_id}/workflow")
async def get_workflow_steps_v2(job_id: str):
    """Get workflow steps for a v2 job."""
    # In-memory first
    if job_id in v2_processing_jobs:
        return {"steps": v2_processing_jobs[job_id].get("workflow_steps", [])}

    # Fall back to persistent store
    stored = document_store.get(job_id)
    if stored:
        return {"steps": stored.get("workflow_steps", [])}

    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/api/v2/jobs")
async def list_jobs_v2():
    """List all v2 processing jobs.

    Merges in-memory active jobs with persisted DB jobs.
    In-memory version wins if a job exists in both (fresher data).
    """
    # Start with all persisted documents
    db_jobs = document_store.list_all()

    # Build a dict keyed by job_id — DB first, then overlay in-memory
    merged: Dict[str, Dict[str, Any]] = {j["job_id"]: j for j in db_jobs}
    merged.update(v2_processing_jobs)   # in-memory wins for active jobs

    # Sort newest first
    jobs = sorted(merged.values(), key=lambda j: j.get("created_at", ""), reverse=True)
    return {"jobs": jobs}


@app.post("/api/v2/samples/process")
async def process_sample_v2(
    file_path: str,
    background_tasks: BackgroundTasks,
    strategy: str = "router",
    skip_classification: bool = False
):
    """
    Process a sample document using v2 pipeline.

    Args:
        file_path: Path to the sample file
        strategy: Retrieval strategy ("simple", "router", "fusion", "chunked")
        skip_classification: If True, skip classification step
    """
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample file not found")

    # Create job
    job_id = str(uuid.uuid4())
    v2_processing_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "file_name": path.name,
        "file_path": str(path),
        "created_at": datetime.now().isoformat(),
        "pipeline_version": "v2",
        "retrieval_strategy": strategy,
        "workflow_steps": []
    }

    # Start background processing
    background_tasks.add_task(
        process_document_v2, job_id, path, strategy, skip_classification
    )

    return {
        "job_id": job_id,
        "status": "pending",
        "pipeline_version": "v2",
        "retrieval_strategy": strategy
    }


@app.delete("/api/v2/jobs/{job_id}")
async def delete_job_v2(job_id: str):
    """Delete a v2 processing job from memory, persistent store, and disk."""
    found = False
    file_path = None

    # Get file path before deleting
    if job_id in v2_processing_jobs:
        file_path = v2_processing_jobs[job_id].get("file_path")
        del v2_processing_jobs[job_id]
        found = True

    stored = document_store.get(job_id)
    if stored:
        file_path = file_path or stored.get("file_path")

    if document_store.delete(job_id):
        found = True

    if not found:
        raise HTTPException(status_code=404, detail="Job not found")

    # Clean up uploaded file from disk
    if file_path:
        try:
            fp = Path(file_path)
            if fp.exists():
                fp.unlink()
                logger.info(f"Deleted file: {fp}")
                # Also delete any converted variants (e.g. .pdf → .png)
                for variant in fp.parent.glob(f"{fp.stem}.*"):
                    if variant != fp:
                        variant.unlink()
                        logger.info(f"Deleted variant: {variant}")
        except Exception as e:
            logger.warning(f"Failed to delete file {file_path}: {e}")

    return {"status": "deleted"}


@app.post("/api/v2/jobs/{job_id}/reprocess")
async def reprocess_job_v2(
    job_id: str,
    background_tasks: BackgroundTasks,
    document_type: Optional[str] = None,
    strategy: str = "router"
):
    """
    Reprocess an existing document, optionally with a different document type.

    Looks up the original file from the existing job, creates a new job,
    and re-runs the extraction pipeline. The old job is deleted.

    Args:
        job_id: ID of the existing job to reprocess
        document_type: Optional document type override for reclassification
        strategy: Retrieval strategy

    Returns:
        New job_id for tracking the reprocessed document
    """
    # Look up the existing job
    existing_job = None
    if job_id in v2_processing_jobs:
        existing_job = v2_processing_jobs[job_id]
    else:
        existing_job = document_store.get(job_id)

    if not existing_job:
        raise HTTPException(status_code=404, detail="Job not found")

    file_path = existing_job.get("file_path")
    if not file_path:
        raise HTTPException(status_code=400, detail="No file path associated with this job")

    file_path = Path(file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Original file no longer exists: {file_path.name}")

    # Delete the old job
    if job_id in v2_processing_jobs:
        del v2_processing_jobs[job_id]
    document_store.delete(job_id)

    # Create new job
    new_job_id = str(uuid.uuid4())
    v2_processing_jobs[new_job_id] = {
        "job_id": new_job_id,
        "status": "pending",
        "file_name": existing_job.get("file_name", file_path.name),
        "file_path": str(file_path),
        "created_at": datetime.now().isoformat(),
        "pipeline_version": "v2",
        "retrieval_strategy": strategy,
        "workflow_steps": [],
        "reprocessed_from": job_id,
    }

    # Start background processing with optional type override
    background_tasks.add_task(
        process_document_v2, new_job_id, file_path, strategy, False, document_type
    )

    return {
        "job_id": new_job_id,
        "status": "pending",
        "pipeline_version": "v2",
        "reprocessed_from": job_id,
        "document_type_override": document_type
    }


class BatchDeleteRequest(BaseModel):
    job_ids: List[str]


@app.post("/api/v2/jobs/batch-delete")
async def batch_delete_jobs_v2(request: BatchDeleteRequest):
    """Delete multiple jobs at once."""
    deleted = 0
    errors = []

    for job_id in request.job_ids:
        try:
            file_path = None

            if job_id in v2_processing_jobs:
                file_path = v2_processing_jobs[job_id].get("file_path")
                del v2_processing_jobs[job_id]
                deleted += 1
            else:
                stored = document_store.get(job_id)
                if stored:
                    file_path = stored.get("file_path")

            if document_store.delete(job_id):
                if job_id not in [r for r, _ in errors]:
                    deleted += 1

            # Clean up file
            if file_path:
                try:
                    fp = Path(file_path)
                    if fp.exists():
                        fp.unlink()
                        for variant in fp.parent.glob(f"{fp.stem}.*"):
                            variant.unlink()
                except Exception:
                    pass

        except Exception as e:
            errors.append({"job_id": job_id, "error": str(e)})

    return {"deleted": deleted, "errors": errors}


# ============================================================================
# Chat with Document
# ============================================================================

class ChatRequest(BaseModel):
    message: str = ""
    clear_history: bool = False


MEDICAL_CHAT_SYSTEM_PROMPT = """You are a medical document assistant. You help healthcare professionals and patients understand the contents of a medical document that has been processed and extracted by an AI system.

DOCUMENT CONTEXT:
{context}

INSTRUCTIONS:
- Answer questions ONLY based on the document data provided above.
- If the information is not in the document, say so clearly. Do not make up information.
- When discussing lab results, mention the reference range and flag (H/L) if available.
- When discussing medications, include dosage and frequency if available.
- Use clear, professional medical language but explain terms if asked.
- If a value is flagged as abnormal (H or L), highlight this in your response.
- If there are critical findings, emphasize their importance.
- Do NOT provide medical diagnoses or treatment recommendations. You are an information tool, not a clinician.
- If asked for medical advice, remind the user to consult their healthcare provider.
- Keep responses concise and factual."""


def _build_chat_context(result: Dict[str, Any]) -> str:
    """Build document context string from extracted job result."""
    parts = []

    # Classification
    classification = result.get("classification", {})
    if isinstance(classification, dict) and classification:
        parts.append(f"Document Type: {classification.get('type', 'unknown')}")

    # Patient info
    sections = result.get("sections", {})
    patient = sections.get("patient_info", {})
    if isinstance(patient, dict) and patient:
        patient_str = ", ".join(f"{k}: {v}" for k, v in patient.items() if v)
        parts.append(f"Patient: {patient_str}")

    # Dates
    dates = sections.get("dates", {})
    if isinstance(dates, dict) and dates:
        dates_str = ", ".join(f"{k}: {v}" for k, v in dates.items() if v)
        parts.append(f"Dates: {dates_str}")

    # Provider info
    provider = sections.get("provider_info", {})
    if isinstance(provider, dict) and provider:
        prov_str = ", ".join(f"{k}: {v}" for k, v in provider.items() if v)
        parts.append(f"Provider: {prov_str}")

    # Extracted values (lab results, etc.)
    extracted_values = result.get("extracted_values", [])
    if extracted_values:
        parts.append(f"\nTest Results ({len(extracted_values)} items):")
        for ev in extracted_values:
            if not isinstance(ev, dict):
                continue
            name = ev.get("field_name", "")
            value = ev.get("value", "")
            unit = ev.get("unit", "")
            ref_min = ev.get("reference_min", "")
            ref_max = ev.get("reference_max", "")
            flag = ev.get("abnormal_flag", "")
            line = f"  - {name}: {value}"
            if unit:
                line += f" {unit}"
            if ref_min and ref_max:
                line += f" (ref: {ref_min}-{ref_max})"
            if flag:
                line += f" [{flag}]"
            parts.append(line)

    # Clinical summary
    if result.get("clinical_summary"):
        parts.append(f"\nClinical Summary: {result['clinical_summary']}")

    # Findings
    findings = sections.get("findings", [])
    if isinstance(findings, list) and findings:
        parts.append("\nFindings:")
        for f in findings[:10]:
            parts.append(f"  - {f}")

    # Critical findings
    critical = result.get("critical_findings", [])
    if isinstance(critical, list) and critical:
        parts.append("\nCRITICAL FINDINGS:")
        for c in critical:
            parts.append(f"  - {c}")

    # Medications (from universal_extraction)
    universal = result.get("universal_extraction", {})
    if isinstance(universal, dict):
        meds = universal.get("medications", [])
        if isinstance(meds, list) and meds:
            parts.append(f"\nMedications ({len(meds)}):")
            for med in meds:
                if isinstance(med, dict):
                    name = med.get("name", "")
                    strength = med.get("strength", "")
                    freq = med.get("frequency", "")
                    parts.append(f"  - {name} {strength} {freq}".strip())

    # Raw fields (first 30)
    raw_fields = result.get("raw_fields", {})
    if isinstance(raw_fields, dict) and raw_fields:
        parts.append("\nAdditional Fields:")
        for i, (k, v) in enumerate(raw_fields.items()):
            if i >= 30 or k.startswith('_'):
                continue
            parts.append(f"  - {k}: {v}")

    return "\n".join(parts)


async def _chat_with_llm(
    system_prompt: str,
    messages: List[Dict[str, str]],
    config: Dict[str, Any],
) -> tuple:
    """
    Send chat messages to LLM. Returns (reply_text, model_used).

    Tries Azure GPT-4o first if USE_CLOUD=true, falls back to Ollama.
    """
    if config.get('use_cloud') and config.get('azure_endpoint') and config.get('azure_api_key'):
        try:
            reply = await _chat_azure(system_prompt, messages, config)
            return reply, "azure_gpt4o"
        except Exception as e:
            logger.warning(f"Azure chat failed, falling back to Ollama: {e}")

    reply = await _chat_ollama(system_prompt, messages, config)
    return reply, "ollama_medgemma"


async def _chat_azure(
    system_prompt: str,
    messages: List[Dict[str, str]],
    config: Dict[str, Any],
) -> str:
    """Chat via Azure OpenAI chat completions."""
    import asyncio
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=config['azure_endpoint'],
        api_key=config['azure_api_key'],
        api_version=config.get('azure_api_version', '2024-02-01'),
    )

    api_messages = [{"role": "system", "content": system_prompt}]
    api_messages.extend(messages)

    def call_api():
        response = client.chat.completions.create(
            model=config.get('azure_deployment', 'gpt-4o'),
            messages=api_messages,
            max_tokens=1000,
            temperature=0.3,
        )
        return response.choices[0].message.content

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, call_api)


async def _chat_ollama(
    system_prompt: str,
    messages: List[Dict[str, str]],
    config: Dict[str, Any],
) -> str:
    """Chat via Ollama MedGemma (flatten history into single prompt)."""
    from medical_ingestion.medgemma.client import create_client

    prompt_parts = [system_prompt, "\n--- Conversation ---"]
    for msg in messages:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        prompt_parts.append(f"{role_label}: {msg['content']}")
    prompt_parts.append("Assistant:")

    full_prompt = "\n".join(prompt_parts)

    client = create_client(config)
    result = await client.generate(
        prompt=full_prompt,
        max_tokens=1000,
        temperature=0.3,
    )

    return result.get("text", "I could not generate a response.")


def _lookup_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Look up a job by ID (in-memory first, then SQLite)."""
    if job_id in v2_processing_jobs:
        return v2_processing_jobs[job_id]
    return document_store.get(job_id)


@app.post("/api/v2/jobs/{job_id}/chat")
async def chat_with_document(job_id: str, request: ChatRequest):
    """Chat with an extracted document using LLM."""
    job = _lookup_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Document processing not yet complete (status: {job.get('status')})"
        )

    result = job.get("result", {})
    if not result:
        raise HTTPException(status_code=400, detail="No extraction results available")

    # Handle clear history
    if request.clear_history:
        chat_histories.pop(job_id, None)
        return {"reply": "Chat history cleared.", "model_used": "system", "history": []}

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Build context and system prompt
    context = _build_chat_context(result)
    system_prompt = MEDICAL_CHAT_SYSTEM_PROMPT.format(context=context)

    # Get or create history
    if job_id not in chat_histories:
        chat_histories[job_id] = []

    history = chat_histories[job_id]
    history.append({"role": "user", "content": request.message.strip()})

    # Trim to last 20 messages
    MAX_HISTORY = 20
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
        chat_histories[job_id] = history

    # Call LLM
    try:
        config = get_config()
        reply, model_used = await _chat_with_llm(system_prompt, history, config)
    except Exception as e:
        logger.error(f"Chat LLM call failed: {e}")
        history.pop()  # Remove the user message since we failed
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

    # Append assistant reply
    history.append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "model_used": model_used,
        "history": [{"role": m["role"], "content": m["content"]} for m in history],
    }


@app.get("/api/v2/jobs/{job_id}/chat/suggestions")
async def get_chat_suggestions(job_id: str):
    """Get suggested questions based on document type and content."""
    job = _lookup_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = job.get("result", {})
    doc_type = (result.get("document_type") or job.get("document_type") or "unknown").lower()

    base = [
        "Summarize this document for me",
        "Are there any abnormal findings?",
    ]

    type_suggestions = {
        "lab": [
            "Which lab values are outside the normal range?",
            "Explain what the abnormal results could indicate",
            "What is the reference range for each test?",
            "Are there any critical values that need immediate attention?",
        ],
        "prescription": [
            "List all medications with their dosages",
            "Are there any potential drug interactions to be aware of?",
            "What are the instructions for each medication?",
            "Which medications should be taken with food?",
        ],
        "radiology": [
            "What are the key findings in this report?",
            "Are there any critical findings that need urgent attention?",
            "Summarize the impression section",
        ],
        "pathology": [
            "What is the primary diagnosis?",
            "Summarize the microscopic findings",
            "Are there any malignant findings?",
        ],
    }

    suggestions = base + type_suggestions.get(doc_type, [
        "What information was extracted from this document?",
        "Who is the patient mentioned in this document?",
    ])

    return {"suggestions": suggestions[:6]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
