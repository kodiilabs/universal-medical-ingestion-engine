# ============================================================================
# ui/services/processing_service.py
# ============================================================================
"""
Processing Service

Interfaces between Streamlit UI and the core medical ingestion engine.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ProcessingService:
    """
    Service layer for document processing.

    Wraps the UniversalOrchestrator for use in Streamlit UI.
    """

    _instance = None
    _orchestrator = None
    _classifier = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._orchestrator is None:
            self._init_orchestrator()
        if self._classifier is None:
            self._init_classifier()

    def _init_orchestrator(self):
        """Initialize the orchestrator."""
        from src.medical_ingestion.core.orchestrator import UniversalOrchestrator
        self._orchestrator = UniversalOrchestrator()
        logger.info("Orchestrator initialized successfully")

    def _init_classifier(self):
        """Initialize the document classifier."""
        from src.medical_ingestion.classifiers.document_classifier import DocumentClassifier
        self._classifier = DocumentClassifier({})
        logger.info("Document classifier initialized successfully")

    @property
    def is_available(self) -> bool:
        """Check if the processing engine is available."""
        return self._orchestrator is not None

    def process_uploaded_file(
        self,
        file_content: bytes,
        file_name: str,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process an uploaded file.

        Args:
            file_content: Raw file bytes
            file_name: Original filename
            patient_context: Optional patient information

        Returns:
            Processing result dict
        """
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=Path(file_name).suffix, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = Path(tmp.name)

        try:
            # Run async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(
                self._orchestrator.process_document(
                    pdf_path=tmp_path,
                    patient_context=patient_context
                )
            )

            loop.close()

            return result

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_name': file_name,
                'processed_time': datetime.now().isoformat()
            }
        finally:
            # Cleanup temp file
            try:
                tmp_path.unlink()
            except:
                pass

    def classify_document(self, file_content: bytes, file_name: str) -> Dict[str, Any]:
        """
        Classify document without full processing.

        Args:
            file_content: Raw file bytes
            file_name: Original filename

        Returns:
            Classification result
        """
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=Path(file_name).suffix, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = Path(tmp.name)

        try:
            from src.medical_ingestion.core.context.processing_context import ProcessingContext

            context = ProcessingContext(document_path=tmp_path)

            # Run async classification
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            result = loop.run_until_complete(self._classifier.execute(context))

            loop.close()

            return {
                'document_type': result.get('type', 'unknown'),
                'confidence': result.get('confidence', 0.0),
                'scores': result.get('all_scores', {}),
                'reasoning': result.get('reasoning', '')
            }

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise
        finally:
            try:
                tmp_path.unlink()
            except:
                pass

    def get_reference_ranges(self) -> Dict[str, Dict]:
        """Get reference ranges from knowledge base."""
        from src.medical_ingestion.constants.reference_ranges import REFERENCE_RANGES
        return REFERENCE_RANGES

    def get_loinc_codes(self) -> Dict[str, str]:
        """Get LOINC codes from knowledge base."""
        from src.medical_ingestion.constants.loinc import LOINC_CODES
        return LOINC_CODES

    def get_snomed_codes(self) -> Dict[str, str]:
        """Get SNOMED codes from knowledge base."""
        from src.medical_ingestion.constants.snomed import SNOMED_CODES
        return SNOMED_CODES

    def get_rxnorm_codes(self) -> Dict[str, str]:
        """Get RxNorm codes from knowledge base."""
        from src.medical_ingestion.constants.rxnorm import RXNORM_CODES
        return RXNORM_CODES

    def get_unit_conversions(self) -> Dict[str, Dict]:
        """Get unit conversions from knowledge base."""
        from src.medical_ingestion.constants.unit_conversions import UNIT_CONVERSIONS
        return UNIT_CONVERSIONS

    def build_fhir_bundle(self, extracted_data: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """
        Build a FHIR bundle from extracted data.

        Args:
            extracted_data: Extracted document data
            document_type: Type of document

        Returns:
            FHIR R4 Bundle
        """
        from src.medical_ingestion.fhir_utils.builder import FHIRBuilder

        builder = FHIRBuilder()
        return builder.build_bundle(extracted_data, document_type)

    def validate_fhir_bundle(self, bundle: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a FHIR bundle.

        Args:
            bundle: FHIR bundle to validate

        Returns:
            Tuple of (is_valid, list of errors)
        """
        from src.medical_ingestion.fhir_utils.validator import FHIRValidator

        validator = FHIRValidator()
        return validator.validate_bundle(bundle)
