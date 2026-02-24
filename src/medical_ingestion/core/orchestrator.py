# ============================================================================
# src/medical_ingestion/core/orchestrator.py
# ============================================================================
"""
Universal Document Orchestrator

This is the MAIN entry point for document processing.

Flow:
1. Classify document type (lab, radiology, pathology, etc.)
2. Route to appropriate processor
3. Coordinate multi-agent execution
4. Generate FHIR output
5. Build audit trail
6. Determine human review needs

This orchestrator manages the entire processing pipeline.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import asyncio
import importlib

from .context.processing_context import ProcessingContext, ReviewPriority
from .audit import AuditLogger
from ..config.hardware_config import HardwareSettings
from ..config.thresholds_config import threshold_settings
from ..constants import DocumentType, PROCESSOR_MAPPING


class UniversalOrchestrator:
    """
    Main orchestration engine for document processing.
    
    Responsibilities:
    1. Document classification and routing
    2. Processor coordination
    3. FHIR bundle generation
    4. Audit trail management
    5. Human review escalation
    
    This is the "conductor" that coordinates all agents.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.audit = AuditLogger()
        
        # Lazy-load processors (only when needed)
        self._processors = {}
        self._classifier = None
        
        self.settings = HardwareSettings()
        
        self.logger.info("Universal Orchestrator initialized")
    
    # ========================================================================
    # MAIN PROCESSING PIPELINE
    # ========================================================================
    
    async def process_document(
        self,
        pdf_path: Path,
        patient_context: Optional[Dict[str, Any]] = None,
        document_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for document processing.
        
        This is the method you call to process any medical document.
        
        Args:
            pdf_path: Path to PDF document
            patient_context: Optional patient info (ID, demographics, history)
            document_metadata: Optional metadata (received date, source, etc.)
            
        Returns:
            {
                "success": bool,
                "document_id": str,
                "document_type": str,
                "processor": str,
                "fhir_bundle": dict,
                "confidence": float,
                "requires_review": bool,
                "review_priority": str,
                "warnings": list,
                "processing_time": float,
                "audit_trail": list
            }
            
        Example:
            result = await orchestrator.process_document(
                pdf_path=Path("lab_report.pdf"),
                patient_context={"patient_id": "12345", "age": 45, "sex": "F"}
            )
        """
        start_time = datetime.now()
        
        # Validate input
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Create processing context
        ctx = ProcessingContext(
            document_path=pdf_path,
            patient_demographics=patient_context or {}
        )
        
        if patient_context and 'patient_id' in patient_context:
            ctx.patient_id = patient_context['patient_id']
        
        self.logger.info(f"Processing document: {pdf_path} (ID: {ctx.document_id})")
        
        try:
            # ================================================================
            # STEP 1: Document Classification
            # ================================================================
            self.logger.info("Step 1: Document Classification")
            classification_result = await self._classify_document(ctx)
            
            ctx.document_type = classification_result['type']
            ctx.confidence_scores['classification'] = classification_result['confidence']

            self.logger.info(
                f"Classified as: {ctx.document_type} "
                f"(confidence: {classification_result['confidence']:.2f})"
            )

            # ================================================================
            # EARLY FAIL: Don't extract data if classification confidence is too low
            # ================================================================
            # This prevents extracting potentially wrong/garbage data from
            # documents we can't properly identify
            classification_confidence = classification_result['confidence']
            min_required = threshold_settings.MIN_CLASSIFICATION_TO_PROCEED

            if classification_confidence < min_required:
                self.logger.warning(
                    f"Classification confidence too low ({classification_confidence:.2f} < "
                    f"{min_required}). Aborting extraction to prevent garbage data."
                )
                processing_time = (datetime.now() - start_time).total_seconds()

                # Log to audit
                self.audit.log_processing_error(
                    ctx,
                    f"Classification confidence too low: {classification_confidence:.2f}"
                )

                return {
                    "success": False,
                    "document_id": ctx.document_id,
                    "document_type": ctx.document_type,
                    "error": "Classification confidence too low to proceed with extraction",
                    "classification_confidence": classification_confidence,
                    "min_required": min_required,
                    "reasoning": classification_result.get('reasoning', ''),
                    "requires_review": True,
                    "review_priority": "critical",
                    "processing_time": processing_time,
                    "audit_trail": ctx.agent_executions,
                    "extracted_data": None  # Explicitly no data extracted
                }

            # ================================================================
            # STEP 2: Processor Selection & Routing
            # ================================================================
            self.logger.info("Step 2: Processor Selection")
            processor_name = self._select_processor(ctx.document_type)
            processor = self._get_processor(processor_name)
            
            self.logger.info(f"Selected processor: {processor_name}")
            
            # ================================================================
            # STEP 3: Execute Processor (Multi-Agent Pipeline)
            # ================================================================
            self.logger.info("Step 3: Processor Execution")
            processing_result = await processor.process(ctx)
            
            # ================================================================
            # STEP 4: Generate FHIR Output
            # ================================================================
            self.logger.info("Step 4: FHIR Generation")
            await self._generate_fhir(ctx)
            
            # ================================================================
            # STEP 5: Determine Review Requirements
            # ================================================================
            self.logger.info("Step 5: Review Assessment")
            self._assess_review_needs(ctx)
            
            # ================================================================
            # STEP 6: Finalize Processing
            # ================================================================
            processing_time = (datetime.now() - start_time).total_seconds()
            ctx.processing_duration = processing_time
            
            # Calculate overall confidence
            ctx.calculate_confidence_level()
            
            # Log to audit trail
            self.audit.log_processing_complete(ctx)
            
            # Build extracted_data for UI consumption
            extracted_data = self._build_extracted_data(ctx)

            # Build result
            result = {
                "success": True,
                "document_id": ctx.document_id,
                "document_type": ctx.document_type,
                "display_name": self._generate_display_name(ctx),
                "processor": processor_name,
                "extracted_data": extracted_data,
                "fhir_bundle": ctx.fhir_bundle,
                "confidence": ctx.overall_confidence,
                "confidence_level": ctx.confidence_level.value if ctx.confidence_level else None,
                "requires_review": ctx.requires_review,
                "review_priority": ctx.review_priority.value,
                "warnings": ctx.warnings,
                "quality_flags": ctx.quality_flags,
                "critical_findings": ctx.critical_findings,
                "processing_time": processing_time,
                "audit_trail": ctx.agent_executions,
                "summary": ctx.get_summary()
            }
            
            self.logger.info(
                f"Processing complete: {ctx.document_id} "
                f"({processing_time:.2f}s, confidence: {ctx.overall_confidence:.2f})"
            )
            
            return result
            
        except Exception as e:
            # Handle processing errors gracefully
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.error(f"Processing failed for {pdf_path}: {e}", exc_info=True)
            
            # Log error to audit
            self.audit.log_processing_error(ctx, str(e))
            
            return {
                "success": False,
                "document_id": ctx.document_id,
                "error": str(e),
                "processing_time": processing_time,
                "audit_trail": ctx.agent_executions
            }
    
    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================
    
    async def process_batch(
        self,
        pdf_paths: List[Path],
        patient_contexts: Optional[List[Dict]] = None,
        max_concurrent: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents concurrently.
        
        Args:
            pdf_paths: List of PDF paths
            patient_contexts: Optional list of patient contexts (matched by index)
            max_concurrent: Maximum concurrent processing (default: from config)
            
        Returns:
            List of processing results (same order as input)
            
        Example:
            results = await orchestrator.process_batch([
                Path("lab1.pdf"),
                Path("lab2.pdf"),
                Path("xray.pdf")
            ])
        """
        max_concurrent = max_concurrent or self.settings.MAX_CONCURRENT_DOCS
        
        self.logger.info(
            f"Batch processing {len(pdf_paths)} documents "
            f"(max concurrent: {max_concurrent})"
        )
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(pdf_path, patient_ctx):
            async with semaphore:
                return await self.process_document(pdf_path, patient_ctx)
        
        # Match patient contexts to paths
        if patient_contexts is None:
            patient_contexts = [None] * len(pdf_paths)
        
        # Process all documents concurrently (with limit)
        tasks = [
            process_with_semaphore(pdf_path, patient_ctx)
            for pdf_path, patient_ctx in zip(pdf_paths, patient_contexts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Count successes/failures
        successes = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
        failures = len(results) - successes
        
        self.logger.info(
            f"Batch processing complete: {successes} succeeded, {failures} failed"
        )
        
        return results
    
    # ========================================================================
    # PRIVATE METHODS (Internal orchestration logic)
    # ========================================================================
    
    async def _classify_document(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Classify document type using classifier agent.
        
        Returns:
            {"type": DocumentType, "confidence": float, "reasoning": str}
        """
        if self._classifier is None:
            from ..classifiers.document_classifier import DocumentClassifier
            self._classifier = DocumentClassifier(self.config)
        
        return await self._classifier.execute(context)
    
    def _select_processor(self, document_type: str) -> str:
        """
        Select appropriate processor for document type.
        
        Uses PROCESSOR_MAPPING from constants.
        """
        # Convert string to DocumentType enum if needed
        if isinstance(document_type, str):
            try:
                doc_type_enum = DocumentType(document_type)
            except ValueError:
                doc_type_enum = DocumentType.UNKNOWN
        else:
            doc_type_enum = document_type
        
        return PROCESSOR_MAPPING.get(doc_type_enum, "fallback")
    
    def _get_processor(self, processor_name: str):
        """
        Get or create processor instance (lazy loading).
        """
        if processor_name not in self._processors:
            self._processors[processor_name] = self._create_processor(processor_name)
        
        return self._processors[processor_name]
    
    def _create_processor(self, processor_name: str):
        """
        Instantiate processor by name.
        """
        processor_map = {
            "lab": "processors.lab.processor.LabProcessor",
            "radiology": "processors.radiology.processor.RadiologyProcessor",
            "pathology": "processors.pathology.processor.PathologyProcessor",
            "prescription": "processors.prescription.processor.PrescriptionProcessor",
            "fallback": "processors.fallback.processor.FallbackProcessor"
        }
        
        if processor_name not in processor_map:
            raise ValueError(f"Unknown processor: {processor_name}")

        # Dynamic import using importlib
        module_path, class_name = processor_map[processor_name].rsplit('.', 1)
        full_module_path = f"src.medical_ingestion.{module_path}"
        module = importlib.import_module(full_module_path)
        processor_class = getattr(module, class_name)

        return processor_class(self.config)
    
    async def _generate_fhir(self, context: ProcessingContext):
        """
        Generate FHIR bundle from extracted data.
        
        Delegates to FHIR builder based on document type.
        """
        from ..fhir_utils.builder import FHIRBuilder
        
        builder = FHIRBuilder()
        context.fhir_bundle = await builder.build_bundle(context)
    
    def _assess_review_needs(self, context: ProcessingContext):
        """
        Determine if document requires human review and priority level.
        
        Escalation triggers:
        - Low confidence (< 0.70)
        - Validation conflicts
        - Specimen quality issues
        - Critical findings
        - Quality flags
        """
        # Check if already flagged for review
        if context.requires_review:
            return
        
        # Check confidence threshold
        if context.overall_confidence < threshold_settings.HUMAN_REVIEW_THRESHOLD:
            context.requires_review = True
            context.review_priority = ReviewPriority.HIGH
            context.review_reasons.append(
                f"Low confidence: {context.overall_confidence:.2f}"
            )
        
        # Check for validation conflicts
        if any(v.validation_conflict for v in context.extracted_values):
            context.requires_review = True
            if context.review_priority != ReviewPriority.CRITICAL:
                context.review_priority = ReviewPriority.HIGH
            context.review_reasons.append("Validation conflicts detected")
        
        # Check for specimen rejection
        if context.specimen_rejected:
            context.requires_review = True
            context.review_priority = ReviewPriority.CRITICAL
            context.review_reasons.append(f"Specimen rejected: {context.rejection_reason}")
        
        # Check for critical findings
        if context.critical_findings:
            context.requires_review = True
            context.review_priority = ReviewPriority.CRITICAL
            context.review_reasons.append("Critical findings detected")
        
        # Check for quality flags
        if context.quality_flags:
            context.requires_review = True
            if context.review_priority == ReviewPriority.LOW:
                context.review_priority = ReviewPriority.MEDIUM
            context.review_reasons.append(f"{len(context.quality_flags)} quality flags")

    def _build_extracted_data(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Build extracted_data dict for UI consumption.

        Converts ProcessingContext.extracted_values to the format expected by UI:
        {
            "lab_results": [{"test": str, "value": str, "unit": str, "reference": str, "flag": str}],
            "patient": {...},
            "findings": [...],
            ...
        }
        """
        extracted_data = {}

        # Convert extracted_values to lab_results format
        if context.extracted_values:
            lab_results = []
            for ev in context.extracted_values:
                # Build reference range string
                ref_range = ""
                if ev.reference_min is not None and ev.reference_max is not None:
                    ref_range = f"{ev.reference_min} - {ev.reference_max}"
                elif ev.reference_min is not None:
                    ref_range = f">= {ev.reference_min}"
                elif ev.reference_max is not None:
                    ref_range = f"<= {ev.reference_max}"

                lab_results.append({
                    "test": ev.field_name,
                    "value": str(ev.value) if ev.value is not None else "",
                    "unit": ev.unit or "",
                    "reference": ref_range,
                    "flag": ev.abnormal_flag or "",
                    "confidence": ev.confidence,
                    "extraction_method": ev.extraction_method
                })

            if lab_results:
                extracted_data["lab_results"] = lab_results

        # Include patient demographics
        if context.patient_demographics:
            extracted_data["patient"] = context.patient_demographics

        # Include findings from sections
        if "findings" in context.sections:
            extracted_data["findings"] = context.sections["findings"]

        if "impression" in context.sections:
            extracted_data["impression"] = context.sections["impression"]

        # Include clinical summary
        if context.clinical_summary:
            extracted_data["clinical_summary"] = context.clinical_summary

        # Include any other sections
        for key, value in context.sections.items():
            if key not in extracted_data and key not in ["findings", "impression"]:
                extracted_data[key] = value

        return extracted_data

    def _generate_display_name(self, context: ProcessingContext) -> str:
        """
        Generate a human-friendly display name from document metadata.

        Format: "{DocType} - {PatientName} - {Date}"
        Fallbacks when fields are missing:
            "Lab Report - John Doe - 2024-01-15"
            "Prescription - Jane Smith"
            "Lab Report - 2024-03-22"
            "Clinical Note"
        """
        import re

        # 1. Document type label
        type_labels = {
            'lab_report': 'Lab Report',
            'prescription': 'Prescription',
            'radiology_report': 'Radiology Report',
            'pathology_report': 'Pathology Report',
            'clinical_note': 'Clinical Note',
            'discharge_summary': 'Discharge Summary',
            'insurance_claim': 'Insurance Claim',
            'dental_record': 'Dental Record',
            'surgical_report': 'Surgical Report',
            'immunization_record': 'Immunization Record',
        }
        doc_type = type_labels.get(
            context.document_type or '',
            (context.document_type or 'Document').replace('_', ' ').title()
        )

        parts = [doc_type]

        # 2. Patient name (from demographics)
        patient = context.patient_demographics or {}
        patient_name = patient.get('name') or patient.get('patient_name') or ''
        if patient_name:
            # Clean up: title case, strip extra whitespace
            patient_name = ' '.join(patient_name.strip().split())
            # Don't include if it looks like a placeholder/ID
            if not re.match(r'^[A-F0-9-]{8,}$', patient_name):
                parts.append(patient_name.title() if patient_name.isupper() or patient_name.islower() else patient_name)

        # 3. Date (from metadata or context timestamp)
        doc_date = None
        if context.document_metadata:
            doc_date = getattr(context.document_metadata, 'document_date', None)
            if not doc_date:
                doc_date = getattr(context.document_metadata, 'collection_date', None)
        if not doc_date and patient.get('date'):
            doc_date = patient['date']
        if not doc_date:
            # Use processing timestamp as last resort
            doc_date = context.timestamp.strftime('%Y-%m-%d')

        if doc_date:
            if isinstance(doc_date, str):
                parts.append(doc_date[:10])
            elif hasattr(doc_date, 'strftime'):
                parts.append(doc_date.strftime('%Y-%m-%d'))

        return ' - '.join(parts)

