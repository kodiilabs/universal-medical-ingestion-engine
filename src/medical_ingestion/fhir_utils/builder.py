# ============================================================================
# FILE: src/medical_ingestion/fhir_utils/builder.py
# ============================================================================
"""
FHIR R4 Bundle Builder

Converts extracted medical data into FHIR R4 compliant resources.

Supports:
- Lab results → Patient + DiagnosticReport + Observation bundle (LOINC codes)
- Radiology reports → DiagnosticReport + ImagingStudy
- Pathology reports → DiagnosticReport (SNOMED codes)
- Prescriptions → MedicationRequest (RxNorm codes)

Key features:
- FHIR R4 + US Core compliance (category on Observations)
- Uses enricher LOINC/RxNorm codes (not static JSON fallback)
- Proper timestamps from actual report/collection dates
- Patient resource from extracted demographics
- DiagnosticReport wrapping Observations
- Provenance tracking and confidence extensions
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from uuid import uuid4
import logging
import re

from fhir.resources.bundle import Bundle, BundleEntry
from fhir.resources.observation import Observation, ObservationReferenceRange
from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.medicationrequest import MedicationRequest
from fhir.resources.patient import Patient
from fhir.resources.practitioner import Practitioner
from fhir.resources.humanname import HumanName
from fhir.resources.contactpoint import ContactPoint
from fhir.resources.address import Address
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.quantity import Quantity
from fhir.resources.reference import Reference
from fhir.resources.identifier import Identifier
from fhir.resources.extension import Extension
from fhir.resources.meta import Meta
from fhir.resources.dosage import Dosage

from ..constants.loinc import LOINC_CODES
from ..core.context.extracted_value import ExtractedValue
from ..core.context.processing_context import ProcessingContext
from ..config.fhir_config import fhir_settings

logger = logging.getLogger(__name__)


class FHIRBuilder:
    """
    FHIR R4 resource builder.

    Converts processing context or v2 result dicts into FHIR-compliant bundles.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # ========================================================================
    # V2 PIPELINE ENTRY POINT
    # ========================================================================

    def build_from_v2_result(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build FHIR bundle from v2 pipeline frontend result dict.

        This is the primary entry point for the extraction-first pipeline.
        The result_dict has: extracted_values, sections, classification,
        clinical_summary, etc.

        Returns:
            FHIR R4 Bundle as dictionary
        """
        doc_type = result_dict.get("document_type", "unknown")
        self.logger.info(f"Building FHIR bundle from v2 result (type={doc_type})")

        sections = result_dict.get("sections", {})
        extracted_values = result_dict.get("extracted_values", [])

        # Resolve timestamps from actual report data
        effective_date = self._resolve_effective_date(sections)
        issued_date = self._resolve_issued_date(sections)

        entries = []
        observation_refs = []

        # --- Patient resource ---
        patient_uuid = str(uuid4())
        patient_info = sections.get("patient_info", {})
        if isinstance(patient_info, dict) and patient_info:
            patient_resource = self._create_patient_resource(
                patient_info, patient_uuid
            )
            entries.append(BundleEntry(
                fullUrl=f"urn:uuid:{patient_uuid}",
                resource=patient_resource
            ))
        patient_ref = Reference(reference=f"urn:uuid:{patient_uuid}")

        # --- Practitioner resources from extracted providers ---
        providers = sections.get("providers", [])
        if isinstance(providers, list):
            for prov in providers:
                if isinstance(prov, dict):
                    pract = self._create_practitioner_resource(prov)
                    if pract:
                        entries.append(BundleEntry(
                            fullUrl=f"urn:uuid:{pract.id}",
                            resource=pract
                        ))

        # --- Route by document type ---
        if doc_type in ("lab", "laboratory"):
            # Create Observations from extracted_values
            for ev in extracted_values:
                if ev.get("category") == "medication":
                    continue  # skip medication entries
                obs_uuid = str(uuid4())
                obs = self._create_observation_from_dict(
                    ev, patient_ref, effective_date, issued_date
                )
                if obs:
                    obs.id = obs_uuid
                    entries.append(BundleEntry(
                        fullUrl=f"urn:uuid:{obs_uuid}",
                        resource=obs
                    ))
                    observation_refs.append(
                        Reference(reference=f"urn:uuid:{obs_uuid}")
                    )

            # Create DiagnosticReport wrapping the Observations
            if observation_refs:
                report = self._create_diagnostic_report(
                    observation_refs, patient_ref,
                    effective_date, issued_date,
                    result_dict.get("clinical_summary", "")
                )
                entries.append(BundleEntry(
                    fullUrl=f"urn:uuid:{uuid4()}",
                    resource=report
                ))

        elif doc_type == "prescription":
            medications = sections.get("medications", [])
            if isinstance(medications, list):
                for med in medications:
                    if isinstance(med, dict):
                        med_req = self._create_medication_request_from_dict(
                            med, patient_ref, issued_date
                        )
                        if med_req:
                            entries.append(BundleEntry(
                                fullUrl=f"urn:uuid:{uuid4()}",
                                resource=med_req
                            ))

        elif doc_type == "radiology":
            report = DiagnosticReport(
                id=str(uuid4()),
                status="final",
                code=CodeableConcept(
                    coding=[Coding(
                        system="http://loinc.org",
                        code="18748-4",
                        display="Diagnostic Imaging Report"
                    )],
                    text="Radiology Report"
                ),
                subject=patient_ref,
            )
            if effective_date:
                report.effectiveDateTime = effective_date
            if issued_date:
                report.issued = issued_date
            summary = result_dict.get("clinical_summary", "")
            if summary:
                report.conclusion = summary
            entries.append(BundleEntry(
                fullUrl=f"urn:uuid:{uuid4()}", resource=report
            ))

        elif doc_type == "pathology":
            report = DiagnosticReport(
                id=str(uuid4()),
                status="final",
                code=CodeableConcept(
                    coding=[Coding(
                        system="http://loinc.org",
                        code="11526-1",
                        display="Pathology Report"
                    )],
                    text="Pathology Report"
                ),
                subject=patient_ref,
            )
            if effective_date:
                report.effectiveDateTime = effective_date
            if issued_date:
                report.issued = issued_date
            summary = result_dict.get("clinical_summary", "")
            if summary:
                report.conclusion = summary
            entries.append(BundleEntry(
                fullUrl=f"urn:uuid:{uuid4()}", resource=report
            ))

        # Build the bundle
        bundle = Bundle(
            id=str(uuid4()),
            type="collection",
            timestamp=issued_date or datetime.now(timezone.utc).isoformat(),
            entry=entries if entries else []
        )

        bundle.meta = Meta(
            lastUpdated=datetime.now(timezone.utc).isoformat(),
            profile=["http://hl7.org/fhir/StructureDefinition/Bundle"]
        )

        self.logger.info(
            f"Built v2 FHIR bundle: {len(entries)} entries "
            f"({len(observation_refs)} observations)"
        )

        return bundle.model_dump()

    # ========================================================================
    # RESOURCE CREATORS (shared by v1 and v2)
    # ========================================================================

    def _create_patient_resource(
        self, patient_info: Dict[str, Any], patient_id: str
    ) -> Patient:
        """Create FHIR Patient from extracted patient_info dict."""
        patient_data: Dict[str, Any] = {"id": patient_id}

        # Name
        name = (
            patient_info.get("name")
            or patient_info.get("patient_name")
            or ""
        ).strip()
        if name:
            parts = name.split()
            patient_data["name"] = [HumanName(
                family=parts[-1] if len(parts) > 1 else parts[0],
                given=parts[:-1] if len(parts) > 1 else [],
                text=name
            )]

        # DOB
        dob = patient_info.get("dob") or patient_info.get("date_of_birth") or ""
        if dob:
            parsed = self._parse_date_to_fhir(str(dob))
            if parsed:
                patient_data["birthDate"] = parsed

        # Gender
        sex = (
            patient_info.get("sex")
            or patient_info.get("gender")
            or ""
        ).strip().lower()
        gender_map = {
            "male": "male", "m": "male",
            "female": "female", "f": "female",
            "other": "other", "unknown": "unknown",
        }
        if sex in gender_map:
            patient_data["gender"] = gender_map[sex]

        # Phone
        phone = patient_info.get("phone") or patient_info.get("phone_number") or ""
        if phone:
            patient_data["telecom"] = [ContactPoint(
                system="phone", value=str(phone)
            )]

        # Address
        addr = patient_info.get("address") or ""
        if addr:
            patient_data["address"] = [Address(text=str(addr))]

        # Identifiers (MRN, account number)
        identifiers = []
        mrn = patient_info.get("patient_id") or patient_info.get("mrn") or ""
        if mrn:
            identifiers.append(Identifier(
                type=CodeableConcept(text="MRN"),
                value=str(mrn)
            ))
        acct = patient_info.get("account_number") or ""
        if acct:
            identifiers.append(Identifier(
                type=CodeableConcept(text="Account Number"),
                value=str(acct)
            ))
        if identifiers:
            patient_data["identifier"] = identifiers

        return Patient(**patient_data)

    def _create_practitioner_resource(
        self, provider: Dict[str, Any]
    ) -> Optional[Practitioner]:
        """Create FHIR Practitioner from extracted provider dict."""
        name = (provider.get("name") or "").strip()
        if not name:
            return None

        pract_id = str(uuid4())
        parts = name.split()
        human_name = HumanName(
            family=parts[-1] if len(parts) > 1 else parts[0],
            given=parts[:-1] if len(parts) > 1 else [],
            text=name
        )

        pract_data: Dict[str, Any] = {
            "id": pract_id,
            "name": [human_name],
        }

        # Add NPI identifier if available
        npi = provider.get("npi")
        if npi:
            pract_data["identifier"] = [Identifier(
                system="http://hl7.org/fhir/sid/us-npi",
                value=str(npi)
            )]

        try:
            return Practitioner(**pract_data)
        except Exception as e:
            self.logger.warning(f"Failed to create Practitioner for {name}: {e}")
            return None

    def _create_observation_from_dict(
        self,
        ev: Dict[str, Any],
        subject_ref: Reference,
        effective_date: Optional[str],
        issued_date: Optional[str],
    ) -> Optional[Observation]:
        """
        Create Observation from a v2 extracted_value dict.

        Uses enricher's loinc_code first, falls back to static LOINC_CODES dict.
        Never creates a Coding with code=None.
        Adds US Core required `category: laboratory`.
        """
        field_name = ev.get("field_name", "")
        if not field_name:
            return None

        # --- Resolve LOINC code ---
        # Priority: enricher loinc_code > static LOINC_CODES dict
        loinc_code = ev.get("loinc_code")
        loinc_display = ev.get("loinc_name") or field_name.replace("_", " ").title()

        if not loinc_code:
            # Fallback to static dict
            lookup_key = field_name.lower().replace(" ", "_")
            loinc_code = LOINC_CODES.get(lookup_key)
            if not loinc_code:
                # Try without underscores
                loinc_code = LOINC_CODES.get(field_name.lower())

        # Build code — only add LOINC system if we have a code
        codings = []
        if loinc_code:
            codings.append(Coding(
                system="http://loinc.org",
                code=str(loinc_code),
                display=loinc_display
            ))

        display_text = field_name.replace("_", " ").title()
        code = CodeableConcept(
            coding=codings if codings else None,
            text=display_text
        )

        # --- Value ---
        value_quantity = None
        value_string = None
        raw_value = ev.get("value")

        if raw_value is not None:
            try:
                numeric_value = float(raw_value)
                unit = ev.get("unit", "")
                value_quantity = Quantity(
                    value=numeric_value,
                    unit=unit or None,
                    system="http://unitsofmeasure.org" if unit else None,
                    code=unit or None
                )
            except (ValueError, TypeError):
                value_string = str(raw_value)

        # --- Reference range ---
        reference_range = None
        ref_min = ev.get("reference_min")
        ref_max = ev.get("reference_max")
        if ref_min is not None and ref_max is not None:
            try:
                unit = ev.get("unit", "")
                reference_range = [ObservationReferenceRange(
                    low=Quantity(
                        value=float(ref_min), unit=unit or None,
                        system="http://unitsofmeasure.org" if unit else None
                    ),
                    high=Quantity(
                        value=float(ref_max), unit=unit or None,
                        system="http://unitsofmeasure.org" if unit else None
                    ),
                    text=f"{ref_min}-{ref_max} {unit}".strip()
                )]
            except (ValueError, TypeError):
                pass

        # --- Interpretation (H/L/N flags) ---
        interpretation = None
        flag = ev.get("abnormal_flag")
        if flag:
            interp_code = self._map_abnormal_flag(str(flag))
            interpretation = [CodeableConcept(
                coding=[Coding(
                    system="http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    code=interp_code,
                    display=str(flag)
                )]
            )]

        # --- US Core category: laboratory ---
        category = [CodeableConcept(
            coding=[Coding(
                system="http://terminology.hl7.org/CodeSystem/observation-category",
                code="laboratory",
                display="Laboratory"
            )]
        )]

        # --- Extensions ---
        extensions = []
        confidence = ev.get("confidence")
        if confidence is not None and fhir_settings.FHIR_INCLUDE_CONFIDENCE:
            extensions.append(Extension(
                url="http://medical-ingestion-engine.local/confidence",
                valueDecimal=float(confidence)
            ))
        validation_status = ev.get("validation_status")
        if validation_status:
            extensions.append(Extension(
                url="http://medical-ingestion-engine.local/validation-status",
                valueString=validation_status
            ))

        # --- Build Observation ---
        obs_data: Dict[str, Any] = {
            "status": "final",
            "code": code,
            "category": category,
            "subject": subject_ref,
        }

        if value_quantity:
            obs_data["valueQuantity"] = value_quantity
        elif value_string:
            obs_data["valueString"] = value_string

        if reference_range:
            obs_data["referenceRange"] = reference_range
        if interpretation:
            obs_data["interpretation"] = interpretation
        if extensions:
            obs_data["extension"] = extensions
        if effective_date:
            obs_data["effectiveDateTime"] = effective_date
        if issued_date:
            obs_data["issued"] = issued_date

        try:
            return Observation(**obs_data)
        except Exception as e:
            self.logger.warning(f"Failed to create Observation for {field_name}: {e}")
            return None

    def _create_diagnostic_report(
        self,
        observation_refs: List[Reference],
        subject_ref: Reference,
        effective_date: Optional[str],
        issued_date: Optional[str],
        clinical_summary: str = "",
    ) -> DiagnosticReport:
        """Create DiagnosticReport wrapping lab Observations."""
        report_data: Dict[str, Any] = {
            "id": str(uuid4()),
            "status": "final",
            "code": CodeableConcept(
                coding=[Coding(
                    system="http://loinc.org",
                    code="11502-2",
                    display="Laboratory report"
                )],
                text="Laboratory Report"
            ),
            "category": [CodeableConcept(
                coding=[Coding(
                    system="http://terminology.hl7.org/CodeSystem/v2-0074",
                    code="LAB",
                    display="Laboratory"
                )]
            )],
            "subject": subject_ref,
            "result": observation_refs,
        }

        if effective_date:
            report_data["effectiveDateTime"] = effective_date
        if issued_date:
            report_data["issued"] = issued_date
        if clinical_summary:
            report_data["conclusion"] = clinical_summary

        return DiagnosticReport(**report_data)

    def _create_medication_request_from_dict(
        self,
        med: Dict[str, Any],
        subject_ref: Reference,
        authored_date: Optional[str],
    ) -> Optional[MedicationRequest]:
        """Create MedicationRequest from a v2 medication dict."""
        med_name = (
            med.get("medication_name")
            or med.get("name")
            or ""
        ).strip()
        if not med_name:
            return None  # Don't create placeholder for unknown meds

        strength = med.get("strength", "")
        rxcui = med.get("rxcui") or med.get("rxnorm_code")
        display_text = f"{med_name} {strength}".strip() if strength else med_name

        # Medication coding
        codings = []
        if rxcui:
            codings.append(Coding(
                system="http://www.nlm.nih.gov/research/umls/rxnorm",
                code=str(rxcui),
                display=med.get("rxnorm_name", med_name)
            ))

        medication_concept = CodeableConcept(
            coding=codings if codings else None,
            text=display_text
        )

        request_data: Dict[str, Any] = {
            "id": str(uuid4()),
            "status": "active",
            "intent": "order",
            "subject": subject_ref,
        }

        if authored_date:
            request_data["authoredOn"] = authored_date
        else:
            request_data["authoredOn"] = datetime.now(timezone.utc).isoformat()

        # Handle medication[x] (R4 vs R5 compat)
        try:
            from fhir.resources.codeablereference import CodeableReference
            request_data["medication"] = CodeableReference(concept=medication_concept)
            med_request = MedicationRequest(**request_data)
        except (ImportError, Exception):
            try:
                request_data["medication"] = medication_concept
                med_request = MedicationRequest(**request_data)
            except Exception:
                if "medication" in request_data:
                    del request_data["medication"]
                request_data["medicationCodeableConcept"] = medication_concept
                try:
                    med_request = MedicationRequest(**request_data)
                except Exception as e:
                    self.logger.error(f"Failed to create MedicationRequest: {e}")
                    return None

        # Dosage
        instructions = med.get("instructions", "")
        frequency = med.get("frequency", "")
        route = med.get("route", "")
        dosage_parts = []
        if instructions:
            dosage_parts.append(str(instructions))
        if frequency and str(frequency) not in str(instructions):
            dosage_parts.append(str(frequency))
        if route and str(route) not in str(instructions):
            dosage_parts.append(f"Route: {route}")
        if dosage_parts:
            med_request.dosageInstruction = [Dosage(text="; ".join(dosage_parts))]

        # Extensions
        extensions = []
        validation_status = med.get("validation_status")
        if validation_status:
            extensions.append(Extension(
                url="http://medical-ingestion-engine.local/validation-status",
                valueString=validation_status
            ))
        if extensions:
            med_request.extension = extensions

        return med_request

    # ========================================================================
    # V1 PIPELINE ENTRY POINT (ProcessingContext-based)
    # ========================================================================

    async def build_bundle(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Build FHIR bundle from v1 ProcessingContext.

        Routes to appropriate builder based on document type.
        """
        self.logger.info(f"Building FHIR bundle for {context.document_type}")

        if context.document_type == "lab":
            bundle = self._build_lab_bundle(context)
        elif context.document_type == "radiology":
            bundle = self._build_radiology_bundle(context)
        elif context.document_type == "pathology":
            bundle = self._build_pathology_bundle(context)
        elif context.document_type == "prescription":
            bundle = self._build_prescription_bundle(context)
        else:
            bundle = self._build_generic_bundle(context)

        bundle.meta = Meta(
            lastUpdated=datetime.now(timezone.utc).isoformat(),
            profile=["http://hl7.org/fhir/StructureDefinition/Bundle"]
        )

        if fhir_settings.FHIR_VALIDATE:
            validation_errors = self._validate_bundle(bundle)
            if validation_errors:
                context.fhir_validation_errors = validation_errors
                self.logger.warning(f"FHIR validation errors: {len(validation_errors)}")

        return bundle.model_dump()

    # ========================================================================
    # V1 LAB BUNDLE
    # ========================================================================

    def _build_lab_bundle(self, context: ProcessingContext) -> Bundle:
        """Build Observation bundle for v1 lab results (ProcessingContext)."""
        entries = []

        # Resolve dates from context metadata
        effective_date = None
        issued_date = None
        if context.document_metadata:
            meta = context.document_metadata
            if hasattr(meta, 'report_info') and meta.report_info:
                if meta.report_info.collection_date:
                    effective_date = str(meta.report_info.collection_date)
                if meta.report_info.report_date:
                    issued_date = str(meta.report_info.report_date)

        patient_ref = None
        if context.patient_id:
            patient_ref = Reference(reference=f"Patient/{context.patient_id}")

        observation_refs = []

        for extracted_value in context.extracted_values:
            obs = self._create_lab_observation(
                extracted_value, context, effective_date, issued_date
            )
            obs_uuid = str(uuid4())
            obs.id = obs_uuid

            entry = BundleEntry(
                fullUrl=f"urn:uuid:{obs_uuid}",
                resource=obs
            )
            entries.append(entry)
            observation_refs.append(Reference(reference=f"urn:uuid:{obs_uuid}"))

        # Add DiagnosticReport
        if observation_refs:
            report = DiagnosticReport(
                id=str(uuid4()),
                status="final",
                code=CodeableConcept(
                    coding=[Coding(
                        system="http://loinc.org",
                        code="11502-2",
                        display="Laboratory report"
                    )],
                    text="Laboratory Report"
                ),
                category=[CodeableConcept(
                    coding=[Coding(
                        system="http://terminology.hl7.org/CodeSystem/v2-0074",
                        code="LAB", display="Laboratory"
                    )]
                )],
                result=observation_refs,
            )
            if patient_ref:
                report.subject = patient_ref
            if effective_date:
                report.effectiveDateTime = effective_date
            if issued_date:
                report.issued = issued_date
            if context.clinical_summary:
                report.conclusion = context.clinical_summary

            entries.append(BundleEntry(
                fullUrl=f"urn:uuid:{uuid4()}", resource=report
            ))

        bundle = Bundle(
            id=str(uuid4()),
            type="collection",
            identifier=Identifier(
                system="http://medical-ingestion-engine.local",
                value=context.document_id
            ),
            timestamp=issued_date or datetime.now(timezone.utc).isoformat(),
            entry=entries
        )

        self.logger.info(f"Created lab bundle with {len(entries)} entries")
        return bundle

    def _create_lab_observation(
        self,
        extracted: ExtractedValue,
        context: ProcessingContext,
        effective_date: Optional[str] = None,
        issued_date: Optional[str] = None,
    ) -> Observation:
        """Create Observation from v1 ExtractedValue dataclass."""

        # Resolve LOINC — use enricher code if available, else static dict
        loinc_code = getattr(extracted, 'loinc_code', None)
        if not loinc_code:
            loinc_code = LOINC_CODES.get(extracted.field_name)

        codings = []
        if loinc_code:
            codings.append(Coding(
                system="http://loinc.org",
                code=str(loinc_code),
                display=extracted.field_name.replace('_', ' ').title()
            ))

        code = CodeableConcept(
            coding=codings if codings else None,
            text=extracted.field_name.replace('_', ' ').title()
        )

        # US Core category
        category = [CodeableConcept(
            coding=[Coding(
                system="http://terminology.hl7.org/CodeSystem/observation-category",
                code="laboratory", display="Laboratory"
            )]
        )]

        # Value
        value_quantity = None
        value_string = None
        if extracted.value is not None:
            try:
                numeric_value = float(extracted.value)
                value_quantity = Quantity(
                    value=numeric_value,
                    unit=extracted.unit,
                    system="http://unitsofmeasure.org",
                    code=extracted.unit
                )
            except (ValueError, TypeError):
                value_string = str(extracted.value)

        # Reference range
        reference_range = None
        if extracted.reference_min is not None and extracted.reference_max is not None:
            reference_range = [ObservationReferenceRange(
                low=Quantity(
                    value=extracted.reference_min, unit=extracted.unit,
                    system="http://unitsofmeasure.org"
                ),
                high=Quantity(
                    value=extracted.reference_max, unit=extracted.unit,
                    system="http://unitsofmeasure.org"
                ),
                text=f"{extracted.reference_min}-{extracted.reference_max} {extracted.unit}"
            )]

        # Interpretation
        interpretation = None
        if extracted.abnormal_flag:
            interp_code = self._map_abnormal_flag(extracted.abnormal_flag)
            interpretation = [CodeableConcept(
                coding=[Coding(
                    system="http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                    code=interp_code, display=extracted.abnormal_flag
                )]
            )]

        # Extensions
        extensions = []
        if fhir_settings.FHIR_INCLUDE_CONFIDENCE:
            extensions.append(Extension(
                url="http://medical-ingestion-engine.local/confidence",
                valueDecimal=extracted.confidence
            ))
        extensions.append(Extension(
            url="http://medical-ingestion-engine.local/extraction-method",
            valueString=extracted.extraction_method
        ))
        if fhir_settings.FHIR_INCLUDE_PROVENANCE and extracted.source_location:
            extensions.append(Extension(
                url="http://medical-ingestion-engine.local/source-location",
                valueString=extracted.source_location
            ))
        if extracted.validation_conflict:
            extensions.append(Extension(
                url="http://medical-ingestion-engine.local/validation-conflict",
                valueBoolean=True
            ))

        # Build observation
        obs_data: Dict[str, Any] = {
            "status": "final",
            "code": code,
            "category": category,
        }

        if value_quantity:
            obs_data["valueQuantity"] = value_quantity
        elif value_string:
            obs_data["valueString"] = value_string

        if reference_range:
            obs_data["referenceRange"] = reference_range
        if interpretation:
            obs_data["interpretation"] = interpretation
        if extensions:
            obs_data["extension"] = extensions
        if effective_date:
            obs_data["effectiveDateTime"] = effective_date
        if issued_date:
            obs_data["issued"] = issued_date

        observation = Observation(**obs_data)

        if context.patient_id:
            observation.subject = Reference(
                reference=f"Patient/{context.patient_id}"
            )

        if extracted.warnings:
            observation.note = [{"text": w} for w in extracted.warnings]

        return observation

    # ========================================================================
    # V1 RADIOLOGY / PATHOLOGY / PRESCRIPTION / GENERIC
    # ========================================================================

    def _build_radiology_bundle(self, context: ProcessingContext) -> Bundle:
        """Build DiagnosticReport for radiology."""
        report = DiagnosticReport(
            id=str(uuid4()),
            status="final",
            code=CodeableConcept(
                coding=[Coding(
                    system="http://loinc.org", code="18748-4",
                    display="Diagnostic Imaging Report"
                )],
                text="Radiology Report"
            ),
        )
        if context.patient_id:
            report.subject = Reference(reference=f"Patient/{context.patient_id}")
        if context.clinical_summary:
            report.conclusion = context.clinical_summary

        entries = [BundleEntry(fullUrl=f"urn:uuid:{uuid4()}", resource=report)]
        return Bundle(
            id=str(uuid4()), type="collection",
            identifier=Identifier(
                system="http://medical-ingestion-engine.local",
                value=context.document_id
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry=entries
        )

    def _build_pathology_bundle(self, context: ProcessingContext) -> Bundle:
        """Build DiagnosticReport for pathology."""
        report = DiagnosticReport(
            id=str(uuid4()),
            status="final",
            code=CodeableConcept(
                coding=[Coding(
                    system="http://loinc.org", code="11526-1",
                    display="Pathology Report"
                )],
                text="Pathology Report"
            ),
        )
        if context.patient_id:
            report.subject = Reference(reference=f"Patient/{context.patient_id}")
        if context.clinical_summary:
            report.conclusion = context.clinical_summary

        entries = [BundleEntry(fullUrl=f"urn:uuid:{uuid4()}", resource=report)]
        return Bundle(
            id=str(uuid4()), type="collection",
            identifier=Identifier(
                system="http://medical-ingestion-engine.local",
                value=context.document_id
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry=entries
        )

    def _build_prescription_bundle(self, context: ProcessingContext) -> Bundle:
        """Build MedicationRequest bundle for prescriptions."""
        entries = []
        medications = context.sections.get('medications', [])

        for med in medications:
            med_request = self._create_medication_request(med, context)
            if med_request:
                entries.append(BundleEntry(
                    fullUrl=f"urn:uuid:{uuid4()}", resource=med_request
                ))

        # No placeholder for empty medications — just return empty bundle

        return Bundle(
            id=str(uuid4()), type="collection",
            identifier=Identifier(
                system="http://medical-ingestion-engine.local",
                value=context.document_id
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry=entries
        )

    def _create_medication_request(
        self, medication: dict, context: ProcessingContext
    ) -> Optional[MedicationRequest]:
        """Create MedicationRequest from v1 medication dict."""
        medication_name = medication.get('medication_name', '').strip()
        if not medication_name:
            return None

        strength = medication.get('strength', '')
        rxcui = medication.get('rxcui')
        display_text = f"{medication_name} {strength}".strip() if strength else medication_name

        codings = []
        if rxcui:
            codings.append(Coding(
                system="http://www.nlm.nih.gov/research/umls/rxnorm",
                code=rxcui,
                display=medication.get('rxnorm_name', medication_name)
            ))

        medication_concept = CodeableConcept(
            coding=codings if codings else None, text=display_text
        )

        subject_id = getattr(context, 'patient_id', None) or "anonymous"
        subject_ref = Reference(reference=f"Patient/{subject_id}")

        request_data = {
            "id": str(uuid4()),
            "status": "active",
            "intent": "order",
            "subject": subject_ref,
            "authoredOn": datetime.now(timezone.utc).isoformat(),
        }

        try:
            from fhir.resources.codeablereference import CodeableReference
            request_data["medication"] = CodeableReference(concept=medication_concept)
            med_request = MedicationRequest(**request_data)
        except (ImportError, Exception):
            try:
                request_data["medication"] = medication_concept
                med_request = MedicationRequest(**request_data)
            except Exception:
                if "medication" in request_data:
                    del request_data["medication"]
                request_data["medicationCodeableConcept"] = medication_concept
                try:
                    med_request = MedicationRequest(**request_data)
                except Exception as e:
                    self.logger.error(f"Failed to create MedicationRequest: {e}")
                    return None

        # Dosage
        instructions = medication.get('instructions', '')
        frequency = medication.get('frequency', '')
        route = medication.get('route', '')
        parts = []
        if instructions:
            parts.append(instructions)
        if frequency and frequency not in instructions:
            parts.append(frequency)
        if route and route not in instructions:
            parts.append(f"Route: {route}")
        if parts:
            med_request.dosageInstruction = [Dosage(text="; ".join(parts))]

        return med_request

    def _build_generic_bundle(self, context: ProcessingContext) -> Bundle:
        """Build empty bundle for unknown document types."""
        return Bundle(
            id=str(uuid4()), type="collection",
            identifier=Identifier(
                system="http://medical-ingestion-engine.local",
                value=context.document_id
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
            entry=[]
        )

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _map_abnormal_flag(self, flag: str) -> str:
        """Map abnormal flags to FHIR interpretation codes."""
        mapping = {
            "H": "H", "HIGH": "H", "HI": "H",
            "L": "L", "LOW": "L", "LO": "L",
            "N": "N", "NORMAL": "N",
            "CRITICAL": "HH", "CRITICAL HIGH": "HH", "HH": "HH",
            "CRITICAL LOW": "LL", "LL": "LL",
            "A": "A", "ABNORMAL": "A",
        }
        return mapping.get(flag.upper().strip(), "A")

    def _resolve_effective_date(self, sections: Dict[str, Any]) -> Optional[str]:
        """
        Resolve effectiveDateTime from sections.

        Handles two formats:
        - Dict: {"collection_date": "2024-01-15", "report_date": "..."}
        - List: [{"date_type": "Study Date", "date_value": "1/4/2016"}, ...]

        Priority: collection_date/Study Date > report_date > test_date
        """
        dates = sections.get("dates", {})

        # Handle list format from content-agnostic extractor
        if isinstance(dates, list):
            date_map = self._dates_list_to_dict(dates)
        elif isinstance(dates, dict):
            date_map = dates
        else:
            return None

        for key in ("collection_date", "study_date", "report_date", "test_date", "date"):
            val = date_map.get(key)
            if val:
                parsed = self._parse_date_to_fhir(str(val))
                if parsed:
                    return parsed
        return None

    def _resolve_issued_date(self, sections: Dict[str, Any]) -> Optional[str]:
        """
        Resolve issued date (when report was produced).

        Handles both dict and list date formats.
        Priority: report_date/Report Signed > resulted_date > collection_date
        """
        dates = sections.get("dates", {})

        if isinstance(dates, list):
            date_map = self._dates_list_to_dict(dates)
        elif isinstance(dates, dict):
            date_map = dates
        else:
            return None

        for key in ("report_date", "report_signed", "resulted_date", "collection_date", "study_date", "date"):
            val = date_map.get(key)
            if val:
                parsed = self._parse_date_to_fhir(str(val))
                if parsed:
                    return parsed
        return None

    def _dates_list_to_dict(self, dates_list: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Convert list of {date_type, date_value} dicts to a flat dict.

        E.g. [{"date_type": "Study Date", "date_value": "1/4/2016"}]
          → {"study_date": "1/4/2016"}
        """
        result = {}
        for d in dates_list:
            if not isinstance(d, dict):
                continue
            date_type = (d.get("date_type") or "").strip()
            date_value = (d.get("date_value") or "").strip()
            if date_type and date_value:
                # Normalize key: "Study Date" → "study_date", "Report Signed" → "report_signed"
                key = date_type.lower().replace(" ", "_")
                result[key] = date_value
        return result

    def _parse_date_to_fhir(self, date_str: str) -> Optional[str]:
        """
        Parse various date formats to FHIR-compatible date string.

        Handles: MM/DD/YYYY, YYYY-MM-DD, Month DD YYYY, DD-Mon-YYYY, etc.
        Returns YYYY-MM-DD format.
        """
        if not date_str or date_str.strip().lower() in ("", "none", "n/a", "unknown"):
            return None

        date_str = date_str.strip()

        # Already FHIR format (YYYY-MM-DD or ISO datetime)
        if re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
            return date_str[:10]

        formats = [
            "%m/%d/%Y",      # 01/21/2025
            "%m-%d-%Y",      # 01-21-2025
            "%d/%m/%Y",      # 21/01/2025
            "%B %d, %Y",     # January 21, 2025
            "%b %d, %Y",     # Jan 21, 2025
            "%d %B %Y",      # 21 January 2025
            "%d %b %Y",      # 21 Jan 2025
            "%d-%b-%Y",      # 21-Jan-2025
            "%Y%m%d",        # 20250121
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        self.logger.debug(f"Could not parse date: {date_str}")
        return None

    def _validate_bundle(self, bundle: Bundle) -> List[str]:
        """Validate FHIR bundle against schema."""
        errors = []
        try:
            bundle.model_dump_json()
        except Exception as e:
            errors.append(str(e))
        return errors

    def get_observation_count(self, bundle_dict: Dict) -> int:
        """Count observations in a bundle dict."""
        if 'entry' not in bundle_dict:
            return 0
        return sum(
            1 for entry in bundle_dict['entry']
            if entry.get('resource', {}).get('resourceType') == 'Observation'
        )
