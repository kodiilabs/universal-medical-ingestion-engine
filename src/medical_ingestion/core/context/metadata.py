# ============================================================================
# src/medical_ingestion/core/context/metadata.py
# ============================================================================
"""
Document Metadata Classes

Structured representations for:
- Patient information
- Practitioner/Provider information
- Organization/Facility information
- Specimen details
- Report dates and identifiers
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, date


@dataclass
class PatientInfo:
    """
    Patient demographic and identification information.
    Extracted from the header/demographic section of lab reports.
    """
    # Core identifiers
    patient_id: Optional[str] = None  # MRN, Account #, Patient ID
    external_id: Optional[str] = None  # SSN last 4, insurance ID

    # Demographics
    name: Optional[str] = None  # Full name as appears on report
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    middle_name: Optional[str] = None

    date_of_birth: Optional[date] = None
    age: Optional[int] = None  # Age at time of test
    age_unit: str = "years"  # "years", "months", "days"

    sex: Optional[str] = None  # "M", "F", "U" (unknown)
    gender: Optional[str] = None  # Gender identity if different from sex

    # Contact
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    phone: Optional[str] = None

    # Extraction metadata
    confidence: float = 0.0
    source_page: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "name": self.name,
            "date_of_birth": self.date_of_birth.isoformat() if self.date_of_birth else None,
            "age": self.age,
            "sex": self.sex,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "zip_code": self.zip_code,
            "phone": self.phone,
        }


@dataclass
class PractitionerInfo:
    """
    Healthcare practitioner information.
    Could be ordering physician, lab director, or other providers.
    """
    # Role
    role: str = "ordering"  # "ordering", "lab_director", "performing", "reviewing"

    # Identifiers
    npi: Optional[str] = None  # National Provider Identifier
    license_number: Optional[str] = None

    # Name
    name: Optional[str] = None  # Full name as appears
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    credentials: Optional[str] = None  # "MD", "DO", "PhD", etc.
    title: Optional[str] = None  # "Dr.", etc.

    # Contact
    phone: Optional[str] = None
    fax: Optional[str] = None
    address: Optional[str] = None

    # Organization affiliation
    organization: Optional[str] = None
    department: Optional[str] = None

    # Extraction metadata
    confidence: float = 0.0
    source_page: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "npi": self.npi,
            "name": self.name,
            "credentials": self.credentials,
            "organization": self.organization,
            "phone": self.phone,
        }


@dataclass
class OrganizationInfo:
    """
    Laboratory or healthcare organization information.
    """
    # Role
    role: str = "performing"  # "performing", "ordering_facility", "sending"

    # Identifiers
    clia_number: Optional[str] = None  # Lab certification
    npi: Optional[str] = None  # Organization NPI
    lab_id: Optional[str] = None  # Internal lab identifier

    # Name and branding
    name: Optional[str] = None  # "LabCorp", "Quest Diagnostics"
    display_name: Optional[str] = None  # Full legal name
    type: Optional[str] = None  # "laboratory", "hospital", "clinic"

    # Location
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: str = "US"

    # Contact
    phone: Optional[str] = None
    fax: Optional[str] = None
    website: Optional[str] = None

    # Accreditation
    accreditations: List[str] = field(default_factory=list)  # CAP, AABB, etc.

    # Extraction metadata
    confidence: float = 0.0
    source_page: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "name": self.name,
            "clia_number": self.clia_number,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "zip_code": self.zip_code,
            "phone": self.phone,
            "accreditations": self.accreditations,
        }


@dataclass
class SpecimenInfo:
    """
    Specimen/sample information.
    """
    # Identifiers
    specimen_id: Optional[str] = None  # Accession number, specimen ID
    accession_number: Optional[str] = None
    container_id: Optional[str] = None

    # Type and source
    specimen_type: Optional[str] = None  # "Blood", "Serum", "Plasma", "Urine"
    specimen_source: Optional[str] = None  # "Venous", "Capillary", "Arterial"
    collection_method: Optional[str] = None  # "Venipuncture", "Fingerstick"

    # Collection details
    collection_datetime: Optional[datetime] = None
    collection_date: Optional[date] = None
    collection_time: Optional[str] = None  # Time string if datetime not parseable
    collected_by: Optional[str] = None  # Phlebotomist name/ID
    collection_site: Optional[str] = None  # "Left arm", "Right hand"

    # Receipt at lab
    received_datetime: Optional[datetime] = None

    # Condition
    volume: Optional[str] = None  # "5 mL"
    condition: Optional[str] = None  # "Satisfactory", "Hemolyzed", "Lipemic"
    rejection_reason: Optional[str] = None  # If rejected

    # Fasting status
    fasting: Optional[bool] = None
    fasting_hours: Optional[int] = None

    # Extraction metadata
    confidence: float = 0.0
    source_page: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "specimen_id": self.specimen_id,
            "accession_number": self.accession_number,
            "specimen_type": self.specimen_type,
            "collection_datetime": self.collection_datetime.isoformat() if self.collection_datetime else None,
            "collection_date": self.collection_date.isoformat() if self.collection_date else None,
            "received_datetime": self.received_datetime.isoformat() if self.received_datetime else None,
            "condition": self.condition,
            "fasting": self.fasting,
        }


@dataclass
class ReportInfo:
    """
    Report-level metadata and dates.
    """
    # Identifiers
    report_id: Optional[str] = None  # Report number
    order_id: Optional[str] = None  # Order/requisition number

    # Dates
    order_date: Optional[date] = None  # When test was ordered
    collection_date: Optional[date] = None  # When specimen collected
    received_date: Optional[date] = None  # When lab received specimen
    resulted_date: Optional[date] = None  # When results available
    report_date: Optional[date] = None  # Report generation date

    # Full datetimes if available
    collection_datetime: Optional[datetime] = None
    received_datetime: Optional[datetime] = None
    resulted_datetime: Optional[datetime] = None
    report_datetime: Optional[datetime] = None

    # Status
    status: str = "final"  # "preliminary", "final", "corrected", "amended"
    is_preliminary: bool = False
    is_corrected: bool = False
    correction_reason: Optional[str] = None

    # Report format
    report_format: Optional[str] = None  # "Enterprise Report", "Patient Report"
    page_count: Optional[int] = None

    # Comments/Notes
    clinical_notes: List[str] = field(default_factory=list)
    technical_notes: List[str] = field(default_factory=list)
    footnotes: List[str] = field(default_factory=list)

    # Extraction metadata
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "order_id": self.order_id,
            "order_date": self.order_date.isoformat() if self.order_date else None,
            "collection_date": self.collection_date.isoformat() if self.collection_date else None,
            "received_date": self.received_date.isoformat() if self.received_date else None,
            "resulted_date": self.resulted_date.isoformat() if self.resulted_date else None,
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "status": self.status,
            "is_preliminary": self.is_preliminary,
            "clinical_notes": self.clinical_notes,
            "technical_notes": self.technical_notes,
        }


@dataclass
class DocumentMetadata:
    """
    Complete document metadata container.
    Aggregates all metadata from a medical document.
    """
    # Core entities
    patient: Optional[PatientInfo] = None
    specimen: Optional[SpecimenInfo] = None
    report: Optional[ReportInfo] = None

    # Practitioners (can have multiple)
    ordering_provider: Optional[PractitionerInfo] = None
    lab_director: Optional[PractitionerInfo] = None
    other_practitioners: List[PractitionerInfo] = field(default_factory=list)

    # Organizations (can have multiple)
    performing_lab: Optional[OrganizationInfo] = None
    ordering_facility: Optional[OrganizationInfo] = None
    other_organizations: List[OrganizationInfo] = field(default_factory=list)

    # Extraction quality
    overall_confidence: float = 0.0
    fields_extracted: int = 0
    fields_missing: List[str] = field(default_factory=list)
    extraction_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient": self.patient.to_dict() if self.patient else None,
            "specimen": self.specimen.to_dict() if self.specimen else None,
            "report": self.report.to_dict() if self.report else None,
            "ordering_provider": self.ordering_provider.to_dict() if self.ordering_provider else None,
            "lab_director": self.lab_director.to_dict() if self.lab_director else None,
            "performing_lab": self.performing_lab.to_dict() if self.performing_lab else None,
            "ordering_facility": self.ordering_facility.to_dict() if self.ordering_facility else None,
            "overall_confidence": self.overall_confidence,
            "fields_extracted": self.fields_extracted,
            "fields_missing": self.fields_missing,
        }

    def get_all_practitioners(self) -> List[PractitionerInfo]:
        """Get all practitioners in a single list."""
        practitioners = []
        if self.ordering_provider:
            practitioners.append(self.ordering_provider)
        if self.lab_director:
            practitioners.append(self.lab_director)
        practitioners.extend(self.other_practitioners)
        return practitioners

    def get_all_organizations(self) -> List[OrganizationInfo]:
        """Get all organizations in a single list."""
        orgs = []
        if self.performing_lab:
            orgs.append(self.performing_lab)
        if self.ordering_facility:
            orgs.append(self.ordering_facility)
        orgs.extend(self.other_organizations)
        return orgs
