#!/usr/bin/env python3
# ============================================================================
# scripts/create_templates.py
# ============================================================================
"""
Create Document Templates

Generates sample medical document templates for testing and development:
- Lab reports (CBC, CMP, Lipid Panel, etc.)
- Prescriptions
- Radiology reports (CT, MRI, X-ray)
- Pathology reports
- Discharge summaries

Usage:
    python scripts/create_templates.py
    python scripts/create_templates.py --type lab
    python scripts/create_templates.py --output-dir ./samples
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path


TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
SAMPLES_DIR = Path(__file__).parent.parent / "samples"


def create_lab_report_template() -> str:
    """Create a lab report template."""
    return """================================================================================
                         LABORATORY REPORT
================================================================================

PATIENT INFORMATION
-------------------
Patient Name:    {{patient_name}}
DOB:             {{patient_dob}}
MRN:             {{patient_mrn}}
Gender:          {{patient_gender}}

SPECIMEN INFORMATION
--------------------
Collection Date: {{collection_date}}
Collection Time: {{collection_time}}
Received Date:   {{received_date}}
Specimen Type:   {{specimen_type}}

ORDERING PROVIDER
-----------------
Provider:        {{provider_name}}
NPI:             {{provider_npi}}

================================================================================
                              TEST RESULTS
================================================================================

COMPLETE BLOOD COUNT (CBC)
--------------------------
Test Name                Result      Units       Reference Range     Flag
--------------------------------------------------------------------------------
WBC                      {{wbc}}          K/uL        4.5-11.0
RBC                      {{rbc}}          M/uL        4.5-5.9 (M) / 4.1-5.1 (F)
Hemoglobin               {{hgb}}          g/dL        13.5-17.5 (M) / 12.0-15.5 (F)
Hematocrit               {{hct}}          %           38.8-50.0 (M) / 34.9-44.5 (F)
MCV                      {{mcv}}          fL          80-100
MCH                      {{mch}}          pg          27-33
MCHC                     {{mchc}}         g/dL        32-36
Platelets                {{plt}}          K/uL        150-400
RDW                      {{rdw}}          %           11.5-14.5

COMPREHENSIVE METABOLIC PANEL (CMP)
-----------------------------------
Test Name                Result      Units       Reference Range     Flag
--------------------------------------------------------------------------------
Glucose                  {{glucose}}      mg/dL       70-100
BUN                      {{bun}}          mg/dL       7-20
Creatinine               {{creatinine}}   mg/dL       0.7-1.3
Sodium                   {{sodium}}       mEq/L       136-145
Potassium                {{potassium}}    mEq/L       3.5-5.0
Chloride                 {{chloride}}     mEq/L       98-106
CO2                      {{co2}}          mEq/L       23-29
Calcium                  {{calcium}}      mg/dL       8.5-10.5
Total Protein            {{total_protein}} g/dL       6.0-8.3
Albumin                  {{albumin}}      g/dL        3.5-5.0
Total Bilirubin          {{bilirubin}}    mg/dL       0.1-1.2
Alkaline Phosphatase     {{alp}}          U/L         44-147
AST                      {{ast}}          U/L         10-40
ALT                      {{alt}}          U/L         7-56

LIPID PANEL
-----------
Test Name                Result      Units       Reference Range     Flag
--------------------------------------------------------------------------------
Total Cholesterol        {{cholesterol}}  mg/dL       <200
Triglycerides            {{triglycerides}} mg/dL      <150
HDL Cholesterol          {{hdl}}          mg/dL       >40
LDL Cholesterol          {{ldl}}          mg/dL       <100
VLDL Cholesterol         {{vldl}}         mg/dL       5-40

================================================================================

Comments: {{comments}}

Verified By: {{verified_by}}
Verified Date: {{verified_date}}

================================================================================
                         END OF REPORT
================================================================================
"""


def create_prescription_template() -> str:
    """Create a prescription template."""
    return """================================================================================
                           PRESCRIPTION
================================================================================

PRESCRIBER INFORMATION
----------------------
Prescriber Name:     {{prescriber_name}}
DEA Number:          {{prescriber_dea}}
NPI:                 {{prescriber_npi}}
License Number:      {{prescriber_license}}
Address:             {{prescriber_address}}
Phone:               {{prescriber_phone}}
Fax:                 {{prescriber_fax}}

PATIENT INFORMATION
-------------------
Patient Name:        {{patient_name}}
Date of Birth:       {{patient_dob}}
Address:             {{patient_address}}
Phone:               {{patient_phone}}
Allergies:           {{patient_allergies}}

================================================================================
                           MEDICATION ORDER
================================================================================

Rx Date:             {{rx_date}}

MEDICATION 1
------------
Drug Name:           {{drug_name_1}}
Strength:            {{drug_strength_1}}
Form:                {{drug_form_1}}
Route:               {{drug_route_1}}
Directions:          {{drug_directions_1}}
Quantity:            {{drug_quantity_1}}
Refills:             {{drug_refills_1}}
DAW:                 {{drug_daw_1}}

MEDICATION 2 (if applicable)
----------------------------
Drug Name:           {{drug_name_2}}
Strength:            {{drug_strength_2}}
Form:                {{drug_form_2}}
Route:               {{drug_route_2}}
Directions:          {{drug_directions_2}}
Quantity:            {{drug_quantity_2}}
Refills:             {{drug_refills_2}}
DAW:                 {{drug_daw_2}}

================================================================================

Special Instructions: {{special_instructions}}

Prescriber Signature: _________________________

Date: {{signature_date}}

================================================================================
"""


def create_radiology_report_template() -> str:
    """Create a radiology report template."""
    return """================================================================================
                         RADIOLOGY REPORT
================================================================================

PATIENT INFORMATION
-------------------
Patient Name:        {{patient_name}}
DOB:                 {{patient_dob}}
MRN:                 {{patient_mrn}}
Gender:              {{patient_gender}}

EXAM INFORMATION
----------------
Exam Date:           {{exam_date}}
Exam Time:           {{exam_time}}
Accession Number:    {{accession_number}}
Modality:            {{modality}}
Exam Description:    {{exam_description}}

ORDERING INFORMATION
--------------------
Ordering Provider:   {{ordering_provider}}
Clinical History:    {{clinical_history}}
Indication:          {{indication}}

================================================================================
                              FINDINGS
================================================================================

{{findings}}

================================================================================
                            IMPRESSION
================================================================================

{{impression}}

================================================================================

COMPARISON: {{comparison}}

TECHNIQUE: {{technique}}

RADIATION DOSE: {{radiation_dose}}

================================================================================

Dictated By:         {{dictated_by}}
Transcribed By:      {{transcribed_by}}
Attending Physician: {{attending_physician}}

Report Status:       {{report_status}}
Report Date:         {{report_date}}

================================================================================
                         END OF REPORT
================================================================================
"""


def create_pathology_report_template() -> str:
    """Create a pathology report template."""
    return """================================================================================
                        PATHOLOGY REPORT
================================================================================

PATIENT INFORMATION
-------------------
Patient Name:        {{patient_name}}
DOB:                 {{patient_dob}}
MRN:                 {{patient_mrn}}
Gender:              {{patient_gender}}

SPECIMEN INFORMATION
--------------------
Accession Number:    {{accession_number}}
Collection Date:     {{collection_date}}
Received Date:       {{received_date}}
Specimen Type:       {{specimen_type}}
Specimen Source:     {{specimen_source}}

CLINICAL INFORMATION
--------------------
Clinical History:    {{clinical_history}}
Preoperative Diagnosis: {{preop_diagnosis}}

================================================================================
                         GROSS DESCRIPTION
================================================================================

{{gross_description}}

================================================================================
                       MICROSCOPIC DESCRIPTION
================================================================================

{{microscopic_description}}

================================================================================
                             DIAGNOSIS
================================================================================

{{diagnosis}}

================================================================================

SYNOPTIC REPORT (if applicable):
{{synoptic_report}}

ANCILLARY STUDIES:
{{ancillary_studies}}

COMMENT:
{{comment}}

================================================================================

Pathologist:         {{pathologist}}
Date Signed:         {{date_signed}}

================================================================================
                         END OF REPORT
================================================================================
"""


def create_discharge_summary_template() -> str:
    """Create a discharge summary template."""
    return """================================================================================
                       DISCHARGE SUMMARY
================================================================================

PATIENT INFORMATION
-------------------
Patient Name:        {{patient_name}}
DOB:                 {{patient_dob}}
MRN:                 {{patient_mrn}}
Gender:              {{patient_gender}}

ADMISSION INFORMATION
---------------------
Admission Date:      {{admission_date}}
Discharge Date:      {{discharge_date}}
Length of Stay:      {{length_of_stay}} days
Attending Physician: {{attending_physician}}

DIAGNOSES
---------
Principal Diagnosis: {{principal_diagnosis}}

Secondary Diagnoses:
{{secondary_diagnoses}}

PROCEDURES PERFORMED
--------------------
{{procedures}}

================================================================================
                        HOSPITAL COURSE
================================================================================

{{hospital_course}}

================================================================================
                     DISCHARGE MEDICATIONS
================================================================================

{{discharge_medications}}

================================================================================
                    DISCHARGE INSTRUCTIONS
================================================================================

Activity:            {{activity_instructions}}
Diet:                {{diet_instructions}}
Wound Care:          {{wound_care}}

Follow-up Appointments:
{{followup_appointments}}

Warning Signs - Return to ED if:
{{warning_signs}}

================================================================================

Discharge Condition: {{discharge_condition}}
Discharge Disposition: {{discharge_disposition}}

Dictated By:         {{dictated_by}}
Attending Physician: {{attending_physician}}

================================================================================
                         END OF SUMMARY
================================================================================
"""


def create_sample_lab_report() -> str:
    """Create a sample filled lab report."""
    template = create_lab_report_template()

    # Generate sample values
    values = {
        "patient_name": "John Doe",
        "patient_dob": "01/15/1965",
        "patient_mrn": "MRN123456",
        "patient_gender": "Male",
        "collection_date": datetime.now().strftime("%m/%d/%Y"),
        "collection_time": "08:30",
        "received_date": datetime.now().strftime("%m/%d/%Y"),
        "specimen_type": "Whole Blood",
        "provider_name": "Dr. Jane Smith, MD",
        "provider_npi": "1234567890",
        # CBC values
        "wbc": "7.2",
        "rbc": "4.8",
        "hgb": "14.5",
        "hct": "43.2",
        "mcv": "90",
        "mch": "30.2",
        "mchc": "33.6",
        "plt": "225",
        "rdw": "12.8",
        # CMP values
        "glucose": "95",
        "bun": "15",
        "creatinine": "1.0",
        "sodium": "140",
        "potassium": "4.2",
        "chloride": "102",
        "co2": "25",
        "calcium": "9.5",
        "total_protein": "7.0",
        "albumin": "4.2",
        "bilirubin": "0.8",
        "alp": "75",
        "ast": "25",
        "alt": "30",
        # Lipid values
        "cholesterol": "195",
        "triglycerides": "120",
        "hdl": "55",
        "ldl": "116",
        "vldl": "24",
        "comments": "No significant abnormalities noted.",
        "verified_by": "Lab Director, PhD",
        "verified_date": datetime.now().strftime("%m/%d/%Y %H:%M"),
    }

    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)

    return template


def create_sample_prescription() -> str:
    """Create a sample filled prescription."""
    template = create_prescription_template()

    values = {
        "prescriber_name": "Dr. Jane Smith, MD",
        "prescriber_dea": "AS1234567",
        "prescriber_npi": "1234567890",
        "prescriber_license": "MD12345",
        "prescriber_address": "123 Medical Center Dr, Suite 100, City, ST 12345",
        "prescriber_phone": "(555) 123-4567",
        "prescriber_fax": "(555) 123-4568",
        "patient_name": "John Doe",
        "patient_dob": "01/15/1965",
        "patient_address": "456 Patient Lane, City, ST 12345",
        "patient_phone": "(555) 987-6543",
        "patient_allergies": "Penicillin, Sulfa",
        "rx_date": datetime.now().strftime("%m/%d/%Y"),
        "drug_name_1": "Lisinopril",
        "drug_strength_1": "10 mg",
        "drug_form_1": "Tablet",
        "drug_route_1": "Oral",
        "drug_directions_1": "Take 1 tablet by mouth once daily",
        "drug_quantity_1": "#30 (Thirty)",
        "drug_refills_1": "3",
        "drug_daw_1": "Substitution Permitted",
        "drug_name_2": "Metformin",
        "drug_strength_2": "500 mg",
        "drug_form_2": "Tablet",
        "drug_route_2": "Oral",
        "drug_directions_2": "Take 1 tablet by mouth twice daily with meals",
        "drug_quantity_2": "#60 (Sixty)",
        "drug_refills_2": "3",
        "drug_daw_2": "Substitution Permitted",
        "special_instructions": "Monitor blood pressure and blood glucose regularly.",
        "signature_date": datetime.now().strftime("%m/%d/%Y"),
    }

    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)

    return template


def create_sample_radiology_report() -> str:
    """Create a sample filled radiology report."""
    template = create_radiology_report_template()

    values = {
        "patient_name": "John Doe",
        "patient_dob": "01/15/1965",
        "patient_mrn": "MRN123456",
        "patient_gender": "Male",
        "exam_date": datetime.now().strftime("%m/%d/%Y"),
        "exam_time": "10:30",
        "accession_number": "RAD-2024-001234",
        "modality": "CT",
        "exam_description": "CT CHEST WITH CONTRAST",
        "ordering_provider": "Dr. Jane Smith, MD",
        "clinical_history": "59-year-old male with persistent cough and weight loss",
        "indication": "Rule out malignancy",
        "findings": """LUNGS: There is a 2.3 cm spiculated nodule in the right upper lobe
(series 4, image 52). No additional pulmonary nodules identified. No pleural effusion.
No pneumothorax.

MEDIASTINUM: No significant mediastinal or hilar lymphadenopathy. Heart size is normal.
No pericardial effusion.

CHEST WALL: No chest wall abnormality.

UPPER ABDOMEN: Limited evaluation. No gross abnormality of the visualized liver,
spleen, or adrenal glands.""",
        "impression": """1. 2.3 cm spiculated nodule in the right upper lobe, suspicious for
   primary lung malignancy. PET-CT recommended for further evaluation.

2. No mediastinal or hilar lymphadenopathy.

3. No pleural effusion.""",
        "comparison": "No prior imaging available for comparison.",
        "technique": "CT of the chest was performed with IV contrast (100 mL Omnipaque 350).",
        "radiation_dose": "DLP: 450 mGy-cm",
        "dictated_by": "Dr. Robert Johnson, MD",
        "transcribed_by": "Medical Transcription Services",
        "attending_physician": "Dr. Robert Johnson, MD",
        "report_status": "FINAL",
        "report_date": datetime.now().strftime("%m/%d/%Y %H:%M"),
    }

    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)

    return template


def save_templates(output_dir: Path) -> None:
    """Save all templates to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    templates = {
        "lab_report_template.txt": create_lab_report_template(),
        "prescription_template.txt": create_prescription_template(),
        "radiology_report_template.txt": create_radiology_report_template(),
        "pathology_report_template.txt": create_pathology_report_template(),
        "discharge_summary_template.txt": create_discharge_summary_template(),
    }

    for filename, content in templates.items():
        filepath = output_dir / filename
        filepath.write_text(content)
        print(f"Created: {filepath}")


def save_samples(output_dir: Path) -> None:
    """Save all sample documents to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "sample_lab_report.txt": create_sample_lab_report(),
        "sample_prescription.txt": create_sample_prescription(),
        "sample_radiology_report.txt": create_sample_radiology_report(),
    }

    for filename, content in samples.items():
        filepath = output_dir / filename
        filepath.write_text(content)
        print(f"Created: {filepath}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create medical document templates and samples"
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["lab", "prescription", "radiology", "pathology", "discharge", "all"],
        default="all",
        help="Type of template to create"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for templates"
    )

    parser.add_argument(
        "--samples",
        action="store_true",
        help="Also create filled sample documents"
    )

    parser.add_argument(
        "--templates-only",
        action="store_true",
        help="Only create blank templates"
    )

    args = parser.parse_args()

    # Determine output directories
    templates_dir = Path(args.output_dir) if args.output_dir else TEMPLATES_DIR
    samples_dir = Path(args.output_dir) / "samples" if args.output_dir else SAMPLES_DIR

    print("=" * 60)
    print("Medical Document Template Generator")
    print("=" * 60)

    # Create templates
    if args.type == "all":
        print("\nCreating all templates...")
        save_templates(templates_dir)
    else:
        templates_dir.mkdir(parents=True, exist_ok=True)

        template_map = {
            "lab": ("lab_report_template.txt", create_lab_report_template),
            "prescription": ("prescription_template.txt", create_prescription_template),
            "radiology": ("radiology_report_template.txt", create_radiology_report_template),
            "pathology": ("pathology_report_template.txt", create_pathology_report_template),
            "discharge": ("discharge_summary_template.txt", create_discharge_summary_template),
        }

        filename, creator = template_map[args.type]
        filepath = templates_dir / filename
        filepath.write_text(creator())
        print(f"Created: {filepath}")

    # Create samples
    if args.samples and not args.templates_only:
        print("\nCreating sample documents...")
        save_samples(samples_dir)

    print("\n" + "=" * 60)
    print("Template generation complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
