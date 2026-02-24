#!/usr/bin/env python3
"""
Template Validation and Audit Script

Validates all lab report templates for:
1. Required fields (aliases, unit_column, flag_in_result_column, column_headers)
2. LOINC code coverage
3. Consistent structure
4. Missing or duplicate aliases across templates

Usage:
    python scripts/validate_templates.py
    python scripts/validate_templates.py --fix  # Auto-fix simple issues
    python scripts/validate_templates.py --verbose  # Show detailed info
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import argparse


# Known LOINC codes for validation
KNOWN_LOINC_CODES = {
    # CBC
    "wbc": "6690-2",
    "rbc": "789-8",
    "hemoglobin": "718-7",
    "hematocrit": "4544-3",
    "mcv": "787-2",
    "mch": "785-6",
    "mchc": "786-4",
    "rdw": "788-0",
    "platelets": "777-3",
    "mpv": "32623-1",
    # Differential
    "neutrophils_percent": "770-8",
    "lymphocytes_percent": "736-9",
    "monocytes_percent": "5905-5",
    "eosinophils_percent": "713-8",
    "basophils_percent": "706-2",
    "neutrophils_absolute": "751-8",
    "lymphocytes_absolute": "731-0",
    "monocytes_absolute": "742-7",
    "eosinophils_absolute": "711-2",
    "basophils_absolute": "704-7",
    "immature_granulocytes_percent": "71695-1",
    "immature_granulocytes_absolute": "53115-2",
    # CD4/CD8
    "absolute_cd4": "24467-3",
    "cd4_percent": "8123-2",
    "absolute_cd8": "8137-2",
    "cd8_percent": "8101-8",
    "cd4_cd8_ratio": "54218-3",
    # CMP
    "glucose": "2345-7",
    "bun": "3094-0",
    "creatinine": "2160-0",
    "egfr": "33914-3",
    "sodium": "2951-2",
    "potassium": "2823-3",
    "chloride": "2075-0",
    "carbon_dioxide": "2028-9",
    "calcium": "17861-6",
    "total_protein": "2885-2",
    "albumin": "1751-7",
    "globulin": "10834-0",
    "ag_ratio": "1759-0",
    "bilirubin_total": "1975-2",
    "alkaline_phosphatase": "6768-6",
    "ast": "1920-8",
    "alt": "1742-6",
    # Lipid Panel
    "cholesterol_total": "2093-3",
    "triglycerides": "2571-8",
    "hdl": "2085-9",
    "ldl": "13457-7",
    "vldl": "13458-5",
    "total_hdl_ratio": "9830-1",
    "non_hdl_cholesterol": "43396-1",
    # Thyroid
    "tsh": "3016-3",
    "t4_free": "3027-0",
    "t4_total": "3026-2",
    "t3_free": "3051-3",
    "t3_total": "3053-9",
    "t3_uptake": "3050-5",
    # Liver
    "bilirubin_direct": "1968-7",
    "ggt": "2324-2",
    # Coagulation
    "pt": "5902-2",
    "inr": "34714-6",
    "ptt": "14975-5",
    "fibrinogen": "3255-7",
}


class TemplateValidator:
    """Validates lab report templates."""

    def __init__(self, templates_dir: Path, verbose: bool = False):
        self.templates_dir = templates_dir
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def validate_all(self) -> Tuple[int, int, int]:
        """
        Validate all templates.

        Returns:
            Tuple of (error_count, warning_count, info_count)
        """
        templates = list(self.templates_dir.glob("*.json"))
        # Exclude internal files
        templates = [t for t in templates if not t.name.startswith("_")]

        print(f"\n{'='*60}")
        print(f"Template Validation Report")
        print(f"{'='*60}")
        print(f"Templates directory: {self.templates_dir}")
        print(f"Templates found: {len(templates)}")
        print(f"{'='*60}\n")

        all_aliases: Dict[str, List[str]] = defaultdict(list)

        for template_path in sorted(templates):
            self._validate_template(template_path, all_aliases)

        # Check for duplicate aliases across templates
        self._check_duplicate_aliases(all_aliases)

        # Print summary
        self._print_summary()

        return len(self.errors), len(self.warnings), len(self.info)

    def _validate_template(self, path: Path, all_aliases: Dict[str, List[str]]):
        """Validate a single template file."""
        print(f"\n--- {path.name} ---")

        try:
            with open(path, 'r') as f:
                template = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"{path.name}: Invalid JSON - {e}")
            print(f"  ERROR: Invalid JSON - {e}")
            return
        except Exception as e:
            self.errors.append(f"{path.name}: Failed to read - {e}")
            print(f"  ERROR: Failed to read - {e}")
            return

        # Basic structure validation
        self._validate_structure(path.name, template)

        # Layout signature validation
        self._validate_layout_signature(path.name, template)

        # Field mappings validation
        self._validate_field_mappings(path.name, template, all_aliases)

        if self.verbose:
            field_count = len(template.get('field_mappings', {}))
            print(f"  Fields: {field_count}")
            print(f"  Version: {template.get('version', 'N/A')}")

    def _validate_structure(self, name: str, template: Dict[str, Any]):
        """Validate basic template structure."""
        required_top_level = ['id', 'vendor', 'test_type', 'version', 'field_mappings']

        for field in required_top_level:
            if field not in template:
                self.errors.append(f"{name}: Missing required field '{field}'")
                print(f"  ERROR: Missing required field '{field}'")

        # Check version is >= 2 (updated templates)
        version = template.get('version', 0)
        if version < 2:
            self.warnings.append(f"{name}: Old template version ({version}), should be >= 2")
            print(f"  WARNING: Old template version ({version})")

    def _validate_layout_signature(self, name: str, template: Dict[str, Any]):
        """Validate layout_signature has required fields."""
        layout = template.get('layout_signature', {})

        if not layout:
            self.warnings.append(f"{name}: Missing layout_signature")
            print(f"  WARNING: Missing layout_signature")
            return

        # Check for new required fields
        if 'flag_in_result_column' not in layout:
            self.warnings.append(f"{name}: layout_signature missing 'flag_in_result_column'")
            print(f"  WARNING: Missing 'flag_in_result_column' in layout_signature")

        if 'column_headers' not in layout:
            self.info.append(f"{name}: layout_signature missing 'column_headers' (optional)")
            if self.verbose:
                print(f"  INFO: Missing 'column_headers' in layout_signature")

    def _validate_field_mappings(
        self,
        name: str,
        template: Dict[str, Any],
        all_aliases: Dict[str, List[str]]
    ):
        """Validate field mappings have required fields and collect aliases."""
        field_mappings = template.get('field_mappings', {})

        if not field_mappings:
            self.errors.append(f"{name}: No field_mappings defined")
            print(f"  ERROR: No field_mappings defined")
            return

        missing_aliases = []
        missing_unit_column = []
        missing_loinc = []
        wrong_loinc = []

        for field_name, field_config in field_mappings.items():
            # Check for aliases
            if 'aliases' not in field_config:
                missing_aliases.append(field_name)
            else:
                # Collect aliases for duplicate checking
                for alias in field_config['aliases']:
                    all_aliases[alias.lower()].append(f"{name}:{field_name}")

            # Check for unit_column
            if 'unit_column' not in field_config:
                missing_unit_column.append(field_name)

            # Check LOINC code
            loinc = field_config.get('loinc_code')
            if not loinc:
                missing_loinc.append(field_name)
            elif field_name in KNOWN_LOINC_CODES:
                expected = KNOWN_LOINC_CODES[field_name]
                if loinc != expected:
                    wrong_loinc.append(f"{field_name}: {loinc} (expected {expected})")

        if missing_aliases:
            self.warnings.append(f"{name}: {len(missing_aliases)} fields missing 'aliases'")
            print(f"  WARNING: {len(missing_aliases)} fields missing 'aliases': {missing_aliases[:5]}...")

        if missing_unit_column:
            self.warnings.append(f"{name}: {len(missing_unit_column)} fields missing 'unit_column'")
            print(f"  WARNING: {len(missing_unit_column)} fields missing 'unit_column': {missing_unit_column[:5]}...")

        if missing_loinc and self.verbose:
            self.info.append(f"{name}: {len(missing_loinc)} fields missing 'loinc_code'")
            print(f"  INFO: {len(missing_loinc)} fields missing 'loinc_code': {missing_loinc[:5]}...")

        if wrong_loinc:
            self.warnings.append(f"{name}: {len(wrong_loinc)} fields with incorrect LOINC codes")
            print(f"  WARNING: Incorrect LOINC codes: {wrong_loinc[:3]}...")

    def _check_duplicate_aliases(self, all_aliases: Dict[str, List[str]]):
        """Check for aliases used in multiple templates."""
        print(f"\n--- Cross-Template Alias Check ---")

        duplicates = {
            alias: sources
            for alias, sources in all_aliases.items()
            if len(sources) > 1
        }

        if duplicates:
            # Filter to only show duplicates across DIFFERENT templates
            cross_template_dups = {}
            for alias, sources in duplicates.items():
                templates = set(s.split(':')[0] for s in sources)
                if len(templates) > 1:
                    cross_template_dups[alias] = sources

            if cross_template_dups:
                self.info.append(f"Found {len(cross_template_dups)} aliases used across multiple templates")
                if self.verbose:
                    print(f"  INFO: {len(cross_template_dups)} aliases shared across templates")
                    for alias, sources in list(cross_template_dups.items())[:5]:
                        print(f"    '{alias}' used in: {', '.join(sources)}")
            else:
                print(f"  OK: No problematic alias duplicates found")
        else:
            print(f"  OK: No duplicate aliases found")

    def _print_summary(self):
        """Print validation summary."""
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Errors:   {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Info:     {len(self.info)}")

        if self.errors:
            print(f"\nERRORS (must fix):")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\nWARNINGS (should fix):")
            for warning in self.warnings[:10]:  # Limit to first 10
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

        print(f"{'='*60}\n")


def auto_fix_template(path: Path) -> bool:
    """
    Auto-fix simple issues in a template.

    Fixes:
    - Add unit_column: null to fields missing it
    - Add flag_in_result_column: false to layout_signature

    Returns:
        True if changes were made
    """
    try:
        with open(path, 'r') as f:
            template = json.load(f)
    except Exception:
        return False

    modified = False

    # Fix layout_signature
    if 'layout_signature' in template:
        if 'flag_in_result_column' not in template['layout_signature']:
            template['layout_signature']['flag_in_result_column'] = False
            modified = True
        if 'column_headers' not in template['layout_signature']:
            template['layout_signature']['column_headers'] = None
            modified = True

    # Fix field_mappings
    for field_name, field_config in template.get('field_mappings', {}).items():
        if 'unit_column' not in field_config:
            field_config['unit_column'] = None
            modified = True

    if modified:
        # Bump version if not already done
        if template.get('version', 1) < 2:
            template['version'] = 2

        with open(path, 'w') as f:
            json.dump(template, f, indent=2)
            f.write('\n')

        print(f"  Fixed: {path.name}")

    return modified


def main():
    parser = argparse.ArgumentParser(description="Validate lab report templates")
    parser.add_argument("--fix", action="store_true", help="Auto-fix simple issues")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed info")
    parser.add_argument("--path", type=str, help="Custom templates path")
    args = parser.parse_args()

    # Find templates directory
    if args.path:
        templates_dir = Path(args.path)
    else:
        # Try to find the templates directory relative to this script
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        templates_dir = project_root / "src" / "medical_ingestion" / "processors" / "lab" / "templates"

    if not templates_dir.exists():
        print(f"ERROR: Templates directory not found: {templates_dir}")
        sys.exit(1)

    # Auto-fix if requested
    if args.fix:
        print("\n=== Auto-fixing templates ===\n")
        templates = list(templates_dir.glob("*.json"))
        templates = [t for t in templates if not t.name.startswith("_")]
        fixed_count = 0
        for template_path in sorted(templates):
            if auto_fix_template(template_path):
                fixed_count += 1
        print(f"\nFixed {fixed_count} templates\n")

    # Run validation
    validator = TemplateValidator(templates_dir, verbose=args.verbose)
    errors, warnings, info = validator.validate_all()

    # Exit with error code if there are errors
    if errors > 0:
        sys.exit(1)
    elif warnings > 0:
        sys.exit(0)  # Warnings don't fail the validation
    else:
        print("All templates are valid!")
        sys.exit(0)


if __name__ == "__main__":
    main()
