# ============================================================================
# Fsrc/medical_ingestion/processors/lab/agents/template_matcher.py
# ============================================================================
"""
Template Matching Agent - Lab Processor

This is THE critical performance agent. Routes documents to:
- Template extraction (fast, deterministic, high accuracy) - 90% of cases
- MedGemma extraction (robust, handles unknowns) - 10% of cases

Strategy:
1. Extract text from PDF
2. Calculate fingerprint match score against known templates
3. If score >= 0.90 → use template extraction (2-4 seconds)
4. If score < 0.90 → fallback to MedGemma (8-12 seconds)

This agent is what makes the system FAST for common formats.
"""

from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import re
from ....core.agent_base import Agent
from ....core.context.processing_context import ProcessingContext
from ....config import base_settings, threshold_settings


class TemplateMatchingAgent(Agent):
    """
    Identifies known lab report formats using structural fingerprinting.
    
    Key innovation: We don't just look at text, we look at:
    - Header patterns (vendor-specific formatting)
    - Field labels (exact wording of test names)
    - Layout structure (column positions, table format)
    - Vendor markers (logos, contact info patterns)
    
    This gives us 90%+ match rate on common formats.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.templates = self._load_templates()
        self.logger.info(f"Loaded {len(self.templates)} lab templates")
    
    def get_name(self) -> str:
        return "TemplateMatchingAgent"
    
    def _load_templates(self) -> Dict[str, Dict]:
        """
        Load all lab templates from JSON files.
        
        Template structure (example):
        {
            "id": "quest_cbc_v1",
            "vendor": "Quest Diagnostics",
            "test_type": "cbc",
            "version": 1,
            "header_pattern": "Quest Diagnostics.*Complete Blood Count",
            "vendor_markers": ["Quest", "www.questdiagnostics.com", "33608"],
            "required_fields": ["WBC", "RBC", "Hemoglobin", "Hematocrit", "Platelets"],
            "layout_signature": {
                "columns": 5,
                "has_reference_range": true,
                "has_abnormal_flags": true
            },
            "field_mappings": {
                "WBC": "wbc",
                "Hemoglobin": "hemoglobin",
                ...
            }
        }
        """
        templates_dir = base_settings.get_processor_template_dir("lab")
        templates = {}
        
        # Create directory if it doesn't exist
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all JSON templates (skip files starting with _ which are helpers)
        for template_file in templates_dir.glob("*.json"):
            if template_file.name.startswith("_"):
                self.logger.debug(f"Skipping helper file: {template_file.name}")
                continue
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template = json.load(f)
                    templates[template['id']] = template
                    self.logger.debug(f"Loaded template: {template['id']}")
            except Exception as e:
                self.logger.warning(f"Failed to load template {template_file}: {e}")
        
        return templates
    
    async def execute(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Match document against known templates.

        Strategy:
        1. Try text-based fingerprint matching (fast)
        2. If confidence is low, use visual identification with MedGemma (robust)
        3. Return best match with confidence

        Returns:
            {
                "decision": "use_template" | "use_medgemma",
                "confidence": float,
                "reasoning": str,
                "template_id": str | None,
                "metadata": {...}
            }
        """
        # Extract text from PDF for fingerprinting
        # Uses pre-extracted text from classification or extracts fresh
        text = await self._extract_text_for_matching(context.document_path, context)

        # Try to match against all templates using text
        matches = []
        for template_id, template in self.templates.items():
            score, details = self._calculate_match_score_with_details(text, template)
            matches.append({
                "template_id": template_id,
                "score": score,
                "template": template,
                "details": details
            })

        # Log top matches for debugging
        for m in sorted(matches, key=lambda x: x['score'], reverse=True)[:3]:
            d = m['details']
            self.logger.debug(
                f"Template {m['template_id']}: score={m['score']:.2f} "
                f"(header={d.get('header', 0):.2f}, vendor={d.get('vendor', 0):.2f}, "
                f"fields={d.get('fields', 0):.2f}, layout={d.get('layout', 0):.2f}, "
                f"capped={d.get('capped', False)})"
            )

        # Sort by score (highest first)
        matches.sort(key=lambda x: x['score'], reverse=True)
        best_match = matches[0] if matches else None
        best_text_score = best_match['score'] if best_match else 0.0

        # Visual identification threshold - use vision if text matching is uncertain
        VISUAL_FALLBACK_THRESHOLD = 0.75

        # Decision logic based on threshold
        if best_match and best_match['score'] >= threshold_settings.TEMPLATE_MATCH_THRESHOLD:
            # HIGH CONFIDENCE - Use template extraction
            decision = "use_template"
            template_id = best_match['template_id']
            confidence = best_match['score']
            reasoning = (
                f"Strong text match to {template_id} "
                f"(score: {confidence:.2f} >= threshold: {threshold_settings.TEMPLATE_MATCH_THRESHOLD})"
            )

            # Store template info in context for extraction agent
            context.template_id = template_id
            context.template_confidence = confidence

            self.logger.info(f"Text-based template match: {template_id} ({confidence:.2f})")

        elif best_text_score < VISUAL_FALLBACK_THRESHOLD:
            # LOW TEXT CONFIDENCE - Try visual identification
            self.logger.info(
                f"Text matching score ({best_text_score:.2f}) below visual fallback threshold "
                f"({VISUAL_FALLBACK_THRESHOLD}), trying visual identification..."
            )

            visual_result = await self._identify_template_visually(context.document_path)

            # Visual identification is only valid if:
            # 1. We got a result
            # 2. Confidence is >= 0.7
            # 3. Vendor was actually identified (not None/Unknown)
            visual_vendor = visual_result.get('vendor') if visual_result else None
            visual_confidence = visual_result.get('confidence', 0) if visual_result else 0

            if visual_result and visual_confidence >= 0.7 and visual_vendor:
                # Visual identification succeeded with known vendor
                suggested_template = visual_result.get('template_id')

                # Check if suggested template exists
                if suggested_template and suggested_template in self.templates:
                    decision = "use_template"
                    template_id = suggested_template
                    confidence = visual_result['confidence']
                    reasoning = (
                        f"Visual identification matched {template_id} "
                        f"(vendor: {visual_result.get('vendor')}, "
                        f"confidence: {confidence:.2f})"
                    )

                    context.template_id = template_id
                    context.template_confidence = confidence

                    self.logger.info(f"Visual template match: {template_id} ({confidence:.2f})")

                else:
                    # Visual ID found vendor but no matching template
                    decision = "use_medgemma"
                    template_id = None
                    confidence = visual_result.get('confidence', 0.5)
                    reasoning = (
                        f"Visual ID found vendor={visual_result.get('vendor')}, "
                        f"type={visual_result.get('document_type')}, "
                        f"but no matching template '{suggested_template}'. "
                        f"Using MedGemma extraction."
                    )

                    # Store vendor info for potential template creation
                    context.sections['visual_identification'] = visual_result

                    self.logger.info(f"Visual ID succeeded but no template: {visual_result}")

            else:
                # Visual identification also failed or low confidence
                decision = "use_medgemma"
                template_id = None
                confidence = best_text_score
                reasoning = (
                    f"Both text matching (score: {best_text_score:.2f}) and "
                    f"visual identification failed. Using MedGemma extraction."
                )

                self.logger.info(f"Both text and visual matching failed")

        else:
            # MEDIUM CONFIDENCE - Not enough for template, use MedGemma
            decision = "use_medgemma"
            template_id = None
            confidence = best_text_score
            reasoning = (
                f"No strong template match "
                f"(best score: {confidence:.2f} < threshold: {threshold_settings.TEMPLATE_MATCH_THRESHOLD}). "
                f"Routing to MedGemma extraction."
            )

            self.logger.info(f"No template match, using MedGemma (best: {confidence:.2f})")

        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "template_id": template_id,
            "metadata": {
                "all_matches": [
                    {"template_id": m['template_id'], "score": m['score']}
                    for m in matches[:3]  # Top 3
                ],
                "total_templates_checked": len(self.templates),
                "used_visual_fallback": best_text_score < VISUAL_FALLBACK_THRESHOLD
            }
        }

    async def _identify_template_visually(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """
        Use MedGemma vision to identify document vendor and type.

        This is the fallback when text-based matching fails due to:
        - Flattened/garbled text extraction
        - Scanned documents
        - Complex layouts

        Returns:
            {
                "vendor": "LabCorp",
                "document_type": "lab_report",
                "subtype": "CD4/CD8 Ratio Profile",
                "template_id": "labcorp_cd4_cd8_v1",
                "confidence": 0.95
            }
        """
        try:
            from ....extractors.vision_extractor import VisionExtractor

            vision = VisionExtractor()
            result = await vision.identify_document_visually(pdf_path)

            if result and not result.get('error'):
                self.logger.info(
                    f"Visual identification: vendor={result.get('vendor')}, "
                    f"type={result.get('document_type')}, "
                    f"template={result.get('template_id')}"
                )
                return result
            else:
                self.logger.warning(f"Visual identification failed: {result.get('error')}")
                return None

        except Exception as e:
            self.logger.warning(f"Visual identification error: {e}")
            return None
    
    def _calculate_match_score_with_details(self, text: str, template: Dict) -> tuple[float, Dict]:
        """Calculate match score and return details for debugging."""
        score = self._calculate_match_score(text, template)
        # Get signals from internal state (stored during calculation)
        details = getattr(self, '_last_signals', {}).copy()
        details['capped'] = getattr(self, '_last_capped', False)
        return score, details

    def _calculate_match_score(self, text: str, template: Dict) -> float:
        """
        Calculate fingerprint match score using multiple signals.

        Scoring components (weighted):
        - Header/Test pattern: 25% (test type identification anywhere in doc)
        - Vendor markers: 30% (company identification) - CRITICAL for avoiding false matches
        - Required fields: 30% (presence of expected test names)
        - Layout signature: 15% (table structure, reference ranges)

        CRITICAL: If vendor score is 0 (no vendor markers found), the final score
        is capped at 0.5 to prevent false matches. This is essential because:
        - Field names (WBC, RBC, Hemoglobin) are common across ALL lab vendors
        - Layout patterns (reference ranges, flags) are also universal
        - Without vendor identification, we cannot reliably match a template

        Returns:
            Score from 0.0 to 1.0
        """
        signals = {}
        text_lower = text.lower()

        # Clean text - remove control characters for better matching
        text_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text_lower)

        # ====================================================================
        # SIGNAL 1: Header/Test Type Pattern (25%)
        # Search ENTIRE document, not just first 500 chars
        # ====================================================================
        if 'header_pattern' in template:
            pattern = template['header_pattern']
            # Search entire document for the pattern
            if re.search(pattern, text_clean, re.IGNORECASE | re.DOTALL):
                signals['header'] = 1.0
            else:
                signals['header'] = 0.0

        # ====================================================================
        # SIGNAL 2: Vendor Markers (30%) - CRITICAL
        # Must identify the specific vendor, not just generic lab content
        # ====================================================================
        if 'vendor_markers' in template:
            markers = template['vendor_markers']
            markers_found = 0

            for marker in markers:
                marker_lower = marker.lower()
                # Exact match (highest confidence)
                if marker_lower in text_clean:
                    markers_found += 1
                # Partial match for multi-word company names
                # BUT: exclude common words like "of", "the", "and" to prevent false matches
                elif len(marker_lower) > 6:
                    marker_words = marker_lower.split()
                    # Only check multi-word markers with significant words
                    significant_words = [w for w in marker_words if len(w) > 3]
                    if len(significant_words) >= 2:
                        # All significant words must appear
                        if all(word in text_clean for word in significant_words):
                            markers_found += 0.7  # Reduced partial credit

            signals['vendor'] = min(1.0, markers_found / len(markers)) if markers else 0.0

        # ====================================================================
        # SIGNAL 3: Required Fields (30%)
        # Lab test names - but many are universal across vendors
        # ====================================================================
        if 'required_fields' in template:
            fields = template['required_fields']
            fields_found = 0

            for field in fields:
                field_lower = field.lower()
                # Check for exact match or as part of a test name
                if field_lower in text_clean:
                    fields_found += 1
                # Also check for common variations (e.g., "Hemoglobin" vs "HGB")
                elif self._check_field_aliases(field_lower, text_clean):
                    fields_found += 0.9

            signals['fields'] = fields_found / len(fields) if fields else 0.0

        # ====================================================================
        # SIGNAL 4: Layout Signature (15%)
        # ====================================================================
        if 'layout_signature' in template:
            layout = template['layout_signature']
            layout_score = self._check_layout_signature(text, layout)
            signals['layout'] = layout_score

        # Calculate weighted confidence
        weights = {
            'header': 0.25,
            'vendor': 0.30,
            'fields': 0.30,
            'layout': 0.15
        }

        raw_score = self.calculate_confidence(signals, weights)

        # Store signals for debugging
        self._last_signals = signals.copy()
        self._last_capped = False

        # ====================================================================
        # CRITICAL: Cap score if vendor is not identified
        # Without vendor identification, we cannot reliably use a vendor-specific template
        # ====================================================================
        vendor_score = signals.get('vendor', 0.0)

        if vendor_score == 0.0:
            # No vendor markers found - cap at 0.5 to prevent false matches
            # This ensures documents from unknown vendors go to MedGemma extraction
            capped_score = min(raw_score, 0.50)
            self._last_capped = True
            self.logger.debug(
                f"Template {template.get('id')}: vendor=0, capping score from {raw_score:.2f} to {capped_score:.2f}"
            )
            return capped_score

        elif vendor_score < 0.3:
            # Very weak vendor signal - cap at 0.65
            capped_score = min(raw_score, 0.65)
            self._last_capped = True
            self.logger.debug(
                f"Template {template.get('id')}: weak vendor ({vendor_score:.2f}), "
                f"capping score from {raw_score:.2f} to {capped_score:.2f}"
            )
            return capped_score

        return raw_score

    def _check_field_aliases(self, field: str, text: str) -> bool:
        """Check for common field name aliases."""
        aliases = {
            'hemoglobin': ['hgb', 'hb'],
            'hematocrit': ['hct'],
            'platelets': ['plt', 'platelet count'],
            'neutrophils': ['neut', 'neutro'],
            'lymphocytes': ['lymph', 'lymphs'],
            'monocytes': ['mono', 'monos'],
            'eosinophils': ['eos', 'eosino'],
            'basophils': ['baso', 'basos'],
        }

        if field in aliases:
            return any(alias in text for alias in aliases[field])
        return False
    
    def _check_layout_signature(self, text: str, layout: Dict) -> float:
        """
        Check if document layout matches expected structure.

        Looks for:
        - Presence of reference ranges (most reliable)
        - Presence of abnormal flags
        - Lab-specific column headers
        - Tabular structure indicators
        """
        score = 0.0
        checks = 0

        # Clean text for pattern matching
        text_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)

        # Check 1: Reference ranges (MOST RELIABLE for lab reports)
        if 'has_reference_range' in layout:
            checks += 1
            # Multiple reference range patterns
            ref_patterns = [
                r'\d+\.?\d*\s*-\s*\d+\.?\d*',           # 12.0 - 15.5
                r'\d+\.?\d*\s*to\s*\d+\.?\d*',          # 12.0 to 15.5
                r'>\s*\d+\.?\d*',                        # > 59
                r'<\s*\d+\.?\d*',                        # < 100
                r'reference\s*(range|interval)',         # Reference Range/Interval header
            ]
            for pattern in ref_patterns:
                if re.search(pattern, text_clean, re.IGNORECASE):
                    score += 1
                    break

        # Check 2: Abnormal flags
        if 'has_abnormal_flags' in layout:
            checks += 1
            # Look for abnormal flags (H, L, High, Low, CRITICAL, *)
            flag_patterns = [
                r'\bHigh\b',
                r'\bLow\b',
                r'\bCRITICAL\b',
                r'\bABNORMAL\b',
                r'FLAG',  # Column header
            ]
            # Also check for standalone H or L with word boundaries
            # But be careful not to match H/L in words
            if re.search(r'(?<![A-Za-z])[HL](?![A-Za-z])', text_clean):
                score += 1
            else:
                for pattern in flag_patterns:
                    if re.search(pattern, text_clean, re.IGNORECASE):
                        score += 1
                        break

        # Check 3: Lab-specific headers/indicators
        checks += 1
        lab_indicators = [
            r'RESULT',
            r'UNITS?',
            r'x10E\d/uL',     # Lab units like x10E3/uL
            r'g/dL',
            r'mg/dL',
            r'mmol/L',
            r'mL/min',
            r'fL\b',
            r'pg\b',
        ]
        indicators_found = sum(1 for p in lab_indicators if re.search(p, text_clean, re.IGNORECASE))
        if indicators_found >= 3:
            score += 1
        elif indicators_found >= 1:
            score += 0.5

        return score / checks if checks > 0 else 0.0
    
    async def _extract_text_for_matching(self, pdf_path: Path, context: ProcessingContext = None) -> str:
        """
        Extract raw text from PDF for fingerprinting.

        Uses text already extracted during classification (stored in context.raw_text)
        or extracts fresh if not available.
        """
        # First, check if text was already extracted during classification
        if context and context.raw_text:
            self.logger.debug(f"Using pre-extracted text from context ({len(context.raw_text)} chars)")
            return context.raw_text

        # Otherwise, extract text from PDF
        from ....extractors.text_extractor import TextExtractor

        try:
            extractor = TextExtractor()
            text = extractor.extract_text(pdf_path)
            self.logger.debug(f"Extracted {len(text)} chars from PDF for template matching")
            return text
        except Exception as e:
            self.logger.warning(f"Failed to extract text from PDF: {e}")
            return ""
