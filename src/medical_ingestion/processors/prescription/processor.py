# ============================================================================
# src/medical_ingestion/processors/prescription/processor.py
# ============================================================================
"""
Prescription Processor

Extracts structured data from prescription/medication orders:
- Medication names and RxNorm codes
- Dosage and frequency
- Route of administration
- Quantity and refills
- Prescriber information
- Drug interactions
- Contraindications
"""

from typing import Dict, Any, List
import logging

from ...core.context.processing_context import ProcessingContext
from ...core.vector_store import get_vector_store
from ..base_processor import BaseProcessor
from ...medgemma.client import create_client
from ...constants.medication_db import get_medication_db, lookup_medication


class PrescriptionProcessor(BaseProcessor):
    """
    Processes prescription and medication order documents.

    Uses MedGemma to extract:
    - Medication names
    - Dosage instructions
    - Prescriber information
    - Drug interaction warnings
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.medgemma = create_client(self.config)
        self.logger = logging.getLogger(__name__)
        self._vector_store = None
        self._current_context = None  # Store context for logging during processing

    @property
    def vector_store(self):
        """Get shared vector store (singleton)."""
        if self._vector_store is None:
            self._vector_store = get_vector_store(self.config)
        return self._vector_store

    def get_name(self) -> str:
        return "PrescriptionProcessor"

    def _get_agents(self) -> List:
        """
        Prescription processor uses direct MedGemma calls rather than agents.
        Returns empty list for compatibility with base class.
        """
        return []

    def _log_step(self, step: str, details: str = None, context: ProcessingContext = None):
        """Log a processing step for agentic flow visibility."""
        if details:
            self.logger.info(f"[STEP] {step}: {details}")
        else:
            self.logger.info(f"[STEP] {step}")

        # Use passed context or fall back to stored context
        ctx = context or self._current_context

        # Also log to context if available for UI visibility
        if ctx is not None:
            ctx.log_processing_step(
                step_name=step,
                status="completed",
                details={"info": details} if details else None
            )

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Process prescription document.

        Args:
            context: Processing context with raw text

        Returns:
            Updated context with extracted prescription data
        """
        # Store context for logging during processing
        self._current_context = context
        self._log_step("Starting prescription processing", str(context.document_path))

        # Extract text if not already available
        if not context.raw_text:
            self._log_step("Extracting text from document")
            from ...extractors.text_extractor import TextExtractor
            text_extractor = TextExtractor()
            context.raw_text = text_extractor.extract_text(context.document_path)
            self._log_step("Text extraction complete", f"{len(context.raw_text)} characters")

        # Extract medication information
        # Log first 500 chars of raw text for debugging
        self._log_step("MedGemma extraction", f"Processing {len(context.raw_text)} chars")
        self.logger.debug(f"OCR text sample: {context.raw_text[:500]}")
        medications = await self._extract_medications(context)
        context.sections['medications'] = medications
        self._log_step("Medication extraction complete", f"{len(medications)} medications found")

        # Extract prescriber information
        self._log_step("Extracting prescriber info")
        prescriber = self._extract_prescriber_info(context.raw_text)
        context.sections['prescriber'] = prescriber

        # Extract patient information
        self._log_step("Extracting patient info")
        patient = self._extract_patient_info(context.raw_text)
        context.sections['patient'] = patient

        # Extract prescription date
        prescription_date = self._extract_prescription_date(context.raw_text)
        context.sections['prescription_date'] = prescription_date

        # Check for drug interactions
        interactions = []
        if medications:
            self._log_step("Drug interaction check", "Checking for interactions")
            interactions = await self._check_drug_interactions(medications)
            context.sections['drug_interactions'] = interactions
            if interactions:
                self._log_step("Drug interactions found", f"{len(interactions)} potential interactions")
        else:
            context.sections['drug_interactions'] = []

        # Check for contraindications
        self._log_step("Contraindication check")
        contraindications = await self._check_contraindications(context)
        context.sections['contraindications'] = contraindications

        # Calculate overall confidence based on medication validation
        confidence = self._calculate_extraction_confidence(medications)

        # Store in vector store for future similar document matching (only at high confidence)
        if self.config.get('use_consensus_extraction', False):
            min_confidence = self.config.get('vector_store_min_confidence', 0.85)
            if confidence >= min_confidence:
                await self._store_in_vector_store(context, medications, confidence)
            else:
                self.logger.debug(
                    f"Skipping vector store: confidence {confidence:.2f} < threshold {min_confidence}"
                )

        self.logger.info(f"Prescription processing complete - {len(medications)} medications, confidence: {confidence:.2f}")

        # Clear stored context
        self._current_context = None

        return {
            "success": True,
            "agent_results": [],
            "extracted_values": len(medications),
            "requires_review": bool(interactions or contraindications),
            "confidence": confidence
        }

    def _calculate_extraction_confidence(self, medications: list) -> float:
        """
        Calculate overall extraction confidence based on validated medications.

        Args:
            medications: List of validated medication dicts

        Returns:
            Overall confidence score (0-1)
        """
        if not medications:
            return 0.3  # Low confidence if no medications found

        # Average the individual medication confidences
        confidences = [
            med.get('extraction_confidence', 0.5)
            for med in medications
        ]
        avg_confidence = sum(confidences) / len(confidences)

        # Boost confidence if we found multiple medications (more reliable)
        if len(medications) >= 3:
            avg_confidence = min(1.0, avg_confidence + 0.05)
        elif len(medications) >= 2:
            avg_confidence = min(1.0, avg_confidence + 0.03)

        return round(avg_confidence, 2)

    def _find_medication_candidates(self, text: str) -> list:
        """
        Pre-filter to find lines that look like medication entries.

        Identifies lines with patterns like:
        - "Akendd Z5my 30" (drug name + dosage + quantity)
        - "Lisinopril 10mg" (drug name + dosage)
        - "Amoxicillin 500mg Cap #21" (with # quantity)
        - "TAB. ABCIXIMAB ... 8 Days (Tot:8 Tab)" (international format)

        Args:
            text: OCR text to scan

        Returns:
            List of candidate medication lines
        """
        import re

        candidates = []
        lines = text.split('\n')

        # Pattern: word followed by number+unit pattern (mg, my, ml, mcg, g, %)
        # Also matches OCR errors like "Z5my" for "25mg", "l0mg" for "10mg"
        # The drug name MUST end with a letter (not a space before dosage)
        # Enhanced quantity patterns: #30, x30, ×30, Qty:30, (30), etc.
        med_pattern = re.compile(
            r'^([A-Za-z][A-Za-z-]{1,29}[A-Za-z])\s+'  # Drug name (no trailing space in name)
            r'([ZzOo\d][\dOoZz.]*\s*(?:mg|my|ml|mcg|g|%|MG|MY|ML|MCG|G))'  # Dosage with OCR errors
            r'(?:\s*(?:cap|tab|capsule|tablet)s?)?'  # Optional form
            r'(?:\s*[x×#÷]?\s*(\d+))?'  # Optional quantity (including ÷ OCR error for x)
            r'(?:\s*\((\d+)\))?',  # Optional quantity in parentheses
            re.IGNORECASE
        )

        # Also check for "Rx" or "R" followed by medication info
        rx_pattern = re.compile(
            r'^R[x]?\s*[:\s]*([A-Za-z][A-Za-z-]{1,29}[A-Za-z])\s+'
            r'([ZzOo\d][\dOoZz.]*\s*(?:mg|my|ml|mcg|g|%|MG|MY|ML|MCG|G))'
            r'(?:\s*(?:cap|tab|capsule|tablet)s?)?'  # Optional form
            r'(?:\s*[x×#]?\s*(\d+))?',  # Optional quantity
            re.IGNORECASE
        )

        # International format: "1) TAB. DRUGNAME" or "1) CAP. DRUGNAME 500"
        # Matches formats like "TAB. ABCIXIMAB", "CAP. ZOCLAR 500", "SYR. AMOXYCILLIN 250 MG"
        intl_pattern = re.compile(
            r'^(?:\d+\)?\s*)?'  # Optional number prefix "1)" or "1"
            r'(?:TAB\.?|CAP\.?|SYR\.?|INJ\.?|SUSP\.?|DROPS?\.?|CREAM\.?|GEL\.?|OINT\.?)\s+'  # Form prefix
            r'([A-Za-z][A-Za-z0-9-]{1,29})'  # Drug name (may include numbers like B12)
            r'(?:\s+(\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|%|MG|ML|MCG|G)?)\s*)?'  # Optional strength
            r'(?:\s*/\s*SR)?',  # Optional SR (sustained release) suffix
            re.IGNORECASE
        )

        # Pattern for total quantity in international format: "(Tot:8 Tab)" or "(Tot: 16 Tab)"
        tot_pattern = re.compile(r'\(Tot[:\s]*(\d+)\s*(?:Tab|Cap|Tabs|Caps)?\)', re.IGNORECASE)

        # Pattern for duration: "8 Days", "3 Days"
        duration_pattern = re.compile(r'(\d+)\s*(?:Days?|Weeks?|Months?)', re.IGNORECASE)

        # Pattern for frequency: "1 Morning", "1 Night", "1 Morning, 1 Night", "After Food"
        frequency_pattern = re.compile(
            r'(\d+\s*(?:Morning|Night|Evening|Noon|Afternoon|Bedtime)'
            r'(?:\s*,\s*\d+\s*(?:Morning|Night|Evening|Noon|Afternoon|Bedtime))?)'
            r'(?:\s*\(([^)]+)\))?',  # Optional instructions like "(After Food)"
            re.IGNORECASE
        )

        # Pattern for handwritten prescriptions with OCR errors:
        # "amoxirillin Joong Cap#21" -> drug name + corrupted dosage + form + quantity
        # Matches: DrugName + any_text + (Cap|Tab|Capsule|Tablet) + #quantity
        handwritten_pattern = re.compile(
            r'^([A-Za-z][A-Za-z]{2,29})\s+'  # Drug name (at least 3 letters)
            r'[A-Za-z0-9\s]*'  # Any intermediate text (potentially corrupted dosage)
            r'(?:cap|tab|capsule|tablet)s?\s*'  # Form
            r'[#x×]?\s*(\d+)',  # Quantity with optional # or x prefix
            re.IGNORECASE
        )

        # Track line indices for multi-line lookups
        processed_indices = set()

        for idx, line in enumerate(lines):
            if idx in processed_indices:
                continue

            line = line.strip()
            if not line or len(line) < 5:
                continue

            # Skip obvious non-medication lines
            skip_patterns = [
                r'^(physician|provider|patient|address|date|phone|fax|dea|npi)',
                r'^(please|see|refill|staff|clinic|center|building|street)',
                r'^\d{1,2}[./]\d{1,2}[./]\d{2,4}$',  # Dates
                r'^\d{3}[-.]?\d{3}[-.]?\d{4}$',  # Phone numbers
                r'^(chief|complaint|diagnosis|advice|follow)',  # Medical form headers
                r'^(weight|height|bp|b\.?m\.?i)',  # Vital signs
            ]
            if any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
                continue

            # Check for medication patterns
            match = med_pattern.match(line)
            if match:
                quantity = match.group(3) or match.group(4) or ''
                candidates.append({
                    'line': line,
                    'name': match.group(1).strip(),
                    'dosage': match.group(2).strip(),
                    'quantity': quantity,
                    'frequency': '',
                    'duration': '',
                    'instructions': ''
                })
                continue

            match = rx_pattern.match(line)
            if match:
                candidates.append({
                    'line': line,
                    'name': match.group(1).strip(),
                    'dosage': match.group(2).strip(),
                    'quantity': match.group(3) if match.group(3) else '',
                    'frequency': '',
                    'duration': '',
                    'instructions': ''
                })
                continue

            # Check for handwritten prescription format (OCR errors in dosage)
            # E.g., "amoxirillin Joong Cap#21" where "Joong" is corrupted "500mg"
            match = handwritten_pattern.match(line)
            if match:
                drug_name = match.group(1).strip()
                quantity = match.group(2) if match.group(2) else ''

                # Try to extract dosage from corrupted text (look for patterns like "500" or "Joong")
                # Also look at next few lines for Sig instructions
                dosage = ''
                frequency = ''
                duration = ''
                instructions = ''

                # Look for a number that could be dosage in the line (before Cap/Tab)
                dosage_search = re.search(r'(\d+)\s*(?:mg|my|ml|mcg|g)?', line, re.IGNORECASE)
                if dosage_search:
                    # Make sure it's not the quantity
                    if dosage_search.group(1) != quantity:
                        dosage = dosage_search.group(1) + 'mg'

                # Look ahead for Sig instructions on next few lines
                for look_idx in range(1, 4):
                    if idx + look_idx < len(lines):
                        next_line = lines[idx + look_idx].strip()
                        if next_line.lower().startswith('sig'):
                            instructions = next_line
                            processed_indices.add(idx + look_idx)
                            # Parse frequency from Sig line
                            # Handle OCR variations: "I cap 3x aday" or "1 cap 3x a day"
                            freq_patterns = [
                                r'(\d+)\s*x\s*a?\s*day',  # "3x aday", "3x a day", "3xday"
                                r'(\d+)\s*times?\s*(?:a\s*)?day',  # "3 times a day"
                                r'(?:cap|tab)s?\s*(\d+)\s*x',  # "cap 3x"
                            ]
                            for fp in freq_patterns:
                                freq_search = re.search(fp, next_line, re.IGNORECASE)
                                if freq_search:
                                    times = freq_search.group(1)
                                    frequency = f"{times}x daily"
                                    break
                            break

                candidates.append({
                    'line': line,
                    'name': drug_name,
                    'dosage': dosage,
                    'quantity': quantity,
                    'frequency': frequency,
                    'duration': duration,
                    'instructions': instructions
                })
                continue

            # Check for international prescription format
            match = intl_pattern.match(line)
            if match:
                drug_name = match.group(1).strip()
                strength = match.group(2).strip() if match.group(2) else ''
                # Add "mg" suffix if strength is just a number
                if strength and not any(strength.lower().endswith(u) for u in ['mg', 'ml', 'mcg', 'g', '%']):
                    strength = strength + 'mg'

                # For multi-line table format (e.g., ColorRx), look at next few lines
                # to extract frequency, duration, and quantity
                # Combine current line with next 4 lines for pattern matching
                lookahead_text = line
                for look_idx in range(1, 5):
                    if idx + look_idx < len(lines):
                        next_line = lines[idx + look_idx].strip()
                        if next_line:
                            # Stop if we hit another medication line
                            if intl_pattern.match(next_line):
                                break
                            lookahead_text += ' ' + next_line
                            processed_indices.add(idx + look_idx)

                # Extract additional fields from combined text
                quantity = ''
                tot_match = tot_pattern.search(lookahead_text)
                if tot_match:
                    quantity = tot_match.group(1)

                duration = ''
                dur_match = duration_pattern.search(lookahead_text)
                if dur_match:
                    duration = dur_match.group(0)

                frequency = ''
                instructions = ''
                freq_match = frequency_pattern.search(lookahead_text)
                if freq_match:
                    frequency = freq_match.group(1).strip()
                    if freq_match.group(2):
                        instructions = freq_match.group(2).strip()

                # Also check for "(After Food)" or similar instructions on separate line
                if not instructions:
                    instr_pattern = re.compile(r'\(([^)]*(?:Food|Meal|Empty|Stomach)[^)]*)\)', re.IGNORECASE)
                    instr_match = instr_pattern.search(lookahead_text)
                    if instr_match:
                        instructions = instr_match.group(1).strip()

                candidates.append({
                    'line': line,
                    'name': drug_name,
                    'dosage': strength,
                    'quantity': quantity,
                    'frequency': frequency,
                    'duration': duration,
                    'instructions': instructions
                })

        return candidates

    async def _extract_medications(
        self,
        context: ProcessingContext
    ) -> list:
        """
        Extract medication list from prescription.

        Args:
            context: Processing context

        Returns:
            List of medication dicts
        """
        import json
        import re
        from json_repair import repair_json

        # STEP 1: Pre-filter to find medication candidates using regex
        candidates = self._find_medication_candidates(context.raw_text)
        self._log_step("Pre-filter candidates", f"Found {len(candidates)} potential medications")

        for c in candidates:
            self.logger.debug(f"Candidate: {c}")

        # If we found clear candidates, use them directly without MedGemma
        if candidates:
            self._log_step("Using regex-extracted medications", f"{len(candidates)} found")
            medications = []
            for c in candidates:
                medications.append({
                    'medication_name': c['name'],
                    'strength': c.get('dosage', ''),
                    'route': '',
                    'frequency': c.get('frequency', ''),
                    'quantity': c.get('quantity', ''),
                    'duration': c.get('duration', ''),
                    'refills': '',
                    'instructions': c.get('instructions', '')
                })

            # Try to find Sig instructions (for prescriptions that use "Sig:" format)
            sig_match = re.search(r'Sig[:\s]+(.+?)(?:\n|$)', context.raw_text, re.IGNORECASE)
            if sig_match:
                sig_text = sig_match.group(1).strip()
                # Apply to first medication if it doesn't already have instructions
                if medications and not medications[0].get('instructions'):
                    medications[0]['instructions'] = sig_text
                    # Also try to extract frequency from Sig
                    freq_patterns = [
                        (r'(\d+)\s*(?:times?|x)\s*(?:a|per)?\s*day', r'\1 times daily'),
                        (r'once\s*(?:a|per)?\s*day', 'once daily'),
                        (r'twice\s*(?:a|per)?\s*day', 'twice daily'),
                        (r'every\s*(\d+)\s*hours?', r'every \1 hours'),
                        (r'(?:at\s*)?bedtime', 'at bedtime'),
                        (r'(?:in the\s*)?morning', 'in the morning'),
                        (r'(?:in the\s*)?evening', 'in the evening'),
                        (r'with\s*(?:food|meals?)', 'with food'),
                        (r'before\s*(?:food|meals?)', 'before food'),
                        (r'after\s*(?:food|meals?)', 'after food'),
                    ]
                    for pattern, replacement in freq_patterns:
                        freq_match = re.search(pattern, sig_text, re.IGNORECASE)
                        if freq_match and not medications[0].get('frequency'):
                            if '\\1' in replacement:
                                medications[0]['frequency'] = re.sub(pattern, replacement, freq_match.group(0), flags=re.IGNORECASE)
                            else:
                                medications[0]['frequency'] = replacement
                            break

            # Look for refill info
            refill_patterns = [
                r'(?:may be )?refilled?\s*(\d+)?\s*times?',
                r'refills?[:\s]*(\d+)',
                r'(\d+)\s*refills?',
            ]
            for pattern in refill_patterns:
                refill_match = re.search(pattern, context.raw_text, re.IGNORECASE)
                if refill_match:
                    if medications:
                        medications[0]['refills'] = refill_match.group(1) if refill_match.group(1) else 'Yes'
                    break

            # Look for quantity if not already found (e.g., "Qty: 30", "Dispense: 30")
            if medications and not medications[0].get('quantity'):
                qty_patterns = [
                    r'(?:qty|quantity|disp(?:ense)?)[:\s]*#?(\d+)',
                    r'#(\d+)\s*(?:tabs?|caps?|tablets?|capsules?)?',
                ]
                for pattern in qty_patterns:
                    qty_match = re.search(pattern, context.raw_text, re.IGNORECASE)
                    if qty_match:
                        medications[0]['quantity'] = qty_match.group(1)
                        break

            # Skip MedGemma if we have clear candidates, go directly to validation
            self._log_step("Skipping MedGemma", "Clear medication patterns found")

        else:
            # STEP 2: Fall back to MedGemma for ambiguous cases
            self._log_step("No clear patterns", "Using MedGemma extraction")

            prompt = f"""You are a prescription OCR text parser. Extract medications ONLY from the text below.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. ONLY extract medications that are EXPLICITLY written in the text below
2. DO NOT invent, guess, or hallucinate medications not present in the text
3. OCR text may have errors - extract the medication name AS WRITTEN (e.g., "Atendld" not "Atenolol")
4. If you cannot find any clear medication names in the text, return []
5. Common OCR errors: l/1, 0/O, rn/m - output the text AS-IS, do not correct

PRESCRIPTION TEXT (OCR output - may contain errors):
---
{context.raw_text[:2000]}
---

Extract ONLY what appears in the text above. Return a JSON array with these fields (use "" if not found):
- medication_name: The drug name AS WRITTEN in the text (preserve OCR errors)
- strength: The strength/dosage as written (e.g., "25my" not "25mg")
- route: How taken - only if explicitly stated in text
- frequency: How often - only if explicitly stated in text
- quantity: Number dispensed - only if explicitly stated
- refills: Number of refills - only if explicitly stated
- instructions: Sig/directions as written in text

IMPORTANT: If you don't see a clear medication in the text, return []. Do NOT make up medications.

JSON array:"""

            try:
                response = await self.medgemma.generate(
                    prompt=prompt,
                    max_tokens=1000,
                    temperature=0.0  # Deterministic for structured output
                )

                response_text = response.get('text', '').strip()

                # Try to extract JSON from response
                medications = self._parse_json_response(response_text)

                if medications is None:
                    self.logger.warning(f"Failed to parse medication JSON, trying repair")
                    # Try to repair malformed JSON
                    try:
                        repaired = repair_json(response_text)
                        medications = json.loads(repaired)
                    except Exception:
                        medications = []

            except Exception as e:
                self.logger.error(f"MedGemma extraction failed: {e}")
                medications = []

        # STEP 3: Validate and clean results (for both regex and MedGemma paths)
        if isinstance(medications, list) and medications:
            # First pass: collect all medication names for context
            # Also check for hallucinations (medications not in source text)
            all_med_names = []
            med_entries = []
            source_text_lower = context.raw_text.lower()

            for med in medications:
                if isinstance(med, dict):
                    name = med.get('medication_name', '').strip()
                    if name and not self._is_garbage_text(name):
                        # HALLUCINATION CHECK: Verify medication appears in source text
                        # Check if name or first 4 chars appear in source
                        name_lower = name.lower()
                        name_prefix = name_lower[:4] if len(name_lower) >= 4 else name_lower

                        if name_lower not in source_text_lower and name_prefix not in source_text_lower:
                            # Medication not found in source - likely hallucination
                            self.logger.warning(
                                f"HALLUCINATION DETECTED: '{name}' not found in OCR text. Skipping."
                            )
                            self._log_step(
                                "Hallucination rejected",
                                f"'{name}' not in source text"
                            )
                            continue

                        all_med_names.append(name)
                        med_entries.append({
                            'medication_name': name,
                            'strength': med.get('strength', ''),
                            'route': med.get('route', ''),
                            'frequency': med.get('frequency', ''),
                            'quantity': med.get('quantity', ''),
                            'duration': med.get('duration', ''),
                            'refills': med.get('refills', ''),
                            'instructions': med.get('instructions', '')
                        })

            # Second pass: validate each medication with context from others
            self._log_step("Validating medications", f"{len(med_entries)} candidates")
            cleaned = []
            for idx, med_entry in enumerate(med_entries):
                name = med_entry['medication_name']
                strength = med_entry.get('strength', '')

                self._log_step(
                    f"Validating medication {idx+1}/{len(med_entries)}",
                    f"'{name}' {strength}"
                )

                # Get context medications (all others on the prescription)
                context_meds = [m for m in all_med_names if m != name]

                # Validate medication entry with context and strength
                self._log_step("Checking RxNorm database", f"'{name}'")
                is_valid, confidence, rxnorm_info = self._validate_medication_entry(
                    med_entry,
                    context_medications=context_meds
                )

                if is_valid:
                    med_entry['extraction_confidence'] = round(confidence, 2)

                    # Add RxNorm info if available from database
                    if rxnorm_info:
                        med_entry['rxcui'] = rxnorm_info.get('rxcui')
                        med_entry['rxnorm_name'] = rxnorm_info.get('name')
                        med_entry['rxnorm_match_type'] = rxnorm_info.get('match_type', 'exact')

                        # Add strength validation info
                        if rxnorm_info.get('strength_validated'):
                            med_entry['strength_validated'] = True
                            self._log_step("Strength validated", f"'{strength}' matches known dosages")

                        # Add drug class info from context
                        if rxnorm_info.get('drug_class'):
                            med_entry['drug_class'] = rxnorm_info.get('drug_class')
                        if rxnorm_info.get('context_match'):
                            med_entry['context_match'] = True
                            self._log_step(
                                "Context match",
                                f"Drug class '{rxnorm_info.get('drug_class')}' matches other Rx medications"
                            )

                            # Add OCR correction info if present
                            if rxnorm_info.get('needs_review'):
                                self._log_step(
                                    "OCR correction detected",
                                    f"'{name}' -> '{rxnorm_info.get('name')}' (distance: {rxnorm_info.get('edit_distance')})"
                                )
                                # Use MedGemma to validate OCR correction
                                self._log_step("Validating OCR correction with MedGemma")
                                medgemma_result = await self._validate_ocr_with_medgemma(
                                    ocr_text=name,
                                    extracted_strength=med_entry.get('strength'),
                                    extracted_frequency=med_entry.get('frequency'),
                                    db_suggestion=rxnorm_info.get('name')
                                )

                                if medgemma_result:
                                    mg_name = medgemma_result.get('corrected_name')
                                    mg_confidence = medgemma_result.get('confidence', 0)

                                    # MedGemma agrees with database
                                    if mg_name and mg_name.lower() == rxnorm_info['name'].lower():
                                        confidence += 0.15  # Boost for LLM agreement
                                        med_entry['medgemma_validated'] = True
                                        med_entry['needs_review'] = False  # No longer needs review
                                        med_entry['validation_status'] = 'ocr_corrected'
                                        self._log_step(
                                            "MedGemma confirmed OCR correction",
                                            f"'{name}' -> '{mg_name}' ✓"
                                        )
                                    # MedGemma suggests different correction
                                    elif mg_name and mg_confidence > 0.7:
                                        # MedGemma is confident in a different answer
                                        med_entry['medgemma_suggestion'] = mg_name
                                        med_entry['medgemma_confidence'] = mg_confidence
                                        med_entry['needs_review'] = True
                                        med_entry['validation_status'] = 'ocr_corrected'
                                        self._log_step(
                                            "MedGemma disagrees with DB",
                                            f"DB: '{rxnorm_info['name']}' vs MedGemma: '{mg_name}' - needs review"
                                        )
                                    else:
                                        med_entry['validation_status'] = 'ocr_corrected'

                                    # Add drug class from MedGemma if available
                                    if medgemma_result.get('drug_class'):
                                        med_entry['drug_class'] = medgemma_result.get('drug_class')

                                    # Validate strength/frequency with MedGemma's knowledge
                                    if medgemma_result.get('strength_valid') is False:
                                        med_entry['strength_warning'] = 'MedGemma flagged unusual strength'
                                    if medgemma_result.get('frequency_valid') is False:
                                        med_entry['frequency_warning'] = 'MedGemma flagged unusual frequency'
                                else:
                                    med_entry['needs_review'] = True
                                    med_entry['validation_status'] = 'ocr_corrected'

                                med_entry['ocr_warning'] = rxnorm_info.get('ocr_warning', '')
                                med_entry['original_ocr_text'] = rxnorm_info.get('original_input', name)
                                med_entry['edit_distance'] = rxnorm_info.get('edit_distance')
                                # Update confidence after MedGemma validation
                                med_entry['extraction_confidence'] = round(min(confidence, 1.0), 2)
                                self.logger.info(
                                    f"OCR-corrected medication: {name} -> {rxnorm_info['name']} "
                                    f"(RXCUI:{rxnorm_info['rxcui']}, confidence: {confidence:.2f}, "
                                    f"review: {med_entry.get('needs_review', True)})"
                                )
                            else:
                                med_entry['validation_status'] = 'verified'
                                self._log_step(
                                    f"RxNorm match ({rxnorm_info.get('match_type', 'exact')})",
                                    f"'{name}' -> RXCUI:{rxnorm_info['rxcui']} (conf: {confidence:.2f})"
                                )
                        else:
                            # No database match - use MedGemma to validate/correct
                            self._log_step("No RxNorm match", f"'{name}' - trying MedGemma")
                            medgemma_result = await self._validate_ocr_with_medgemma(
                                ocr_text=name,
                                extracted_strength=med_entry.get('strength'),
                                extracted_frequency=med_entry.get('frequency'),
                                db_suggestion=None
                            )

                            if medgemma_result:
                                mg_name = medgemma_result.get('corrected_name')
                                mg_confidence = medgemma_result.get('confidence', 0)

                                if mg_name and mg_confidence >= 0.7:
                                    # MedGemma identified the medication
                                    med_entry['medgemma_corrected'] = True
                                    med_entry['medgemma_name'] = mg_name
                                    med_entry['medgemma_confidence'] = mg_confidence
                                    confidence = 0.5 + (mg_confidence * 0.3)  # 0.5-0.8 range
                                    med_entry['extraction_confidence'] = round(confidence, 2)

                                    self._log_step(
                                        "MedGemma identified medication",
                                        f"'{name}' -> '{mg_name}' (conf: {mg_confidence})"
                                    )

                                    # Re-lookup in database with corrected name
                                    self._log_step("Re-checking RxNorm with corrected name")
                                    from ...constants.medication_db import lookup_medication
                                    corrected_result = lookup_medication(mg_name, ocr_correction=False)
                                    if corrected_result:
                                        med_entry['rxcui'] = corrected_result.get('rxcui')
                                        med_entry['rxnorm_name'] = corrected_result.get('name')
                                        med_entry['rxnorm_match_type'] = 'medgemma_corrected'
                                        med_entry['validation_status'] = 'medgemma_verified'
                                        confidence += 0.10  # Bonus for database confirmation
                                        med_entry['extraction_confidence'] = round(min(confidence, 1.0), 2)
                                        self._log_step(
                                            "RxNorm confirmed MedGemma correction",
                                            f"RXCUI:{corrected_result.get('rxcui')}"
                                        )
                                    else:
                                        med_entry['validation_status'] = 'medgemma_verified'

                                    if medgemma_result.get('drug_class'):
                                        med_entry['drug_class'] = medgemma_result.get('drug_class')

                                    med_entry['needs_review'] = mg_confidence < 0.85
                                else:
                                    # MedGemma couldn't identify it either
                                    med_entry['validation_status'] = 'unverified'
                                    med_entry['needs_review'] = True
                                    self._log_step(
                                        "UNVERIFIED - Heuristic only",
                                        f"'{name}' (conf: {confidence:.2f}) - no DB/MedGemma match"
                                    )
                            else:
                                # MedGemma unavailable or failed
                                med_entry['validation_status'] = 'unverified'
                                med_entry['needs_review'] = True
                                self._log_step(
                                    "UNVERIFIED - Heuristic only",
                                    f"'{name}' (conf: {confidence:.2f})"
                                )

                    self._log_step("Medication validated", f"'{name}' ✓ (conf: {med_entry.get('extraction_confidence', confidence):.2f}, status: {med_entry.get('validation_status', 'unknown')})")
                    cleaned.append(med_entry)
                else:
                    self._log_step("Rejected", f"'{name}' - not a valid medication")

            # Summary with validation status breakdown
            status_counts = {}
            for m in cleaned:
                vs = m.get('validation_status', 'unknown')
                status_counts[vs] = status_counts.get(vs, 0) + 1
            needs_review = sum(1 for m in cleaned if m.get('needs_review'))

            status_str = " | ".join(f"{count} {status}" for status, count in sorted(status_counts.items()))
            self._log_step(
                "Medication extraction summary",
                f"{len(cleaned)}/{len(medications)} accepted | "
                f"{status_str} | "
                f"{needs_review} need review"
            )
            return cleaned

        return []

    def _parse_json_response(self, text: str) -> list | None:
        """Parse JSON from LLM response, handling common issues."""
        import json

        # Clean up the response
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith('```json'):
            text = text[7:]
        elif text.startswith('```'):
            text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()

        # Try to find JSON array in the text
        start_idx = text.find('[')
        end_idx = text.rfind(']')

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Try parsing the whole text
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

        return None

    async def _validate_ocr_with_medgemma(
        self,
        ocr_text: str,
        extracted_strength: str = None,
        extracted_frequency: str = None,
        db_suggestion: str = None
    ) -> dict | None:
        """
        Use MedGemma to validate and correct OCR-corrupted medication data.

        MedGemma's medical knowledge can recognize medication names even when
        OCR produces errors that the database fuzzy matching can't resolve.

        Args:
            ocr_text: The OCR-extracted medication name (possibly corrupted)
            extracted_strength: Strength if extracted (helps identify the drug)
            extracted_frequency: Frequency if extracted (helps identify the drug)
            db_suggestion: Database's best guess (for validation)

        Returns:
            Dict with corrected data or None if MedGemma can't identify it
        """
        # Build context from available data
        context_parts = [f"Medication name from OCR: '{ocr_text}'"]
        if extracted_strength:
            context_parts.append(f"Strength: {extracted_strength}")
        if extracted_frequency:
            context_parts.append(f"Frequency: {extracted_frequency}")
        if db_suggestion:
            context_parts.append(f"Database suggested: '{db_suggestion}'")

        context_str = "\n".join(context_parts)

        prompt = f"""You are a pharmacist reviewing prescription OCR output. The OCR may have misread the medication name.

{context_str}

Based on your medical knowledge:
1. What is the CORRECT medication name? Consider common OCR errors (l/t, rn/m, n/u confusion)
2. Is the strength appropriate for this medication?
3. Is the frequency appropriate?

Return ONLY valid JSON (no explanation):
{{
    "corrected_name": "the correct medication name",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "strength_valid": true/false,
    "frequency_valid": true/false,
    "drug_class": "e.g., beta_blocker, statin, etc."
}}

If you cannot identify the medication with reasonable confidence, return:
{{"corrected_name": null, "confidence": 0.0, "reasoning": "cannot identify"}}

JSON response:"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.0
            )

            response_text = response.get('text', '').strip()

            # Parse JSON response
            import json
            from json_repair import repair_json

            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                try:
                    result = json.loads(repair_json(response_text))
                except Exception:
                    return None

            if not isinstance(result, dict):
                return None

            corrected_name = result.get('corrected_name')
            confidence = result.get('confidence', 0.0)

            if corrected_name and confidence >= 0.6:
                self.logger.info(
                    f"MedGemma OCR correction: '{ocr_text}' -> '{corrected_name}' "
                    f"(confidence: {confidence}, reason: {result.get('reasoning', 'N/A')})"
                )
                return result

            return None

        except Exception as e:
            self.logger.debug(f"MedGemma OCR validation failed: {e}")
            return None

    def _is_garbage_text(self, text: str) -> bool:
        """Check if text is garbage/thinking output from LLM."""
        garbage_indicators = [
            'thought', 'process', 'analyze', 'understand',
            'goal', 'request', 'looking', 'search',
            '**', '__', 'step', 'scan'
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in garbage_indicators)

    def _is_likely_medication(
        self,
        name: str,
        extracted_strength: str = None,
        context_medications: list = None
    ) -> tuple[bool, dict | None]:
        """
        Validate if a string is likely a medication name.

        Uses RxNorm database for validation (primary), with heuristic fallback.
        Enhanced with dosage validation and context scoring.

        Args:
            name: Potential medication name
            extracted_strength: Strength from prescription for dosage validation
            context_medications: Other medications on prescription for context scoring

        Returns:
            Tuple of (is_valid, rxnorm_info) where rxnorm_info contains rxcui if found
        """
        import re

        if not name or len(name) < 3:
            return False, None

        name_lower = name.lower().strip()
        name_clean = re.sub(r'[^\w\s-]', '', name_lower)

        # Reject if too long (probably a sentence or description)
        if len(name) > 50:
            return False, None

        # Reject obvious non-medication patterns first
        non_medication_patterns = [
            r'^(patient|doctor|dr\.|physician|prescriber|pharmacy|pharmacist)$',
            r'^(name|address|phone|fax|date|signature|license|npi|dea)$',
            r'^(refill|quantity|qty|directions|sig|instructions)$',
            r'^(street|avenue|ave|road|rd|boulevard|blvd|suite|ste)$',
            r'^(city|state|zip|zipcode)$',
            r'^\d+$',
            r'^\d{3}[-.]?\d{3}[-.]?\d{4}$',
            r'^\d{5}(-\d{4})?$',
            r'^(the|and|for|with|take|use|apply|as|needed|prn)$',
            r'^(none|n/a|na|not applicable|unknown)$',
            r'^(mr|mrs|ms|dr|md|do|rn|np|pa)\.?$',
        ]

        for pattern in non_medication_patterns:
            if re.match(pattern, name_clean, re.IGNORECASE):
                return False, None

        # PRIMARY: Check RxNorm database with context
        med_db = get_medication_db()
        if med_db.is_available:
            # Clean the name for database lookup (remove dosage forms)
            lookup_name = self._extract_drug_name(name)
            db_result = lookup_medication(
                lookup_name,
                extracted_strength=extracted_strength,
                context_medications=context_medications
            )

            if db_result:
                self.logger.debug(
                    f"RxNorm match: '{name}' -> {db_result['name']} "
                    f"(RXCUI: {db_result['rxcui']}, match: {db_result.get('match_type', 'exact')})"
                )
                return True, db_result

            # If no match, also try the original name
            if lookup_name != name:
                db_result = lookup_medication(
                    name,
                    extracted_strength=extracted_strength,
                    context_medications=context_medications
                )
                if db_result:
                    return True, db_result

        # FALLBACK: Heuristic validation
        # Be STRICT if regional database is available - reject if no match
        # Fall back to heuristics only when:
        # 1. Database is not available (file doesn't exist for this region)
        # 2. Database is available but this might be a regional brand name

        if med_db.is_available:
            # Database is available but no match found
            # Check if heuristics pass - if so, it might be a valid international brand name
            is_heuristic_valid = self._heuristic_medication_check(name, name_lower, name_clean)

            if is_heuristic_valid:
                # Medication passes heuristics - might be valid but not in this region's database
                from ...constants.medication_db import get_medication_region
                region = get_medication_region()
                self.logger.info(
                    f"Heuristic match: '{name}' not found in {region.upper()} database but passes "
                    "heuristic validation (may be regional brand name)"
                )
                return True, None
            else:
                # Fails both database and heuristics - likely garbage
                self.logger.warning(
                    f"Rejected '{name}' - not found in database and failed heuristic validation"
                )
                return False, None

        # Database completely unavailable - use heuristics only
        return self._heuristic_medication_check(name, name_lower, name_clean), None

    def _extract_drug_name(self, text: str) -> str:
        """
        Extract the drug name from text that may include strength/form.

        Handles international prescription formats (especially Indian):
            "TAB. ABCIXIMAB 1 Morning" -> "ABCIXIMAB"
            "CAP. OMEPRAZOLE 20mg" -> "OMEPRAZOLE"
            "INJ. INSULIN 10 units" -> "INSULIN"
            "Metformin 500mg" -> "Metformin"
            "Lisinopril 10 mg tablets" -> "Lisinopril"
            "Advair Diskus 250/50" -> "Advair Diskus"
        """
        import re

        cleaned = text.strip()

        # Strip leading dosage form prefixes (Indian Rx format: "TAB.", "CAP.", "INJ.", etc.)
        cleaned = re.sub(
            r'^(TAB|CAP|CAPS|INJ|SYR|SUSP|OINT|GEL|DRP|DROPS|CR|LOT|LOTION|INHALER|PATCH|SUPP)\.?\s+',
            '', cleaned, flags=re.IGNORECASE
        )

        # Remove trailing schedule/timing words
        cleaned = re.sub(
            r'\s+\d*\s*(morning|evening|night|bedtime|afternoon|before\s+food|after\s+food|empty\s+stomach|sos|stat|prn)$',
            '', cleaned, flags=re.IGNORECASE
        )

        # Remove strength patterns (number + unit at end)
        cleaned = re.sub(r'\s+\d+(\.\d+)?\s*(mg|mcg|g|ml|units?|%|/\d+).*$', '', cleaned, flags=re.IGNORECASE)

        # Remove trailing bare numbers (e.g., "ABCIXIMAB 1" -> "ABCIXIMAB")
        cleaned = re.sub(r'\s+\d+(\.\d+)?$', '', cleaned)

        # Remove common dosage form words at end
        dosage_forms = r'\s+(tablet|capsule|cap|tab|pill|cream|ointment|gel|solution|suspension|syrup|injection|patch|inhaler|spray|drops|suppository|liquid|powder)s?$'
        cleaned = re.sub(dosage_forms, '', cleaned, flags=re.IGNORECASE)

        return cleaned.strip() or text

    def _heuristic_medication_check(self, name: str, name_lower: str, name_clean: str) -> bool:
        """
        Heuristic check for medication names (fallback when database unavailable).

        Args:
            name: Original medication name
            name_lower: Lowercase version
            name_clean: Cleaned version (alphanumeric only)

        Returns:
            True if likely a medication based on heuristics
        """
        import re

        # Common pharmaceutical suffixes
        med_suffixes = [
            'in', 'ol', 'ide', 'ate', 'one', 'ine', 'an', 'il', 'al',
            'am', 'em', 'um', 'ic', 'ax', 'ex', 'ix', 'ox', 'ux',
            'zole', 'pril', 'sartan', 'statin', 'mycin', 'cycline',
            'cillin', 'floxacin', 'mab', 'nib', 'tinib', 'zumab',
            'prazole', 'tidine', 'setron', 'triptan', 'afil', 'fen',
            'profen', 'caine', 'pam', 'lam', 'pine', 'dipine',
            'azine', 'asone', 'olone', 'isone', 'derm', 'cort'
        ]

        has_med_suffix = any(name_clean.endswith(suffix) for suffix in med_suffixes)

        # Common medications list
        common_meds = [
            'aspirin', 'tylenol', 'advil', 'motrin', 'aleve', 'ibuprofen',
            'acetaminophen', 'naproxen', 'penicillin', 'amoxicillin',
            'metformin', 'lisinopril', 'atorvastatin', 'omeprazole',
            'metoprolol', 'amlodipine', 'losartan', 'gabapentin',
            'hydrocodone', 'oxycodone', 'tramadol', 'prednisone',
            'levothyroxine', 'pantoprazole', 'sertraline', 'fluoxetine',
            'escitalopram', 'duloxetine', 'bupropion', 'trazodone',
            'alprazolam', 'lorazepam', 'clonazepam', 'diazepam',
            'insulin', 'warfarin', 'eliquis', 'xarelto', 'plavix',
            'lipitor', 'crestor', 'zoloft', 'lexapro', 'prozac',
            'xanax', 'ativan', 'valium', 'ambien', 'lunesta',
            'viagra', 'cialis', 'humira', 'enbrel', 'remicade',
            'synthroid', 'nexium', 'prilosec', 'zantac', 'pepcid',
            'zyrtec', 'claritin', 'allegra', 'benadryl', 'flonase',
            'advair', 'symbicort', 'ventolin', 'albuterol', 'proair',
            'lantus', 'humalog', 'novolog', 'jardiance', 'ozempic',
            'trulicity', 'victoza', 'farxiga', 'invokana', 'januvia'
        ]

        is_known_med = any(med in name_clean for med in common_meds)

        # Dosage form indicators
        dosage_forms = [
            'tablet', 'capsule', 'cap', 'tab', 'pill', 'cream', 'ointment',
            'gel', 'solution', 'suspension', 'syrup', 'injection', 'patch',
            'inhaler', 'spray', 'drops', 'suppository', 'liquid', 'powder'
        ]
        has_dosage_form = any(form in name_clean for form in dosage_forms)

        if has_med_suffix or is_known_med:
            return True

        if has_dosage_form:
            return True

        # Check for alphanumeric pattern (brand names)
        if re.match(r'^[a-zA-Z][a-zA-Z0-9\s-]{2,30}$', name):
            common_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day',
                'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new',
                'now', 'old', 'see', 'way', 'who', 'boy', 'did', 'own',
                'say', 'she', 'too', 'use', 'take', 'with', 'daily'
            }
            words = name_lower.split()
            if all(w in common_words for w in words):
                return False

            if len(words) == 1 or any(c.isdigit() for c in name):
                return True

        return False

    def _validate_medication_entry(
        self,
        med: dict,
        context_medications: list = None
    ) -> tuple[bool, float, dict | None]:
        """
        Validate a complete medication entry for coherence.

        Checks:
        - Medication name looks valid (via RxNorm database or heuristics)
        - Has supporting evidence (strength, frequency, etc.)
        - Data consistency

        Enhanced with:
        - Dosage validation: Cross-checks extracted strength against known strengths
        - Context scoring: Uses other medications on prescription for drug class inference

        Args:
            med: Medication dictionary
            context_medications: Other medication names on same prescription

        Returns:
            Tuple of (is_valid, confidence_score, rxnorm_info)
        """
        name = med.get('medication_name', '').strip()
        strength = med.get('strength', '').strip()

        if not name:
            return False, 0.0, None

        # Check if name is likely a medication with context and strength for better matching
        is_valid, rxnorm_info = self._is_likely_medication(
            name,
            extracted_strength=strength if strength else None,
            context_medications=context_medications
        )

        if not is_valid:
            self.logger.debug(f"Rejected '{name}' - doesn't look like a medication")
            return False, 0.0, None

        # Calculate confidence based on validation source and supporting fields
        # Database match gives higher base confidence
        if rxnorm_info:
            match_type = rxnorm_info.get('match_type', 'exact')
            if match_type == 'exact':
                confidence = 0.85  # High confidence for exact RxNorm match
            elif match_type == 'prefix':
                confidence = 0.75  # Good confidence for prefix match
            elif match_type == 'word_prefix':
                confidence = 0.70  # Good confidence for word boundary match
            elif match_type == 'ocr_corrected':
                # OCR correction: lower confidence based on edit distance and similarity
                edit_distance = rxnorm_info.get('edit_distance', 3)
                similarity = rxnorm_info.get('similarity', 0.5)
                # Base confidence for OCR: 0.45-0.65 depending on similarity
                confidence = 0.40 + (similarity * 0.25)
                # Penalty for high edit distance
                if edit_distance >= 4:
                    confidence -= 0.10

                # Boost confidence if strength was validated against known dosages
                if rxnorm_info.get('strength_validated'):
                    confidence += 0.10
                    self.logger.debug(f"OCR confidence boost: strength validated for {name}")

                # Boost confidence if drug class matches prescription context
                if rxnorm_info.get('context_match'):
                    confidence += 0.08
                    self.logger.debug(f"OCR confidence boost: context match for {name}")

                # Mark for review
                rxnorm_info['needs_review'] = True
                rxnorm_info['ocr_warning'] = f"OCR correction: '{rxnorm_info.get('original_input', name)}' -> '{rxnorm_info['name']}' (distance: {edit_distance})"
                self.logger.warning(rxnorm_info['ocr_warning'])
            else:
                confidence = 0.65  # Moderate for other match types
        else:
            confidence = 0.5  # Base confidence for heuristic match only

        # Strength adds confidence
        strength = med.get('strength', '').strip()
        if strength and any(unit in strength.lower() for unit in ['mg', 'mcg', 'g', 'ml', 'unit', '%']):
            confidence += 0.05 if rxnorm_info else 0.15

        # Frequency adds confidence
        frequency = med.get('frequency', '').strip()
        if frequency and any(term in frequency.lower() for term in [
            'daily', 'twice', 'once', 'hour', 'bid', 'tid', 'qid', 'prn',
            'morning', 'evening', 'night', 'bedtime', 'meal', 'weekly'
        ]):
            confidence += 0.03 if rxnorm_info else 0.15

        # Route adds confidence
        route = med.get('route', '').strip()
        if route and any(r in route.lower() for r in [
            'oral', 'topical', 'iv', 'im', 'subcutaneous', 'sublingual',
            'inhaled', 'nasal', 'ophthalmic', 'otic', 'rectal', 'vaginal',
            'transdermal', 'by mouth', 'injection', 'applied'
        ]):
            confidence += 0.02 if rxnorm_info else 0.1

        # Quantity adds confidence
        quantity = med.get('quantity', '').strip()
        if quantity and any(c.isdigit() for c in quantity):
            confidence += 0.02 if rxnorm_info else 0.05

        # Instructions add confidence
        instructions = med.get('instructions', '').strip()
        if instructions and len(instructions) > 5:
            confidence += 0.03 if rxnorm_info else 0.05

        return True, min(confidence, 1.0), rxnorm_info

    def _extract_prescriber_info(self, text: str) -> Dict[str, str]:
        """
        Extract prescriber information from prescription.

        Args:
            text: Prescription text

        Returns:
            Dict with prescriber info
        """
        import re

        prescriber = {
            'name': '',
            'npi': '',
            'dea': '',
            'license': '',
            'phone': '',
            'address': ''
        }

        # Extract NPI (10 digits)
        npi_match = re.search(r'NPI[:\s]*(\d{10})', text, re.IGNORECASE)
        if npi_match:
            prescriber['npi'] = npi_match.group(1)

        # Extract DEA (2 letters + 7 digits)
        dea_match = re.search(r'DEA[:\s]*([A-Z]{2}\d{7})', text, re.IGNORECASE)
        if dea_match:
            prescriber['dea'] = dea_match.group(1)

        # Extract phone number
        phone_match = re.search(r'(?:phone|tel|ph)[:\s]*(\(?[\d]{3}\)?[-.\s]?[\d]{3}[-.\s]?[\d]{4})', text, re.IGNORECASE)
        if phone_match:
            prescriber['phone'] = phone_match.group(1)

        return prescriber

    def _extract_patient_info(self, text: str) -> Dict[str, str]:
        """
        Extract patient information from prescription.

        Args:
            text: Prescription text

        Returns:
            Dict with patient info
        """
        import re

        patient = {
            'name': '',
            'dob': '',
            'mrn': '',
            'allergies': []
        }

        # Extract date of birth
        dob_match = re.search(r'(?:DOB|Date of Birth)[:\s]*([\d]{1,2}[/-][\d]{1,2}[/-][\d]{2,4})', text, re.IGNORECASE)
        if dob_match:
            patient['dob'] = dob_match.group(1)

        # Extract MRN/patient ID
        mrn_match = re.search(r'(?:MRN|Patient ID|Medical Record)[:\s]*([\w\d-]+)', text, re.IGNORECASE)
        if mrn_match:
            patient['mrn'] = mrn_match.group(1)

        # Extract allergies
        allergies_match = re.search(r'(?:Allergies|Allergy)[:\s]*([^\n]+)', text, re.IGNORECASE)
        if allergies_match:
            allergies_text = allergies_match.group(1)
            if 'none' not in allergies_text.lower() and 'nkda' not in allergies_text.lower():
                patient['allergies'] = [a.strip() for a in allergies_text.split(',')]

        return patient

    def _extract_prescription_date(self, text: str) -> str:
        """
        Extract prescription date from text.

        Args:
            text: Prescription text

        Returns:
            Prescription date string
        """
        import re

        # Common date patterns
        date_patterns = [
            r'(?:Date|Prescribed|Rx Date)[:\s]*([\d]{1,2}[/-][\d]{1,2}[/-][\d]{2,4})',
            r'([\d]{1,2}[/-][\d]{1,2}[/-][\d]{2,4})',
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return ''

    async def _check_drug_interactions(
        self,
        medications: list
    ) -> list:
        """
        Check for drug-drug interactions.

        Args:
            medications: List of medication dicts

        Returns:
            List of interaction warnings
        """
        if len(medications) < 2:
            return []

        med_names = [med.get('medication_name', '') for med in medications]
        med_list = ', '.join(med_names)

        prompt = f"""Check for drug-drug interactions between these medications:

Medications: {med_list}

List any significant interactions or contraindications.
Return "None" if no significant interactions."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.1
            )

            if 'none' in response['text'].lower():
                return []

            # Parse interactions
            interactions = []
            for line in response['text'].split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•')):
                    interactions.append(line.lstrip('- •').strip())

            return interactions
        except Exception as e:
            self.logger.error(f"Drug interaction check failed: {e}")
            return []

    async def _check_contraindications(
        self,
        context: ProcessingContext
    ) -> list:
        """
        Check for contraindications based on patient conditions.

        Args:
            context: Processing context

        Returns:
            List of contraindication warnings
        """
        medications = context.sections.get('medications', [])
        if not medications:
            return []

        prompt = f"""Check for contraindications for these medications:

Medications: {', '.join([m.get('medication_name', '') for m in medications])}

List any contraindications, warnings, or special considerations.
Return "None" if no contraindications."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=250,
                temperature=0.1
            )

            if 'none' in response['text'].lower():
                return []

            # Parse contraindications
            contraindications = []
            for line in response['text'].split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•')):
                    contraindications.append(line.lstrip('- •').strip())

            return contraindications
        except Exception as e:
            self.logger.error(f"Contraindication check failed: {e}")
            return []

    async def _store_in_vector_store(
        self,
        context: ProcessingContext,
        medications: list,
        confidence: float
    ):
        """Store successful extraction in vector store for future reference."""
        try:
            extracted_values = {
                'medications': medications,
                'prescriber': context.sections.get('prescriber', {}),
                'patient': context.sections.get('patient', {}),
                'prescription_date': context.sections.get('prescription_date', ''),
            }

            await self.vector_store.store(
                text=context.raw_text[:2000],
                extracted_values=extracted_values,
                template_id='prescription',
                source_file=str(context.document_path),
                confidence=confidence
            )
            self.logger.info(f"Stored prescription extraction in vector store")
        except Exception as e:
            self.logger.debug(f"Failed to store in vector store: {e}")
