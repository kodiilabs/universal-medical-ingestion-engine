# ============================================================================
# src/medical_ingestion/extractors/consensus_extractor.py
# ============================================================================
"""
Consensus-Based Text Extraction

Runs both VLM (Vision Language Model) and OCR (PaddleOCR) in parallel,
then merges their results using intelligent consensus algorithms.

Why Consensus?
- PaddleOCR: Fast, good character-level accuracy for clean printed text
- VLM: Better semantic understanding, handles complex layouts and tables

By running both and merging:
- Get OCR's precise character recognition
- Get VLM's layout understanding and field extraction
- Cross-validate results for higher confidence
- Handle edge cases where one method fails

Merge Strategy:
1. Run OCR and VLM in parallel
2. If one fails, use the other's result
3. If both succeed:
   - Compare text length (longer usually means more complete)
   - Check for key-value pairs extraction
   - Merge fields from both (VLM fields often more structured)
   - Use confidence-weighted combination for final text
   - Prefer VLM for tables and complex layouts
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Result from consensus extraction."""

    # Final merged text
    full_text: str

    # Source attributions
    ocr_text: str
    vlm_text: str

    # Structured fields (merged from both)
    fields: Dict[str, Any] = field(default_factory=dict)

    # Confidence and metadata
    confidence: float = 0.0
    ocr_confidence: float = 0.0
    vlm_confidence: float = 0.0

    # Which source contributed what
    primary_source: str = "consensus"  # "ocr", "vlm", or "consensus"
    ocr_contribution: float = 0.0  # 0-1, how much OCR contributed
    vlm_contribution: float = 0.0  # 0-1, how much VLM contributed

    # Processing info
    ocr_time: float = 0.0
    vlm_time: float = 0.0
    merge_time: float = 0.0

    # Warnings
    warnings: List[str] = field(default_factory=list)


class ConsensusExtractor:
    """
    Extracts text using both VLM and OCR in parallel, then merges results.

    Usage:
        extractor = ConsensusExtractor(config)
        result = await extractor.extract(image_path)
        print(result.full_text)
        print(result.fields)
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Lazy-loaded components
        self._vlm_client = None
        self._ocr_router = None
        self._document_pipeline = None

        # Configuration
        self.vlm_weight = self.config.get('vlm_weight', 0.6)  # VLM trusted more for structure
        self.ocr_weight = self.config.get('ocr_weight', 0.4)  # OCR trusted for raw text
        self.min_similarity_for_merge = 0.3  # If texts too different, prefer longer one
        self.prefer_vlm_for_tables = True
        self.prefer_vlm_for_low_quality = True

    async def extract(
        self,
        document_path: Path,
        preprocessed_image: Any = None
    ) -> ConsensusResult:
        """
        Extract text using consensus of VLM and OCR.

        Args:
            document_path: Path to document (image or PDF)
            preprocessed_image: Optional pre-processed image (numpy array)

        Returns:
            ConsensusResult with merged extraction
        """
        document_path = Path(document_path)

        logger.info(f"Starting consensus extraction for {document_path.name}")

        # Run VLM and OCR in parallel
        vlm_task = self._extract_with_vlm(document_path)
        ocr_task = self._extract_with_ocr(document_path, preprocessed_image)

        # Wait for both to complete
        results = await asyncio.gather(
            vlm_task,
            ocr_task,
            return_exceptions=True
        )

        vlm_result = results[0] if not isinstance(results[0], Exception) else None
        ocr_result = results[1] if not isinstance(results[1], Exception) else None

        if isinstance(results[0], Exception):
            logger.warning(f"VLM extraction failed: {results[0]}")
        if isinstance(results[1], Exception):
            logger.warning(f"OCR extraction failed: {results[1]}")

        # Merge results
        merge_start = time.time()
        consensus = self._merge_results(vlm_result, ocr_result)
        consensus.merge_time = time.time() - merge_start

        logger.info(
            f"Consensus extraction complete: {len(consensus.full_text)} chars, "
            f"confidence={consensus.confidence:.2f}, "
            f"source={consensus.primary_source}"
        )

        return consensus

    async def _extract_with_vlm(
        self,
        document_path: Path
    ) -> Optional[Dict[str, Any]]:
        """Extract using VLM."""
        start_time = time.time()

        try:
            from .vlm_client import VLMClient

            if self._vlm_client is None:
                self._vlm_client = VLMClient(self.config)

            # For PDFs, we need to convert to image first
            import fitz
            suffix = document_path.suffix.lower()

            if suffix == '.pdf':
                # Convert first page to image
                import tempfile
                doc = fitz.open(str(document_path))
                page = doc[0]
                mat = fitz.Matrix(2, 2)  # 2x zoom
                pix = page.get_pixmap(matrix=mat)

                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    pix.save(f.name)
                    image_path = f.name

                doc.close()

                result = await self._vlm_client.extract_from_image(image_path, extract_all=True)

                # Clean up temp file
                import os
                os.unlink(image_path)
            else:
                # For camera images: apply EXIF orientation correction before
                # sending to VLM so it sees the image right-side-up
                import tempfile, os
                from ..utils.image_utils import load_image_for_ocr
                prepared = load_image_for_ocr(document_path)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    prepared.save(f, format='PNG')
                    image_path = f.name
                result = await self._vlm_client.extract_from_image(image_path, extract_all=True)
                os.unlink(image_path)

            return {
                'text': result.text,
                'fields': result.fields,
                'confidence': result.confidence,
                'time': time.time() - start_time,
                'raw_response': result.raw_response
            }

        except Exception as e:
            logger.error(f"VLM extraction error: {e}")
            return None

    async def _extract_with_ocr(
        self,
        document_path: Path,
        preprocessed_image: Any = None
    ) -> Optional[Dict[str, Any]]:
        """Extract using OCR pipeline."""
        start_time = time.time()

        try:
            from ..core.document_pipeline import DocumentPipeline

            if self._document_pipeline is None:
                self._document_pipeline = DocumentPipeline({
                    'enable_preprocessing': True,
                    'enable_region_detection': True,
                    'use_vlm_classification': False  # Don't use VLM here, we run it separately
                })

            suffix = document_path.suffix.lower()

            if suffix == '.pdf':
                results = await self._document_pipeline.process_pdf(document_path)
                # Combine all pages
                all_text = []
                total_confidence = 0.0
                for i, result in enumerate(results):
                    all_text.append(result.full_text)
                    total_confidence += result.average_confidence

                text = "\n".join(all_text)
                confidence = total_confidence / len(results) if results else 0.0
                engines = list(set(
                    eng for r in results for eng in r.ocr_result.engines_used
                ))
            else:
                result = await self._document_pipeline.process(document_path)
                text = result.full_text
                confidence = result.average_confidence
                engines = result.ocr_result.engines_used

            # Extract key-value pairs from OCR text
            fields = self._extract_fields_from_text(text)

            return {
                'text': text,
                'fields': fields,
                'confidence': confidence,
                'time': time.time() - start_time,
                'engines': engines
            }

        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return None

    def _extract_fields_from_text(self, text: str) -> Dict[str, Any]:
        """Extract key-value fields from raw OCR text."""
        import re

        fields = {}
        if not text:
            return fields

        # Pattern: "Label: Value" or "Label : Value"
        for line in text.split('\n'):
            line = line.strip()
            if ':' in line:
                match = re.match(r'^([^:]+?)\s*:\s*(.+)$', line)
                if match:
                    label = match.group(1).strip()
                    value = match.group(2).strip()

                    # Clean label
                    if len(label) <= 50 and len(value) >= 2:
                        key = label.lower().replace(' ', '_').replace('-', '_')
                        key = re.sub(r'[^a-z0-9_]', '', key)
                        key = re.sub(r'_+', '_', key).strip('_')

                        if key and len(key) >= 2:
                            fields[key] = value

        # Extract amounts
        amounts = re.findall(r'\$\s*([\d,]+\.?\d*)', text)
        if amounts:
            fields['_amounts'] = amounts

        # Extract dates
        dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2}', text)
        if dates:
            fields['_dates'] = dates

        return fields

    def _merge_results(
        self,
        vlm_result: Optional[Dict[str, Any]],
        ocr_result: Optional[Dict[str, Any]]
    ) -> ConsensusResult:
        """
        Merge VLM and OCR results using intelligent consensus.

        Strategy:
        1. If only one succeeded, use that
        2. If both succeeded:
           - Compare text similarity
           - If similar, merge fields and use weighted text
           - If different, prefer the more complete one
           - Always merge fields (VLM fields usually more structured)
        """
        # Case 1: Only OCR succeeded
        if vlm_result is None and ocr_result is not None:
            return ConsensusResult(
                full_text=ocr_result['text'],
                ocr_text=ocr_result['text'],
                vlm_text="",
                fields=ocr_result.get('fields', {}),
                confidence=ocr_result['confidence'],
                ocr_confidence=ocr_result['confidence'],
                vlm_confidence=0.0,
                primary_source="ocr",
                ocr_contribution=1.0,
                vlm_contribution=0.0,
                ocr_time=ocr_result['time'],
                vlm_time=0.0,
                warnings=["VLM extraction failed, using OCR only"]
            )

        # Case 2: Only VLM succeeded
        if ocr_result is None and vlm_result is not None:
            return ConsensusResult(
                full_text=vlm_result['text'],
                ocr_text="",
                vlm_text=vlm_result['text'],
                fields=vlm_result.get('fields', {}),
                confidence=vlm_result['confidence'],
                ocr_confidence=0.0,
                vlm_confidence=vlm_result['confidence'],
                primary_source="vlm",
                ocr_contribution=0.0,
                vlm_contribution=1.0,
                ocr_time=0.0,
                vlm_time=vlm_result['time'],
                warnings=["OCR extraction failed, using VLM only"]
            )

        # Case 3: Both failed
        if vlm_result is None and ocr_result is None:
            return ConsensusResult(
                full_text="",
                ocr_text="",
                vlm_text="",
                fields={},
                confidence=0.0,
                primary_source="none",
                warnings=["Both VLM and OCR extraction failed"]
            )

        # Case 4: Both succeeded - merge intelligently
        ocr_text = ocr_result['text']
        vlm_text = vlm_result['text']

        ocr_fields = ocr_result.get('fields', {})
        vlm_fields = vlm_result.get('fields', {})

        ocr_conf = ocr_result['confidence']
        vlm_conf = vlm_result['confidence']

        # Calculate text similarity
        similarity = self._calculate_similarity(ocr_text, vlm_text)

        # Determine primary source based on quality metrics
        ocr_len = len(ocr_text.strip())
        vlm_len = len(vlm_text.strip())

        # VLM usually better for:
        # - Low confidence OCR (poor quality images)
        # - Documents with tables (VLM understands structure)
        # - When VLM extracted more structured fields

        prefer_vlm = (
            (ocr_conf < 0.5 and self.prefer_vlm_for_low_quality) or
            (len(vlm_fields) > len(ocr_fields) * 1.5) or
            (vlm_conf > ocr_conf + 0.2)
        )

        # OCR usually better for:
        # - Clean printed text
        # - When OCR has higher confidence
        # - When OCR extracted more text

        prefer_ocr = (
            (ocr_conf > 0.8 and ocr_len > vlm_len * 1.2) or
            (ocr_conf > vlm_conf + 0.2)
        )

        # Decide merge strategy
        if similarity > 0.7:
            # Texts are very similar - merge and average confidence
            primary_source = "consensus"
            merged_text = self._merge_texts(ocr_text, vlm_text, similarity)
            merged_confidence = (ocr_conf * self.ocr_weight + vlm_conf * self.vlm_weight)
            ocr_contrib = 0.5
            vlm_contrib = 0.5

        elif prefer_vlm:
            # VLM seems more reliable for this document
            primary_source = "vlm"
            merged_text = vlm_text
            merged_confidence = vlm_conf
            ocr_contrib = 0.2
            vlm_contrib = 0.8

        elif prefer_ocr:
            # OCR seems more reliable
            primary_source = "ocr"
            merged_text = ocr_text
            merged_confidence = ocr_conf
            ocr_contrib = 0.8
            vlm_contrib = 0.2

        else:
            # Default: prefer longer text (usually more complete)
            if vlm_len > ocr_len * 1.1:
                primary_source = "vlm"
                merged_text = vlm_text
                merged_confidence = vlm_conf
                ocr_contrib = 0.3
                vlm_contrib = 0.7
            elif ocr_len > vlm_len * 1.1:
                primary_source = "ocr"
                merged_text = ocr_text
                merged_confidence = ocr_conf
                ocr_contrib = 0.7
                vlm_contrib = 0.3
            else:
                # Very similar length - use consensus
                primary_source = "consensus"
                merged_text = self._merge_texts(ocr_text, vlm_text, similarity)
                merged_confidence = (ocr_conf * self.ocr_weight + vlm_conf * self.vlm_weight)
                ocr_contrib = 0.5
                vlm_contrib = 0.5

        # Always merge fields (VLM fields take precedence for structured data)
        merged_fields = self._merge_fields(ocr_fields, vlm_fields)

        # Build warnings
        warnings = []
        if similarity < 0.3:
            warnings.append(f"Low agreement between VLM and OCR (similarity={similarity:.2f})")
        if ocr_conf < 0.5:
            warnings.append(f"Low OCR confidence ({ocr_conf:.2f})")
        if vlm_conf < 0.5:
            warnings.append(f"Low VLM confidence ({vlm_conf:.2f})")

        return ConsensusResult(
            full_text=merged_text,
            ocr_text=ocr_text,
            vlm_text=vlm_text,
            fields=merged_fields,
            confidence=merged_confidence,
            ocr_confidence=ocr_conf,
            vlm_confidence=vlm_conf,
            primary_source=primary_source,
            ocr_contribution=ocr_contrib,
            vlm_contribution=vlm_contrib,
            ocr_time=ocr_result['time'],
            vlm_time=vlm_result['time'],
            warnings=warnings
        )

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        t1 = ' '.join(text1.lower().split())
        t2 = ' '.join(text2.lower().split())

        # Use SequenceMatcher for similarity
        return SequenceMatcher(None, t1, t2).ratio()

    def _merge_texts(self, ocr_text: str, vlm_text: str, similarity: float) -> str:
        """
        Merge two similar texts.

        For high similarity, just return the longer/better one.
        For medium similarity, try to combine unique parts.
        """
        if similarity > 0.9:
            # Nearly identical - return whichever is longer
            return vlm_text if len(vlm_text) >= len(ocr_text) else ocr_text

        # For medium similarity, prefer VLM as it usually has better structure
        return vlm_text if len(vlm_text) > len(ocr_text) * 0.9 else ocr_text

    def _merge_fields(
        self,
        ocr_fields: Dict[str, Any],
        vlm_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge fields from OCR and VLM.

        VLM fields take precedence for same keys (usually more accurate).
        Keep unique fields from both.
        """
        merged = {}

        # Add OCR fields first
        for key, value in ocr_fields.items():
            if not key.startswith('_'):  # Skip internal fields
                merged[key] = value

        # VLM fields overwrite OCR (VLM usually more accurate for structure)
        for key, value in vlm_fields.items():
            if not key.startswith('_'):
                merged[key] = value

        # Keep special fields from OCR if VLM didn't extract them
        if '_amounts' in ocr_fields and '_amounts' not in vlm_fields:
            merged['_amounts'] = ocr_fields['_amounts']
        if '_dates' in ocr_fields and '_dates' not in vlm_fields:
            merged['_dates'] = ocr_fields['_dates']

        return merged


# Convenience function
async def extract_with_consensus(
    document_path: Path,
    config: Dict[str, Any] = None
) -> ConsensusResult:
    """
    Convenience function for consensus extraction.

    Args:
        document_path: Path to document
        config: Optional configuration

    Returns:
        ConsensusResult with merged extraction
    """
    extractor = ConsensusExtractor(config or {})
    return await extractor.extract(document_path)
