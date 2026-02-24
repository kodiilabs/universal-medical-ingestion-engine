# src/medical_ingestion/extractors/layout_analyzer.py
"""
Layout analysis utilities for PDFs.
- Preserves columns, positions
- Can be extended for OCR / scanned PDFs
"""

from pathlib import Path
import pdfplumber
import logging

class LayoutAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_text_with_layout(self, pdf_path: Path, page_num: int) -> str:
        """Extract text preserving layout (columns, spacing)."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[page_num]
                return page.extract_text(layout=True)
        except Exception as e:
            self.logger.warning(f"Layout extraction failed: {e}")
            return ""
