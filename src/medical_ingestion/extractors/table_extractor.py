# src/medical_ingestion/extractors/table_extractor.py
"""
Table extraction from PDFs using Camelot + pdfplumber + PyMuPDF.

Extraction cascade:
1. Camelot (lattice) - Best for tables with visible borders
2. Camelot (stream) - For borderless tables with clear structure
3. pdfplumber - Alternative for complex layouts
4. PyMuPDF layout analysis - For borderless tables using bounding boxes
5. OCR + pdfplumber - For scanned PDFs
6. MedGemma vision - Last resort for complex layouts

Camelot is the primary method as it handles lab report formats well.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging
import asyncio
import base64
import json
import re
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import pdfplumber
import pymupdf  # PyMuPDF
from json_repair import repair_json

from ..core.bbox_utils import normalize_bbox, validate_bbox, fix_bbox_ordering

# Try to import camelot (optional but preferred)
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound PDF operations
_pdf_executor = ThreadPoolExecutor(max_workers=4)


@dataclass
class CellBox:
    """A table cell with its content and bounding box."""
    text: str
    row: int
    col: int
    bbox: Optional[Tuple[float, float, float, float]] = None  # (x0, y0, x1, y1) normalized 0-1


@dataclass
class ExtractedTable:
    """A table extracted from a PDF."""
    page_number: int
    table_index: int
    headers: List[str]
    rows: List[List[str]]
    confidence: float
    bbox: Optional[Tuple[float, float, float, float]] = None  # Table bbox normalized 0-1
    extraction_method: str = "pdfplumber"
    # Cell-level bounding boxes for value highlighting
    cell_boxes: List[CellBox] = None  # All cells with their positions
    page_width: float = 0  # PDF page width in points
    page_height: float = 0  # PDF page height in points

    def get_cell_bbox(self, row: int, col: int) -> Optional[Tuple[float, float, float, float]]:
        """Get the bounding box for a specific cell (normalized 0-1)."""
        if not self.cell_boxes:
            return None
        for cell in self.cell_boxes:
            if cell.row == row and cell.col == col:
                return cell.bbox
        return None

    def find_value_bbox(self, value: str) -> Optional[Tuple[float, float, float, float]]:
        """Find the bounding box for a specific value in the table."""
        if not self.cell_boxes:
            return None
        value_lower = str(value).lower().strip()
        for cell in self.cell_boxes:
            if cell.text.lower().strip() == value_lower:
                return cell.bbox
        # Try partial match
        for cell in self.cell_boxes:
            if value_lower in cell.text.lower() or cell.text.lower() in value_lower:
                return cell.bbox
        return None


class TableExtractor:
    """
    Robust table extraction with pdfplumber + PyMuPDF.

    pdfplumber advantages over Camelot:
    - Better handling of borderless tables
    - More accurate cell detection
    - Faster processing
    - Better text positioning

    PyMuPDF adds:
    - Precise bounding box detection
    - Layout analysis for complex documents
    - Fast page rendering for OCR fallback
    """

    def __init__(self, use_ocr_fallback: bool = True, use_vision_fallback: bool = True):
        """
        Initialize table extractor.

        Args:
            use_ocr_fallback: Use OCR when text extraction fails
            use_vision_fallback: Use MedGemma vision as last resort
        """
        self.logger = logging.getLogger(__name__)
        self.use_ocr_fallback = use_ocr_fallback
        self.use_vision_fallback = use_vision_fallback

    def extract_tables(
        self,
        pdf_path: Path,
        pages: Optional[List[int]] = None
    ) -> List[ExtractedTable]:
        """
        Extract all tables from a PDF.

        Cascade:
        1. Camelot lattice (tables with borders)
        2. Camelot stream (borderless tables)
        3. pdfplumber (alternative)
        4. PyMuPDF layout analysis
        5. OCR (for scanned PDFs)
        6. MedGemma vision (last resort)

        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to extract (1-indexed), None for all

        Returns:
            List of extracted tables
        """
        self.logger.debug(f"Extracting tables from {pdf_path}")

        # Check if PDF has extractable text
        has_text, text_density = self._check_text_density(pdf_path)

        if has_text and text_density > 0.1:
            # ================================================================
            # PRIMARY: Try Camelot first (best for lab reports)
            # ================================================================
            if CAMELOT_AVAILABLE:
                # Try lattice mode (tables with visible borders)
                tables = self._extract_with_camelot(pdf_path, pages, flavor='lattice')
                if tables and len(tables) >= 1:
                    self.logger.info(f"Camelot lattice extracted {len(tables)} tables")
                    return tables

                # Try stream mode (borderless tables)
                tables = self._extract_with_camelot(pdf_path, pages, flavor='stream')
                if tables and len(tables) >= 1:
                    self.logger.info(f"Camelot stream extracted {len(tables)} tables")
                    return tables

            # ================================================================
            # FALLBACK 1: pdfplumber
            # ================================================================
            tables = self._extract_with_pdfplumber(pdf_path, pages)
            if tables:
                self.logger.info(f"pdfplumber extracted {len(tables)} tables")
                return tables

            # ================================================================
            # FALLBACK 2: PyMuPDF layout analysis
            # ================================================================
            tables = self._extract_with_pymupdf_layout(pdf_path, pages)
            if tables:
                self.logger.info(f"PyMuPDF layout extracted {len(tables)} tables")
                return tables

        # Low text density - likely scanned PDF
        if self.use_ocr_fallback and text_density < 0.1:
            self.logger.info(f"Low text density ({text_density:.2f}), trying OCR...")
            tables = self._extract_with_ocr(pdf_path, pages)
            if tables:
                self.logger.info(f"OCR extracted {len(tables)} tables")
                return tables

        # Last resort: MedGemma vision
        if self.use_vision_fallback:
            self.logger.info("Trying MedGemma vision extraction...")
            tables = self._extract_with_vision(pdf_path, pages)
            if tables:
                self.logger.info(f"Vision extracted {len(tables)} tables")
                return tables

        self.logger.warning("No tables extracted from PDF")
        return []

    def _extract_with_camelot(
        self,
        pdf_path: Path,
        pages: Optional[List[int]] = None,
        flavor: str = 'lattice'
    ) -> List[ExtractedTable]:
        """
        Extract tables using Camelot.

        Camelot is excellent for lab reports because:
        - Handles both bordered and borderless tables
        - Good at detecting table regions
        - Preserves column structure well

        Args:
            pdf_path: Path to PDF
            pages: Page numbers (1-indexed)
            flavor: 'lattice' for bordered tables, 'stream' for borderless
        """
        if not CAMELOT_AVAILABLE:
            return []

        tables = []

        try:
            # Get page dimensions using PyMuPDF
            doc = pymupdf.open(str(pdf_path))
            page_dimensions = {}
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_dimensions[page_num + 1] = (float(page.rect.width), float(page.rect.height))
            doc.close()

            # Convert pages to camelot format (comma-separated string)
            if pages:
                pages_str = ','.join(str(p) for p in pages)
            else:
                pages_str = 'all'

            # Extract with camelot
            camelot_tables = camelot.read_pdf(
                str(pdf_path),
                pages=pages_str,
                flavor=flavor,
                suppress_stdout=True
            )

            for i, ct in enumerate(camelot_tables):
                df = ct.df

                if df.empty or len(df) < 2:
                    continue

                # Get page dimensions
                page_num = ct.page if hasattr(ct, 'page') else 1
                page_width, page_height = page_dimensions.get(page_num, (612.0, 792.0))

                # Convert to our format
                # First row as headers
                headers = [str(cell).strip() for cell in df.iloc[0].tolist()]
                rows = []
                for _, row in df.iloc[1:].iterrows():
                    rows.append([str(cell).strip() for cell in row.tolist()])

                # Skip if too few columns
                if len(headers) < 2:
                    continue

                # Extract cell bounding boxes from Camelot's cells attribute
                cell_boxes = []
                try:
                    if hasattr(ct, 'cells') and ct.cells is not None:
                        for row_idx, row_cells in enumerate(ct.cells):
                            for col_idx, cell in enumerate(row_cells):
                                if cell is not None:
                                    # Camelot cells use PDF coordinate system (bottom-left origin)
                                    # x1,y1 = bottom-left corner; x2,y2 = top-right corner
                                    # We need to convert to top-left origin with (x0,y0)=top-left, (x1,y1)=bottom-right
                                    x0 = getattr(cell, 'x1', getattr(cell, 'lb', [0])[0] if hasattr(cell, 'lb') else 0)
                                    y_bottom = getattr(cell, 'y1', getattr(cell, 'lb', [0, 0])[1] if hasattr(cell, 'lb') else 0)
                                    x1 = getattr(cell, 'x2', getattr(cell, 'rb', [0])[0] if hasattr(cell, 'rb') else 0)
                                    y_top = getattr(cell, 'y2', getattr(cell, 'lt', [0, 0])[1] if hasattr(cell, 'lt') else 0)

                                    # Get cell text
                                    cell_text = ""
                                    if row_idx < len(df):
                                        if col_idx < len(df.columns):
                                            cell_text = str(df.iloc[row_idx, col_idx]).strip()

                                    # Use normalize_bbox utility for proper coordinate conversion
                                    # Pass origin="bottom-left" since Camelot uses PDF coordinates
                                    raw_bbox = (x0, y_bottom, x1, y_top)
                                    norm_bbox = normalize_bbox(
                                        raw_bbox,
                                        page_width=page_width,
                                        page_height=page_height,
                                        origin="bottom-left"
                                    )

                                    if norm_bbox and validate_bbox(norm_bbox):
                                        cell_boxes.append(CellBox(
                                            text=cell_text,
                                            row=row_idx,
                                            col=col_idx,
                                            bbox=norm_bbox
                                        ))
                except Exception as cell_err:
                    self.logger.debug(f"Camelot cell bbox extraction failed: {cell_err}")

                # Get table-level bbox using normalize_bbox utility
                table_bbox = None
                try:
                    if hasattr(ct, '_bbox') and ct._bbox is not None:
                        tb = ct._bbox
                        # Camelot _bbox is (x0, y0, x1, y1) in PDF coordinates (bottom-left origin)
                        raw_table_bbox = (tb[0], tb[1], tb[2], tb[3])
                        table_bbox = normalize_bbox(
                            raw_table_bbox,
                            page_width=page_width,
                            page_height=page_height,
                            origin="bottom-left"
                        )
                except Exception:
                    pass

                # Calculate confidence from camelot's accuracy
                confidence = ct.accuracy / 100.0 if hasattr(ct, 'accuracy') else 0.85

                tables.append(ExtractedTable(
                    page_number=page_num,
                    table_index=i,
                    headers=headers,
                    rows=rows,
                    confidence=confidence,
                    bbox=table_bbox,
                    extraction_method=f"camelot_{flavor}",
                    cell_boxes=cell_boxes if cell_boxes else None,
                    page_width=page_width,
                    page_height=page_height
                ))

            return tables

        except Exception as e:
            self.logger.warning(f"Camelot {flavor} extraction failed: {e}")
            return []

    def _check_text_density(self, pdf_path: Path) -> Tuple[bool, float]:
        """
        Check if PDF has extractable text and calculate density.

        Returns:
            (has_text, density) where density is chars per page
        """
        try:
            doc = pymupdf.open(str(pdf_path))
            total_chars = 0
            total_area = 0

            for page in doc:
                text = page.get_text()
                total_chars += len(text.strip())
                rect = page.rect
                total_area += rect.width * rect.height

            doc.close()

            # Density = chars per 1000 sq units
            density = (total_chars / max(total_area, 1)) * 1000 if total_area > 0 else 0
            has_text = total_chars > 100

            self.logger.debug(f"Text check: {total_chars} chars, density={density:.3f}")
            return has_text, density

        except Exception as e:
            self.logger.warning(f"Text density check failed: {e}")
            return True, 1.0  # Assume text exists

    def _extract_with_pdfplumber(
        self,
        pdf_path: Path,
        pages: Optional[List[int]] = None
    ) -> List[ExtractedTable]:
        """
        Extract tables using pdfplumber.

        pdfplumber settings tuned for lab reports:
        - Snap tolerance for slight misalignments
        - Join tolerance for split cells
        - Text tolerance for character grouping
        """
        tables = []

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                page_nums = pages if pages else range(1, len(pdf.pages) + 1)

                for page_num in page_nums:
                    if page_num < 1 or page_num > len(pdf.pages):
                        continue

                    page = pdf.pages[page_num - 1]  # 0-indexed
                    page_width = float(page.width)
                    page_height = float(page.height)

                    # Table detection settings optimized for lab reports
                    table_settings = {
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                        "snap_tolerance": 5,
                        "join_tolerance": 5,
                        "edge_min_length": 10,
                        "min_words_vertical": 1,
                        "min_words_horizontal": 1,
                    }

                    # Try strict line detection first - use find_tables to get bbox info
                    found_tables = page.find_tables(table_settings)

                    # If no tables found, try text-based detection
                    if not found_tables:
                        table_settings["vertical_strategy"] = "text"
                        table_settings["horizontal_strategy"] = "text"
                        found_tables = page.find_tables(table_settings)

                    for i, table_obj in enumerate(found_tables):
                        table_data = table_obj.extract()
                        if not table_data or len(table_data) < 2:
                            continue

                        # Clean and parse table
                        cleaned = self._clean_table_data(table_data)
                        if not cleaned or len(cleaned) < 2:
                            continue

                        # First row as headers
                        headers = [str(cell or '').strip() for cell in cleaned[0]]
                        rows = [[str(cell or '').strip() for cell in row] for row in cleaned[1:]]

                        # Skip if too few columns or rows
                        if len(headers) < 2 or len(rows) < 1:
                            continue

                        # Extract cell-level bounding boxes
                        cell_boxes = []
                        try:
                            # pdfplumber table cells have bbox attribute
                            for row_idx, row_cells in enumerate(table_obj.cells):
                                if row_idx >= len(cleaned):
                                    break
                                for col_idx, cell_bbox in enumerate(row_cells if isinstance(row_cells, list) else [row_cells]):
                                    if col_idx >= len(cleaned[row_idx]):
                                        break
                                    # cell_bbox is (x0, y0, x1, y1) in PDF points
                                    if cell_bbox and len(cell_bbox) >= 4:
                                        x0, y0, x1, y1 = cell_bbox[:4]
                                        # Normalize to 0-1 range
                                        norm_bbox = (
                                            x0 / page_width,
                                            y0 / page_height,
                                            x1 / page_width,
                                            y1 / page_height
                                        )
                                        cell_text = cleaned[row_idx][col_idx] if col_idx < len(cleaned[row_idx]) else ""
                                        cell_boxes.append(CellBox(
                                            text=cell_text,
                                            row=row_idx,
                                            col=col_idx,
                                            bbox=norm_bbox
                                        ))
                        except Exception as cell_err:
                            # Fall back to row-based cell extraction
                            self.logger.debug(f"Cell bbox extraction failed, trying rows: {cell_err}")
                            try:
                                for row_idx, row_obj in enumerate(table_obj.rows):
                                    if row_idx >= len(cleaned):
                                        break
                                    for col_idx, cell in enumerate(row_obj.cells):
                                        if col_idx >= len(cleaned[row_idx]):
                                            break
                                        if cell and hasattr(cell, '__iter__') and len(cell) >= 4:
                                            x0, y0, x1, y1 = cell[:4]
                                            norm_bbox = (
                                                x0 / page_width,
                                                y0 / page_height,
                                                x1 / page_width,
                                                y1 / page_height
                                            )
                                            cell_text = cleaned[row_idx][col_idx] if col_idx < len(cleaned[row_idx]) else ""
                                            cell_boxes.append(CellBox(
                                                text=cell_text,
                                                row=row_idx,
                                                col=col_idx,
                                                bbox=norm_bbox
                                            ))
                            except Exception as row_err:
                                self.logger.debug(f"Row-based cell bbox extraction also failed: {row_err}")

                        # Get table-level bbox (normalized)
                        table_bbox = None
                        if hasattr(table_obj, 'bbox') and table_obj.bbox:
                            tb = table_obj.bbox
                            table_bbox = (
                                tb[0] / page_width,
                                tb[1] / page_height,
                                tb[2] / page_width,
                                tb[3] / page_height
                            )

                        # Calculate confidence based on structure
                        confidence = self._calculate_table_confidence(headers, rows)

                        tables.append(ExtractedTable(
                            page_number=page_num,
                            table_index=i,
                            headers=headers,
                            rows=rows,
                            confidence=confidence,
                            bbox=table_bbox,
                            extraction_method="pdfplumber",
                            cell_boxes=cell_boxes if cell_boxes else None,
                            page_width=page_width,
                            page_height=page_height
                        ))

            return tables

        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}")
            return []

    def _extract_with_pymupdf_layout(
        self,
        pdf_path: Path,
        pages: Optional[List[int]] = None
    ) -> List[ExtractedTable]:
        """
        Extract tables using PyMuPDF layout analysis.

        Uses text blocks and their bounding boxes to detect
        table structure even without visible borders.
        """
        tables = []

        try:
            doc = pymupdf.open(str(pdf_path))
            page_nums = pages if pages else range(1, len(doc) + 1)

            for page_num in page_nums:
                if page_num < 1 or page_num > len(doc):
                    continue

                page = doc[page_num - 1]
                page_width = float(page.rect.width)
                page_height = float(page.rect.height)

                # Get text blocks with positions
                blocks = page.get_text("dict")["blocks"]

                # Filter text blocks
                text_blocks = [b for b in blocks if b.get("type") == 0]

                if not text_blocks:
                    continue

                # Detect table regions using line clustering
                table_regions = self._detect_table_regions(text_blocks, page.rect)

                for i, region in enumerate(table_regions):
                    table_data, cell_boxes = self._extract_region_as_table_with_boxes(
                        text_blocks, region, page_width, page_height
                    )

                    if table_data and len(table_data) >= 2:
                        headers = table_data[0]
                        rows = table_data[1:]

                        if len(headers) >= 2 and len(rows) >= 1:
                            # Dynamic confidence based on table quality
                            table_confidence = 0.70  # Base
                            if len(headers) >= 3:
                                table_confidence += 0.05  # More columns = more structured
                            if len(rows) >= 3:
                                table_confidence += 0.05  # More rows = more data
                            if len(rows) >= 10:
                                table_confidence += 0.05  # Many rows = likely real data

                            # Normalize region bbox
                            norm_region = (
                                region[0] / page_width,
                                region[1] / page_height,
                                region[2] / page_width,
                                region[3] / page_height
                            )

                            tables.append(ExtractedTable(
                                page_number=page_num,
                                table_index=i,
                                headers=headers,
                                rows=rows,
                                confidence=min(0.90, table_confidence),
                                bbox=norm_region,
                                extraction_method="pymupdf_layout",
                                cell_boxes=cell_boxes if cell_boxes else None,
                                page_width=page_width,
                                page_height=page_height
                            ))

            doc.close()
            return tables

        except Exception as e:
            self.logger.warning(f"PyMuPDF layout extraction failed: {e}")
            return []

    def _detect_table_regions(
        self,
        blocks: List[Dict],
        page_rect: pymupdf.Rect
    ) -> List[Tuple[float, float, float, float]]:
        """
        Detect table regions by analyzing text block positions.

        Lab reports typically have:
        - Consistent left margins
        - Columnar alignment
        - Regular vertical spacing
        """
        if not blocks:
            return []

        # Collect all line positions
        lines = []
        for block in blocks:
            for line in block.get("lines", []):
                bbox = line.get("bbox", [0, 0, 0, 0])
                text = " ".join(span.get("text", "") for span in line.get("spans", []))
                if text.strip():
                    lines.append({
                        "y": bbox[1],
                        "x0": bbox[0],
                        "x1": bbox[2],
                        "text": text.strip(),
                        "bbox": bbox
                    })

        if len(lines) < 3:
            return []

        # Sort by y position
        lines.sort(key=lambda l: l["y"])

        # Detect columnar regions (multiple items at similar y)
        table_start = None
        table_end = None
        y_tolerance = 5

        for i, line in enumerate(lines):
            # Check if this line has columnar structure
            # (look for multiple text spans at similar y)
            same_row = [l for l in lines if abs(l["y"] - line["y"]) < y_tolerance]

            if len(same_row) >= 2:
                if table_start is None:
                    table_start = i
                table_end = i

        if table_start is not None and table_end is not None and table_end > table_start:
            # Get bounding box of table region
            table_lines = lines[table_start:table_end + 1]
            x0 = min(l["x0"] for l in table_lines)
            y0 = min(l["y"] for l in table_lines)
            x1 = max(l["x1"] for l in table_lines)
            y1 = max(l["bbox"][3] for l in table_lines)

            return [(x0, y0, x1, y1)]

        return []

    def _extract_region_as_table(
        self,
        blocks: List[Dict],
        region: Tuple[float, float, float, float]
    ) -> List[List[str]]:
        """
        Extract text from a region as table rows/columns.
        """
        table_data, _ = self._extract_region_as_table_with_boxes(blocks, region, 1.0, 1.0)
        return table_data

    def _extract_region_as_table_with_boxes(
        self,
        blocks: List[Dict],
        region: Tuple[float, float, float, float],
        page_width: float,
        page_height: float
    ) -> Tuple[List[List[str]], List[CellBox]]:
        """
        Extract text from a region as table rows/columns with bounding boxes.

        Returns:
            Tuple of (table_data, cell_boxes)
        """
        x0, y0, x1, y1 = region

        # Collect text items in region
        items = []
        for block in blocks:
            for line in block.get("lines", []):
                bbox = line.get("bbox", [0, 0, 0, 0])
                # Check if line is in region
                if bbox[1] >= y0 - 5 and bbox[3] <= y1 + 5:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            span_bbox = span.get("bbox", bbox)
                            items.append({
                                "x": span_bbox[0],
                                "y": span_bbox[1],
                                "x1": span_bbox[2] if len(span_bbox) > 2 else span_bbox[0] + 50,
                                "y1": span_bbox[3] if len(span_bbox) > 3 else span_bbox[1] + 12,
                                "text": text
                            })

        if not items:
            return [], []

        # Group by y position (rows)
        items.sort(key=lambda i: (round(i["y"] / 10) * 10, i["x"]))

        rows = []
        current_y = None
        current_row = []
        y_tolerance = 8

        for item in items:
            if current_y is None or abs(item["y"] - current_y) > y_tolerance:
                if current_row:
                    rows.append(current_row)
                current_row = [item]
                current_y = item["y"]
            else:
                current_row.append(item)

        if current_row:
            rows.append(current_row)

        # Convert to table format (sort each row by x) and extract bounding boxes
        table = []
        cell_boxes = []

        for row_idx, row in enumerate(rows):
            row.sort(key=lambda i: i["x"])
            table.append([item["text"] for item in row])

            # Create cell boxes for each item in this row
            for col_idx, item in enumerate(row):
                # Normalize bbox to 0-1 range
                norm_bbox = (
                    item["x"] / page_width,
                    item["y"] / page_height,
                    item["x1"] / page_width,
                    item["y1"] / page_height
                )
                cell_boxes.append(CellBox(
                    text=item["text"],
                    row=row_idx,
                    col=col_idx,
                    bbox=norm_bbox
                ))

        return table, cell_boxes

    def _extract_with_ocr(
        self,
        pdf_path: Path,
        pages: Optional[List[int]] = None
    ) -> List[ExtractedTable]:
        """
        Extract tables using OCR for scanned PDFs.

        Uses PyMuPDF to render pages, then pytesseract for OCR,
        then pdfplumber-style parsing on the OCR text.
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            self.logger.warning("pytesseract not installed, skipping OCR")
            return []

        tables = []

        try:
            doc = pymupdf.open(str(pdf_path))
            page_nums = pages if pages else range(1, len(doc) + 1)

            for page_num in page_nums:
                if page_num < 1 or page_num > len(doc):
                    continue

                page = doc[page_num - 1]
                # Get actual page dimensions (in PDF points)
                page_width = float(page.rect.width)
                page_height = float(page.rect.height)

                # Render page at high DPI for OCR
                scale = 2.0  # 2x zoom = ~144 DPI
                mat = pymupdf.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Run OCR with table detection
                ocr_data = pytesseract.image_to_data(
                    img,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'  # Assume uniform block of text
                )

                # Parse OCR output into table structure with bboxes
                table_data, cell_boxes = self._parse_ocr_to_table_with_boxes(
                    ocr_data, pix.width, pix.height, scale
                )

                if table_data and len(table_data) >= 2:
                    tables.append(ExtractedTable(
                        page_number=page_num,
                        table_index=0,
                        headers=table_data[0],
                        rows=table_data[1:],
                        confidence=0.70,
                        extraction_method="ocr",
                        cell_boxes=cell_boxes if cell_boxes else None,
                        page_width=page_width,
                        page_height=page_height
                    ))

            doc.close()
            return tables

        except Exception as e:
            self.logger.warning(f"OCR extraction failed: {e}")
            return []

    def _parse_ocr_to_table(
        self,
        ocr_data: Dict,
        width: int,
        height: int
    ) -> List[List[str]]:
        """
        Parse OCR output into table structure.

        Groups text by line (y position) and column (x position).
        """
        table_data, _ = self._parse_ocr_to_table_with_boxes(ocr_data, width, height, 1.0)
        return table_data

    def _parse_ocr_to_table_with_boxes(
        self,
        ocr_data: Dict,
        width: int,
        height: int,
        scale: float = 1.0
    ) -> Tuple[List[List[str]], List[CellBox]]:
        """
        Parse OCR output into table structure with bounding boxes.

        Groups text by line (y position) and column (x position).

        Args:
            ocr_data: Tesseract OCR output dictionary
            width: Image width in pixels
            height: Image height in pixels
            scale: Scale factor used for rendering (to convert back to PDF coords)

        Returns:
            Tuple of (table_data, cell_boxes)
        """
        n_boxes = len(ocr_data['text'])

        # Collect valid text items
        items = []
        for i in range(n_boxes):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])

            if text and conf > 30:  # Filter low confidence
                items.append({
                    'x': ocr_data['left'][i],
                    'y': ocr_data['top'][i],
                    'w': ocr_data['width'][i],
                    'h': ocr_data['height'][i],
                    'text': text
                })

        if not items:
            return [], []

        # Group by y position (rows)
        items.sort(key=lambda i: (i['y'] // 20, i['x']))

        rows = []
        current_y = None
        current_row = []
        y_tolerance = 15

        for item in items:
            if current_y is None or abs(item['y'] - current_y) > y_tolerance:
                if current_row:
                    rows.append(current_row)
                current_row = [item]
                current_y = item['y']
            else:
                current_row.append(item)

        if current_row:
            rows.append(current_row)

        # Convert to table (sort each row by x) and extract bounding boxes
        table = []
        cell_boxes = []

        for row_idx, row in enumerate(rows):
            row.sort(key=lambda i: i['x'])
            table.append([item['text'] for item in row])

            # Create cell boxes for each item
            for col_idx, item in enumerate(row):
                # Convert from scaled image coordinates to normalized 0-1 range
                # First divide by scale to get back to PDF points, then normalize
                x0 = item['x'] / width  # Already normalized by image dimensions
                y0 = item['y'] / height
                x1 = (item['x'] + item['w']) / width
                y1 = (item['y'] + item['h']) / height

                cell_boxes.append(CellBox(
                    text=item['text'],
                    row=row_idx,
                    col=col_idx,
                    bbox=(x0, y0, x1, y1)
                ))

        return table, cell_boxes

    def _extract_with_vision(
        self,
        pdf_path: Path,
        pages: Optional[List[int]] = None
    ) -> List[ExtractedTable]:
        """
        Extract tables using MedGemma vision (last resort).
        """
        try:
            doc = pymupdf.open(str(pdf_path))
            all_tables = []
            page_nums = pages if pages else range(1, len(doc) + 1)

            for page_num in page_nums:
                if page_num < 1 or page_num > len(doc):
                    continue

                page = doc[page_num - 1]

                # Render page
                mat = pymupdf.Matrix(1.5, 1.5)  # 1.5x zoom
                pix = page.get_pixmap(matrix=mat)

                # Convert to base64
                img_bytes = pix.tobytes("jpeg")
                image_b64 = base64.b64encode(img_bytes).decode('utf-8')

                # Extract tables via vision
                tables = asyncio.get_event_loop().run_until_complete(
                    self._vision_extract_page(image_b64, page_num)
                )
                all_tables.extend(tables)

            doc.close()
            return all_tables

        except Exception as e:
            self.logger.error(f"Vision extraction failed: {e}")
            return []

    async def _vision_extract_page(
        self,
        image_b64: str,
        page_num: int
    ) -> List[ExtractedTable]:
        """Extract tables from a single page image using MedGemma vision."""
        import aiohttp

        prompt = """Analyze this medical document image and extract ALL tables you can see.

A table is any structured data with rows and columns, even if there are no visible borders.
Lab reports typically have tables with columns like: Test Name | Result | Flag | Units | Reference Range

For EACH table found, extract:
1. The column headers (first row)
2. All data rows

Return a JSON object with this structure:
{
    "tables": [
        {
            "headers": ["Test", "Result", "Flag", "Units", "Reference Range"],
            "rows": [
                ["WBC", "6.4", "", "x10E3/uL", "3.4 - 10.8"],
                ["RBC", "4.33", "", "x10E6/uL", "3.77 - 5.28"]
            ]
        }
    ]
}

Extract ALL tables on this page. Use empty string "" for empty cells.
JSON:"""

        try:
            from ..medgemma.client import create_client
            client = create_client({})
            host = getattr(client, 'host', 'http://localhost:11434')
            model = getattr(client, '_model_name', 'MedAIBase/MedGemma1.5:4b-it-q8_0')

            payload = {
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {
                    "num_predict": 2000,
                    "temperature": 0.1,
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{host}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    if response.status != 200:
                        return []

                    data = await response.json()
                    response_text = data.get('response', '')

            result = self._parse_json_response(response_text)

            if not result or 'tables' not in result:
                return []

            extracted = []
            for i, table_data in enumerate(result['tables']):
                headers = table_data.get('headers', [])
                rows = table_data.get('rows', [])

                if headers and rows:
                    extracted.append(ExtractedTable(
                        page_number=page_num,
                        table_index=i,
                        headers=headers,
                        rows=rows,
                        confidence=0.80,
                        extraction_method="vision"
                    ))

            return extracted

        except Exception as e:
            self.logger.error(f"Vision extraction for page {page_num} failed: {e}")
            return []

    def _clean_table_data(self, table_data: List[List]) -> List[List[str]]:
        """Clean and normalize table data."""
        if not table_data:
            return []

        cleaned = []
        for row in table_data:
            if row is None:
                continue
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append('')
                else:
                    # Clean cell text
                    text = str(cell).strip()
                    # Remove excessive whitespace
                    text = re.sub(r'\s+', ' ', text)
                    cleaned_row.append(text)
            # Skip completely empty rows
            if any(cell for cell in cleaned_row):
                cleaned.append(cleaned_row)

        return cleaned

    def _calculate_table_confidence(
        self,
        headers: List[str],
        rows: List[List[str]]
    ) -> float:
        """
        Calculate confidence score for extracted table.

        Based on:
        - Consistent column count
        - Presence of lab-related keywords
        - Numeric values in expected positions
        """
        if not headers or not rows:
            return 0.0

        score = 0.5  # Base score

        # Check header keywords
        lab_keywords = ['test', 'result', 'value', 'unit', 'reference', 'range', 'flag', 'normal']
        header_text = ' '.join(headers).lower()
        if any(kw in header_text for kw in lab_keywords):
            score += 0.2

        # Check column consistency
        header_count = len(headers)
        consistent_rows = sum(1 for row in rows if len(row) == header_count)
        consistency = consistent_rows / len(rows) if rows else 0
        score += consistency * 0.15

        # Check for numeric values (lab reports have numbers)
        numeric_pattern = re.compile(r'\d+\.?\d*')
        has_numbers = any(
            numeric_pattern.search(cell)
            for row in rows
            for cell in row
        )
        if has_numbers:
            score += 0.15

        return min(score, 1.0)

    def _parse_json_response(self, text: str) -> Optional[Dict]:
        """Parse JSON from model response with repair fallback."""
        if not text or not text.strip():
            return None

        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try json_repair
        try:
            repaired = repair_json(text, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            pass

        # Try extracting JSON block
        try:
            start = text.find('{')
            if start != -1:
                depth = 0
                for i, char in enumerate(text[start:], start=start):
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            json_str = text[start:i + 1]
                            return json.loads(json_str)
        except Exception:
            pass

        return None

    async def extract_tables_async(
        self,
        pdf_path: Path,
        pages: Optional[List[int]] = None
    ) -> List[ExtractedTable]:
        """
        Async version of table extraction.

        Runs CPU-bound extraction in thread pool.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _pdf_executor,
            self.extract_tables,
            pdf_path,
            pages
        )
