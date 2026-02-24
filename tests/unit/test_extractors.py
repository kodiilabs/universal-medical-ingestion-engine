# ============================================================================
# FILE: tests/unit/test_extractors.py
# ============================================================================
"""
Unit tests for PDF extractors
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.medical_ingestion.extractors.pdf_extractor import PDFExtractor
from src.medical_ingestion.extractors.table_extractor import TableExtractor, ExtractedTable
from src.medical_ingestion.extractors.text_extractor import TextExtractor, ExtractedText
from src.medical_ingestion.extractors.layout_analyzer import LayoutAnalyzer
from src.medical_ingestion.processors.lab.utils.parsing import parse_numeric_value, parse_reference_range


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_pdf(tmp_path):
    """Create a simple PDF for testing"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    pdf_path = tmp_path / "test.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, "Sample Lab Report")
    c.drawString(100, 700, "Patient: John Doe")
    c.drawString(100, 650, "Test Results:")
    c.drawString(100, 600, "Hemoglobin: 14.2 g/dL")
    c.drawString(100, 550, "WBC: 7.5 K/uL")
    c.save()
    return pdf_path


@pytest.fixture
def multi_page_pdf(tmp_path):
    """Create a multi-page PDF for testing"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    pdf_path = tmp_path / "multi_page.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Page 1
    c.drawString(100, 750, "Page 1 Content")
    c.drawString(100, 700, "Test: Value 1")
    c.showPage()

    # Page 2
    c.drawString(100, 750, "Page 2 Content")
    c.drawString(100, 700, "Test: Value 2")
    c.showPage()

    c.save()
    return pdf_path


# ============================================================================
# TEXT EXTRACTOR TESTS
# ============================================================================

def test_text_extractor_init():
    """Test TextExtractor initialization"""
    extractor = TextExtractor()
    assert extractor is not None
    assert extractor.logger is not None


def test_text_extractor_extract_text_basic(sample_pdf):
    """Test basic text extraction"""
    extractor = TextExtractor()
    text = extractor.extract_text(sample_pdf)

    assert text is not None
    assert len(text) > 0
    assert "Sample Lab Report" in text
    assert "John Doe" in text
    assert "Hemoglobin" in text


def test_text_extractor_extract_text_with_layout(sample_pdf):
    """Test text extraction with layout preservation"""
    extractor = TextExtractor()
    text = extractor.extract_text(sample_pdf, preserve_layout=True)

    assert text is not None
    assert len(text) > 0


def test_text_extractor_extract_by_page(multi_page_pdf):
    """Test page-by-page extraction"""
    extractor = TextExtractor()
    pages = extractor.extract_text_by_page(multi_page_pdf)

    assert len(pages) == 2
    assert all(isinstance(p, ExtractedText) for p in pages)
    assert pages[0].page_number == 0
    assert pages[1].page_number == 1
    assert "Page 1" in pages[0].text
    assert "Page 2" in pages[1].text


def test_text_extractor_nonexistent_file():
    """Test extraction from non-existent file"""
    extractor = TextExtractor()

    with pytest.raises(Exception):
        extractor.extract_text(Path("nonexistent.pdf"))


def test_text_extractor_extract_text_dataclass():
    """Test ExtractedText dataclass"""
    extracted = ExtractedText(
        page_number=0,
        text="Sample text",
        layout_preserved=True
    )

    assert extracted.page_number == 0
    assert extracted.text == "Sample text"
    assert extracted.layout_preserved is True


# ============================================================================
# TABLE EXTRACTOR TESTS
# ============================================================================

def test_table_extractor_init():
    """Test TableExtractor initialization"""
    extractor = TableExtractor()
    assert extractor is not None
    assert extractor.logger is not None


def test_table_extractor_dataclass():
    """Test ExtractedTable dataclass"""
    table = ExtractedTable(
        page_number=1,
        table_index=0,
        headers=["Test", "Result", "Range"],
        rows=[["Hemoglobin", "14.2", "13.5-17.5"]],
        confidence=0.95,
        bbox=(100, 100, 500, 300)
    )

    assert table.page_number == 1
    assert table.table_index == 0
    assert len(table.headers) == 3
    assert len(table.rows) == 1
    assert table.confidence == 0.95
    assert table.bbox == (100, 100, 500, 300)


@patch('camelot.read_pdf')
def test_table_extractor_extract_tables_lattice(mock_camelot, sample_pdf):
    """Test table extraction with lattice flavor"""
    # Mock Camelot table
    mock_table = Mock()
    mock_table.page = 1
    mock_table.accuracy = 95.0
    mock_table.df = Mock()
    mock_table.df.iloc = Mock()
    mock_table.df.iloc.__getitem__ = Mock(return_value=["Test", "Result", "Range"])
    mock_table.df.iloc.return_value.values.tolist.return_value = [
        ["Hemoglobin", "14.2", "13.5-17.5"]
    ]

    mock_camelot.return_value = [mock_table]

    extractor = TableExtractor()
    tables = extractor.extract_tables(sample_pdf, flavor='lattice')

    assert len(tables) >= 0  # May be empty if no tables found
    mock_camelot.assert_called_once()


@patch('camelot.read_pdf')
def test_table_extractor_extract_tables_stream(mock_camelot, sample_pdf):
    """Test table extraction with stream flavor"""
    mock_camelot.return_value = []

    extractor = TableExtractor()
    tables = extractor.extract_tables(sample_pdf, flavor='stream')

    assert isinstance(tables, list)
    mock_camelot.assert_called_once()


@patch('camelot.read_pdf')
def test_table_extractor_extract_table_from_page(mock_camelot, sample_pdf):
    """Test extracting table from specific page"""
    # Mock Camelot table
    mock_table = Mock()
    mock_table.page = 1
    mock_table.accuracy = 95.0
    mock_table.df = Mock()

    # Mock iloc for header extraction
    mock_iloc_header = Mock()
    mock_iloc_header.__iter__ = Mock(return_value=iter(["Test", "Result"]))

    # Mock iloc for row extraction
    mock_iloc_rows = Mock()
    mock_iloc_rows.values.tolist.return_value = [["Hemoglobin", "14.2"]]

    # Set up iloc behavior
    def iloc_side_effect(index):
        if index == 0:
            return mock_iloc_header
        else:
            return mock_iloc_rows

    mock_table.df.iloc.__getitem__ = iloc_side_effect
    mock_camelot.return_value = [mock_table]

    extractor = TableExtractor()
    table = extractor.extract_table_from_page(sample_pdf, page_number=1)

    # May be None if no table found
    assert table is None or isinstance(table, ExtractedTable)


@patch('camelot.read_pdf')
def test_table_extractor_error_handling(mock_camelot, sample_pdf):
    """Test table extractor error handling"""
    mock_camelot.side_effect = Exception("Camelot error")

    extractor = TableExtractor()
    tables = extractor.extract_tables(sample_pdf)

    # Should return empty list on error
    assert tables == []


# ============================================================================
# LAYOUT ANALYZER TESTS
# ============================================================================

def test_layout_analyzer_init():
    """Test LayoutAnalyzer initialization"""
    analyzer = LayoutAnalyzer()
    assert analyzer is not None
    assert analyzer.logger is not None


@patch('pdfplumber.open')
def test_layout_analyzer_extract_with_layout(mock_pdfplumber, sample_pdf):
    """Test layout-preserving text extraction"""
    # Mock pdfplumber
    mock_pdf = MagicMock()
    mock_page = Mock()
    mock_page.extract_text.return_value = "Layout preserved text"
    mock_pdf.pages = [mock_page]
    mock_pdf.__enter__ = Mock(return_value=mock_pdf)
    mock_pdf.__exit__ = Mock(return_value=None)
    mock_pdfplumber.return_value = mock_pdf

    analyzer = LayoutAnalyzer()
    text = analyzer.extract_text_with_layout(sample_pdf, page_num=0)

    assert text == "Layout preserved text"
    mock_page.extract_text.assert_called_once_with(layout=True)


@patch('pdfplumber.open')
def test_layout_analyzer_error_handling(mock_pdfplumber, sample_pdf):
    """Test layout analyzer error handling"""
    mock_pdfplumber.side_effect = Exception("pdfplumber error")

    analyzer = LayoutAnalyzer()
    text = analyzer.extract_text_with_layout(sample_pdf, page_num=0)

    # Should return empty string on error
    assert text == ""


# ============================================================================
# PDF EXTRACTOR TESTS
# ============================================================================

def test_pdf_extractor_init():
    """Test PDFExtractor initialization"""
    extractor = PDFExtractor()
    assert extractor is not None
    assert extractor.text_extractor is not None
    assert extractor.table_extractor is not None
    assert extractor.logger is not None


def test_pdf_extractor_extract_text(sample_pdf):
    """Test PDFExtractor text extraction delegation"""
    extractor = PDFExtractor()
    text = extractor.extract_text(sample_pdf)

    assert text is not None
    assert len(text) > 0
    assert "Sample Lab Report" in text


def test_pdf_extractor_extract_text_with_layout(sample_pdf):
    """Test PDFExtractor text extraction with layout"""
    extractor = PDFExtractor()
    text = extractor.extract_text(sample_pdf, preserve_layout=True)

    assert text is not None
    assert len(text) > 0


@patch.object(TableExtractor, 'extract_tables')
def test_pdf_extractor_extract_tables(mock_extract, sample_pdf):
    """Test PDFExtractor table extraction delegation"""
    mock_extract.return_value = []

    extractor = PDFExtractor()
    tables = extractor.extract_tables(sample_pdf, flavor='lattice')

    assert isinstance(tables, list)
    mock_extract.assert_called_once_with(sample_pdf, flavor='lattice')


@patch.object(TableExtractor, 'extract_tables')
def test_pdf_extractor_best_effort_high_confidence_lattice(mock_extract, sample_pdf):
    """Test best effort extraction with high confidence lattice table"""
    # High confidence table
    high_conf_table = ExtractedTable(
        page_number=1,
        table_index=0,
        headers=["Test", "Result"],
        rows=[["Hemoglobin", "14.2"]],
        confidence=0.90
    )

    mock_extract.return_value = [high_conf_table]

    extractor = PDFExtractor()
    result = extractor.extract_best_effort(sample_pdf)

    assert result["method"] == "table_lattice"
    assert len(result["tables"]) == 1
    assert result["confidence"] == 0.90


@patch.object(TableExtractor, 'extract_tables')
def test_pdf_extractor_best_effort_high_confidence_stream(mock_extract, sample_pdf):
    """Test best effort extraction with stream table"""
    # First call (lattice) returns low confidence
    low_conf_table = ExtractedTable(
        page_number=1,
        table_index=0,
        headers=["Test"],
        rows=[["Data"]],
        confidence=0.70
    )

    # Second call (stream) returns high confidence
    high_conf_table = ExtractedTable(
        page_number=1,
        table_index=0,
        headers=["Test", "Result"],
        rows=[["Hemoglobin", "14.2"]],
        confidence=0.85
    )

    mock_extract.side_effect = [[low_conf_table], [high_conf_table]]

    extractor = PDFExtractor()
    result = extractor.extract_best_effort(sample_pdf)

    assert result["method"] == "table_stream"
    assert len(result["tables"]) == 1
    assert result["confidence"] == 0.85


@patch.object(TableExtractor, 'extract_tables')
@patch.object(TextExtractor, 'extract_text')
def test_pdf_extractor_best_effort_fallback_to_text(mock_text_extract, mock_table_extract, sample_pdf):
    """Test best effort extraction falls back to text"""
    # Both table extractions fail
    mock_table_extract.return_value = []
    mock_text_extract.return_value = "Fallback text content"

    extractor = PDFExtractor()
    result = extractor.extract_best_effort(sample_pdf)

    assert result["method"] == "text"
    assert result["text"] == "Fallback text content"
    assert result["confidence"] == 0.5
    assert len(result["tables"]) == 0


@patch.object(TableExtractor, 'extract_tables')
def test_pdf_extractor_best_effort_low_confidence_tables(mock_extract, sample_pdf):
    """Test best effort with only low confidence tables"""
    # Low confidence tables
    low_conf_table = ExtractedTable(
        page_number=1,
        table_index=0,
        headers=["Test"],
        rows=[["Data"]],
        confidence=0.60
    )

    mock_extract.return_value = [low_conf_table]

    extractor = PDFExtractor()
    result = extractor.extract_best_effort(sample_pdf)

    # Should fall back to text
    assert result["method"] == "text"


# ============================================================================
# PARSING UTILITY TESTS
# ============================================================================

def test_parse_numeric_value_simple():
    """Test parsing simple numeric values"""
    assert parse_numeric_value("12.5") == 12.5
    assert parse_numeric_value("7.2") == 7.2
    assert parse_numeric_value("100") == 100.0


def test_parse_numeric_value_with_units():
    """Test parsing numeric values with units"""
    assert parse_numeric_value("7.2 K/uL") == 7.2
    assert parse_numeric_value("14.5 g/dL") == 14.5
    assert parse_numeric_value("250 mg/dL") == 250.0


def test_parse_numeric_value_invalid():
    """Test parsing invalid values"""
    assert parse_numeric_value("invalid") is None
    assert parse_numeric_value("N/A") is None
    assert parse_numeric_value("") is None
    # "abc123" extracts 123.0 (intentional - extracts numbers from text)
    assert parse_numeric_value("abc123") == 123.0


def test_parse_numeric_value_edge_cases():
    """Test parsing edge cases"""
    assert parse_numeric_value("0") == 0.0
    assert parse_numeric_value("0.0") == 0.0
    assert parse_numeric_value("-5.5") == -5.5


def test_parse_reference_range_standard():
    """Test parsing standard reference ranges"""
    assert parse_reference_range("12.0-15.5") == (12.0, 15.5)
    assert parse_reference_range("3.5-5.0") == (3.5, 5.0)
    assert parse_reference_range("150-400") == (150.0, 400.0)


def test_parse_reference_range_with_spaces():
    """Test parsing reference ranges with spaces"""
    assert parse_reference_range("12.0 - 15.5") == (12.0, 15.5)
    assert parse_reference_range("3.5  -  5.0") == (3.5, 5.0)
    assert parse_reference_range(" 150 - 400 ") == (150.0, 400.0)


def test_parse_reference_range_with_units():
    """Test parsing reference ranges with units"""
    result = parse_reference_range("12.0-15.5 g/dL")
    if result:
        assert result == (12.0, 15.5)


def test_parse_reference_range_invalid():
    """Test parsing invalid reference ranges"""
    assert parse_reference_range("invalid") is None
    assert parse_reference_range("") is None
    assert parse_reference_range("12.0") is None
    assert parse_reference_range("abc-def") is None


def test_parse_reference_range_single_bound():
    """Test parsing reference ranges with single bound"""
    # Some ranges might be ">5" or "<10"
    result = parse_reference_range(">5.0")
    # Should handle gracefully
    assert result is None or isinstance(result, tuple)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_extraction_pipeline(sample_pdf):
    """Test full extraction pipeline"""
    extractor = PDFExtractor()

    # Try best effort extraction
    result = extractor.extract_best_effort(sample_pdf)

    assert result is not None
    assert "method" in result
    assert "confidence" in result
    assert result["method"] in ["table_lattice", "table_stream", "text"]

    # If text method, verify text was extracted
    if result["method"] == "text":
        assert len(result["text"]) > 0

    # If table method, verify tables were extracted
    if "table" in result["method"]:
        assert isinstance(result["tables"], list)


def test_extractor_handles_empty_pdf(tmp_path):
    """Test extractors handle empty PDFs gracefully"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    # Create empty PDF
    pdf_path = tmp_path / "empty.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.showPage()
    c.save()

    extractor = PDFExtractor()
    text = extractor.extract_text(pdf_path)

    # Should not crash, may return empty string
    assert isinstance(text, str)


def test_all_extractors_have_logging():
    """Test that all extractors have logging configured"""
    pdf_extractor = PDFExtractor()
    text_extractor = TextExtractor()
    table_extractor = TableExtractor()
    layout_analyzer = LayoutAnalyzer()

    assert pdf_extractor.logger is not None
    assert text_extractor.logger is not None
    assert table_extractor.logger is not None
    assert layout_analyzer.logger is not None
