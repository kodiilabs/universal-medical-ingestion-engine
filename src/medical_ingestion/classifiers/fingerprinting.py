# ============================================================================
# FILE 1: src/medical_ingestion/classifiers/fingerprinting.py
# ============================================================================
"""
Document Fingerprinting - Structural Analysis

Analyzes document structure without relying on content:
- Layout patterns (headers, sections, tables)
- Text density and distribution
- Page structure
- Formatting signatures

This provides fast, deterministic classification hints.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
import re


class DocumentFingerprinter:
    """
    Analyzes document structure for classification hints.
    
    Uses layout analysis, not just keywords, to identify document types.
    This is faster than AI classification and more reliable than
    keyword matching alone.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze(self, text: str, page_count: int = 1) -> Dict[str, Any]:
        """
        Generate structural fingerprint of document.
        
        Args:
            text: Extracted text from PDF
            page_count: Number of pages
            
        Returns:
            {
                "has_table_structure": bool,
                "has_sectioned_narrative": bool,
                "text_density": float,
                "header_count": int,
                "numeric_density": float,
                "layout_type": "tabular" | "narrative" | "mixed",
                "structural_hints": List[str]
            }
        """
        fingerprint = {
            "has_table_structure": self._detect_table_structure(text),
            "has_sectioned_narrative": self._detect_sections(text),
            "text_density": self._calculate_text_density(text, page_count),
            "header_count": self._count_headers(text),
            "numeric_density": self._calculate_numeric_density(text),
            "layout_type": "unknown",
            "structural_hints": []
        }
        
        # Determine layout type
        fingerprint["layout_type"] = self._determine_layout_type(fingerprint)
        
        # Generate structural hints
        fingerprint["structural_hints"] = self._generate_hints(fingerprint, text)
        
        return fingerprint
    
    def _detect_table_structure(self, text: str) -> bool:
        """
        Detect if document has tabular layout.
        
        Indicators:
        - Multiple lines with consistent column structure
        - Whitespace-separated values
        - Tab characters
        """
        lines = text.split('\n')
        
        # Count lines that look like table rows
        table_like_lines = 0
        
        for line in lines:
            # Skip very short lines
            if len(line.strip()) < 10:
                continue
            
            # Check for tab-separated values
            if '\t' in line:
                table_like_lines += 1
                continue
            
            # Check for multiple whitespace-separated values
            # (at least 3 "columns")
            parts = re.split(r'\s{2,}', line.strip())
            if len(parts) >= 3:
                table_like_lines += 1
        
        # If >30% of lines are table-like, it's a table document
        non_empty_lines = len([l for l in lines if l.strip()])
        if non_empty_lines > 0:
            table_ratio = table_like_lines / non_empty_lines
            return table_ratio > 0.3
        
        return False
    
    def _detect_sections(self, text: str) -> bool:
        """
        Detect if document has narrative sections.
        
        Indicators:
        - Section headers (ALL CAPS, followed by content)
        - Consistent section structure
        """
        # Look for common section headers
        section_patterns = [
            r'^[A-Z][A-Z\s]{3,}:',  # ALL CAPS header with colon
            r'^[A-Z][A-Z\s]{3,}\n',  # ALL CAPS header on own line
            r'^IMPRESSION:',
            r'^FINDINGS?:',
            r'^INDICATION:',
            r'^DIAGNOSIS:',
            r'^GROSS:',
            r'^MICROSCOPIC:',
            r'^TECHNIQUE:',
            r'^COMPARISON:',
            r'^CONCLUSION:',
        ]
        
        section_count = 0
        
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            section_count += len(matches)
        
        # If document has 2+ sections, it's sectioned narrative
        return section_count >= 2
    
    def _calculate_text_density(self, text: str, page_count: int) -> float:
        """
        Calculate text density (words per page).
        
        Lab reports: Low density (~200-500 words/page)
        Radiology/Pathology: Medium-high density (~500-1000 words/page)
        """
        words = len(text.split())
        if page_count > 0:
            return words / page_count
        return 0.0
    
    def _count_headers(self, text: str) -> int:
        """Count section headers in document"""
        # Simple heuristic: all-caps lines
        lines = text.split('\n')
        header_count = 0
        
        for line in lines:
            stripped = line.strip()
            # Must be 3-50 chars, all uppercase, optionally with colon
            if 3 <= len(stripped) <= 50:
                if stripped.isupper() or stripped.rstrip(':').isupper():
                    header_count += 1
        
        return header_count
    
    def _calculate_numeric_density(self, text: str) -> float:
        """
        Calculate ratio of numeric characters to total.
        
        Lab reports: High numeric density (lots of values)
        Narrative reports: Low numeric density (mostly text)
        """
        if not text:
            return 0.0
        
        # Count digits
        digit_count = sum(1 for c in text if c.isdigit())
        
        # Count non-whitespace characters
        non_whitespace = sum(1 for c in text if not c.isspace())
        
        if non_whitespace > 0:
            return digit_count / non_whitespace
        
        return 0.0
    
    def _determine_layout_type(self, fingerprint: Dict) -> str:
        """
        Determine overall layout type from fingerprint.
        
        Types:
        - tabular: Structured data in tables (labs)
        - narrative: Prose with sections (radiology, pathology)
        - mixed: Both tables and narrative
        """
        has_tables = fingerprint['has_table_structure']
        has_sections = fingerprint['has_sectioned_narrative']
        numeric_density = fingerprint['numeric_density']
        
        if has_tables and not has_sections and numeric_density > 0.15:
            return "tabular"
        elif has_sections and not has_tables and numeric_density < 0.10:
            return "narrative"
        elif has_tables and has_sections:
            return "mixed"
        else:
            return "unknown"
    
    def _generate_hints(self, fingerprint: Dict, text: str) -> List[str]:
        """
        Generate classification hints based on structural analysis.
        
        Returns list of likely document types.
        """
        hints = []
        
        layout_type = fingerprint['layout_type']
        numeric_density = fingerprint['numeric_density']
        text_density = fingerprint['text_density']
        has_sections = fingerprint['has_sectioned_narrative']
        
        # Tabular + high numeric density = likely lab
        if layout_type == "tabular" and numeric_density > 0.15:
            hints.append("lab")
        
        # Narrative + specific sections = radiology or pathology
        if layout_type == "narrative" and has_sections:
            # Check for radiology-specific patterns
            if any(word in text.lower() for word in ['impression', 'findings', 'comparison', 'technique']):
                hints.append("radiology")
            
            # Check for pathology-specific patterns
            if any(word in text.lower() for word in ['diagnosis', 'gross', 'microscopic', 'specimen']):
                hints.append("pathology")
        
        # Low text density + structured = likely prescription
        if text_density < 300 and layout_type != "narrative":
            if any(word in text.lower() for word in ['rx', 'medication', 'dosage', 'refill']):
                hints.append("prescription")
        
        return hints