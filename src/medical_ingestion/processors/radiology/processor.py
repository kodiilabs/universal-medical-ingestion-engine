# ============================================================================
# src/medical_ingestion/processors/radiology/processor.py
# ============================================================================
"""
Radiology Report Processor

Extracts structured data from radiology/imaging reports:
- Impression/Conclusion
- Findings
- Comparison to prior studies
- Recommendations
- Critical findings
"""

from typing import Dict, Any, List
import logging

from ...core.context.processing_context import ProcessingContext
from ...core.vector_store import get_vector_store
from ..base_processor import BaseProcessor
from ...medgemma.client import create_client


class RadiologyProcessor(BaseProcessor):
    """
    Processes radiology imaging reports.

    Uses MedGemma to extract:
    - Impression/conclusion
    - Key findings
    - Comparison statements
    - Recommendations
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.medgemma = create_client(self.config)
        self.logger = logging.getLogger(__name__)
        self._vector_store = None

    @property
    def vector_store(self):
        """Get shared vector store (singleton)."""
        if self._vector_store is None:
            self._vector_store = get_vector_store(self.config)
        return self._vector_store

    def get_name(self) -> str:
        return "RadiologyProcessor"

    def _get_agents(self) -> List:
        """
        Radiology processor uses direct MedGemma calls rather than agents.
        Returns empty list for compatibility with base class.
        """
        return []

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Process radiology report.

        Args:
            context: Processing context with raw text

        Returns:
            Updated context with extracted findings
        """
        self.logger.info(f"Processing radiology report: {context.document_path}")

        # Extract text if not already available
        if not context.raw_text:
            from ...extractors.text_extractor import TextExtractor
            text_extractor = TextExtractor()
            context.raw_text = text_extractor.extract_text(context.document_path)

        # Extract sections
        sections = self._extract_sections(context.raw_text)

        # Extract impression
        impression = await self._extract_impression(sections, context)
        context.clinical_summary = impression

        # Extract findings
        findings = await self._extract_findings(sections, context)
        context.sections['findings'] = findings

        # Extract comparison
        comparison = self._extract_comparison(sections)
        context.sections['comparison'] = comparison

        # Extract recommendations
        recommendations = await self._extract_recommendations(sections, context)
        context.sections['recommendations'] = recommendations

        # Check for critical findings
        critical = await self._check_critical_findings(context)
        if critical:
            context.add_critical_finding("Critical findings detected in radiology report")

        # Store in vector store for future similar document matching
        confidence = 0.85
        if self.config.get('use_consensus_extraction', False):
            min_confidence = self.config.get('vector_store_min_confidence', 0.85)
            if confidence >= min_confidence:
                await self._store_in_vector_store(context, confidence)

        self.logger.info(f"Radiology processing complete")

        return {
            "success": True,
            "agent_results": [],
            "extracted_values": 0,
            "requires_review": bool(critical),
            "confidence": confidence
        }

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract standard radiology report sections.

        Args:
            text: Raw report text

        Returns:
            Dict of section name to content
        """
        import re

        sections = {
            'impression': '',
            'findings': '',
            'technique': '',
            'indication': '',
            'comparison': '',
            'clinical_information': ''
        }

        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Section headers to look for (order matters - more specific first)
        section_headers = [
            'IMPRESSION', 'CONCLUSION', 'SUMMARY',
            'FINDINGS', 'OBSERVATIONS',
            'TECHNIQUE', 'PROTOCOL',
            'INDICATION', 'CLINICAL HISTORY', 'CLINICAL INFORMATION', 'HISTORY',
            'COMPARISON', 'PRIOR', 'PREVIOUS',
            'CONTRAST', 'RECOMMENDATION'
        ]

        # Build a regex to find all section boundaries
        header_pattern = r'\n(' + '|'.join(section_headers) + r')[ :]*\n?'

        # Find all section starts
        matches = list(re.finditer(header_pattern, text, re.IGNORECASE))

        # Also check start of document for headers
        start_match = re.match(r'^(' + '|'.join(section_headers) + r')[ :]*\n?', text, re.IGNORECASE)
        if start_match:
            # Create a fake match object
            class FakeMatch:
                def __init__(self, m):
                    self._start = m.start()
                    self._end = m.end()
                    self._group1 = m.group(1)
                def start(self): return self._start
                def end(self): return self._end
                def group(self, n): return self._group1 if n == 1 else None
            matches.insert(0, FakeMatch(start_match))

        # Extract content between headers
        for i, match in enumerate(matches):
            header = match.group(1).upper()
            start = match.end()

            # End is either next section or end of text
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            content = text[start:end].strip()

            # Map header to section key
            if header in ('IMPRESSION', 'CONCLUSION', 'SUMMARY'):
                if not sections['impression']:
                    sections['impression'] = content
            elif header in ('FINDINGS', 'OBSERVATIONS'):
                if not sections['findings']:
                    sections['findings'] = content
            elif header in ('TECHNIQUE', 'PROTOCOL'):
                if not sections['technique']:
                    sections['technique'] = content
            elif header in ('INDICATION', 'CLINICAL HISTORY', 'CLINICAL INFORMATION', 'HISTORY'):
                if not sections['indication'] and not sections['clinical_information']:
                    sections['clinical_information'] = content
            elif header in ('COMPARISON', 'PRIOR', 'PREVIOUS'):
                if not sections['comparison']:
                    sections['comparison'] = content

        # Log what was extracted
        for section, content in sections.items():
            if content:
                self.logger.debug(f"Extracted {section}: {len(content)} chars")

        return sections

    def _is_valid_text(self, text: str) -> bool:
        """
        Check if text is valid (not garbage/noise).

        Returns False for binary-looking strings, repeated patterns, etc.
        """
        if not text or len(text) < 5:
            return False

        # Check for binary-looking output (mostly 0s and 1s)
        if text.replace('0', '').replace('1', '').strip() == '':
            return False

        # Check for repeated backticks or other noise patterns
        if text.count('`') > len(text) * 0.3:  # More than 30% backticks
            return False

        # Check for reasonable word structure
        words = text.split()
        if len(words) < 2:
            return False

        # Check that most "words" have reasonable length
        reasonable_words = sum(1 for w in words if 2 <= len(w) <= 20)
        if reasonable_words < len(words) * 0.5:
            return False

        return True

    async def _extract_impression(
        self,
        sections: Dict[str, str],
        context: ProcessingContext
    ) -> str:
        """
        Extract impression/conclusion using MedGemma.

        Args:
            sections: Extracted sections
            context: Processing context

        Returns:
            Impression text
        """
        # If impression section exists and is valid, return it
        if sections.get('impression') and self._is_valid_text(sections['impression']):
            return sections['impression']

        # Try MedGemma to extract
        prompt = f"""Extract the impression/conclusion from this radiology report.

Report text:
{context.raw_text[:1000]}

Return only the impression/conclusion in 1-2 sentences."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1
            )
            result = response['text'].strip()

            # Validate MedGemma output
            if self._is_valid_text(result):
                return result
            else:
                self.logger.warning(f"MedGemma returned invalid impression: {result[:50]}...")
        except Exception as e:
            self.logger.warning(f"MedGemma impression extraction failed: {e}")

        # Fallback: extract from raw text
        return self._fallback_extract_impression(context.raw_text)

    def _fallback_extract_impression(self, raw_text: str) -> str:
        """
        Fallback impression extraction using simple text analysis.

        Looks for common impression patterns in the text.
        """
        import re

        text = raw_text.replace('\r\n', '\n').replace('\r', '\n')

        # Try to find impression section with simple regex
        patterns = [
            r'(?:IMPRESSION|CONCLUSION|SUMMARY)[:\s]*\n?(.*?)(?=\n[A-Z]{3,}|\nPage|\Z)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result = match.group(1).strip()
                # Get first 2-3 sentences
                sentences = re.split(r'[.!?]+', result)
                if sentences:
                    return '. '.join(s.strip() for s in sentences[:3] if s.strip()) + '.'

        return "Impression not found in document"

    def _fallback_extract_findings(self, raw_text: str) -> str:
        """
        Fallback findings extraction using simple text analysis.
        """
        import re

        text = raw_text.replace('\r\n', '\n').replace('\r', '\n')

        # Try to find findings section
        match = re.search(
            r'(?:FINDINGS|OBSERVATIONS)[:\s]*\n?(.*?)(?=\n(?:IMPRESSION|CONCLUSION|RECOMMENDATION|Page)|\Z)',
            text,
            re.IGNORECASE | re.DOTALL
        )

        if match:
            return match.group(1).strip()

        return "Findings not found in document"

    async def _extract_findings(
        self,
        sections: Dict[str, str],
        context: ProcessingContext
    ) -> str:
        """
        Extract key findings using MedGemma.

        Args:
            sections: Extracted sections
            context: Processing context

        Returns:
            Findings text
        """
        # If findings section exists and is valid, return it
        if sections.get('findings') and self._is_valid_text(sections['findings']):
            return sections['findings']

        # Try MedGemma
        prompt = f"""List the key findings from this radiology report.

Report text:
{context.raw_text[:1000]}

Return bullet points of key findings."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=300,
                temperature=0.1
            )
            result = response['text'].strip()

            if self._is_valid_text(result):
                return result
            else:
                self.logger.warning(f"MedGemma returned invalid findings: {result[:50]}...")
        except Exception as e:
            self.logger.warning(f"MedGemma findings extraction failed: {e}")

        # Fallback to regex extraction
        return self._fallback_extract_findings(context.raw_text)

    def _extract_comparison(self, sections: Dict[str, str]) -> str:
        """
        Extract comparison to prior studies.

        Args:
            sections: Extracted sections

        Returns:
            Comparison text
        """
        return sections.get('comparison', '')

    async def _extract_recommendations(
        self,
        sections: Dict[str, str],
        context: ProcessingContext
    ) -> str:
        """
        Extract recommendations using MedGemma.

        Args:
            sections: Extracted sections
            context: Processing context

        Returns:
            Recommendations text
        """
        # First try to find recommendations in the text via regex
        import re

        text = context.raw_text.replace('\r\n', '\n').replace('\r', '\n')

        # Look for recommendation patterns
        rec_patterns = [
            r'(?:RECOMMEND|RECOMMENDATION)[S]?[:\s]*(.*?)(?=\n[A-Z]{3,}|\Z)',
            r'(?:is recommended|recommend(?:ed)?)[:\s]*(.*?)(?=\.|$)',
            r'(?:follow[- ]?up|further)[:\s]*(.*?)(?=\.|$)',
        ]

        for pattern in rec_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                result = match.group(1).strip()
                if self._is_valid_text(result):
                    return result

        # Try MedGemma
        prompt = f"""Extract any recommendations from this radiology report.

Report text:
{context.raw_text[:1000]}

Return recommendations or "None" if no recommendations."""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.1
            )
            result = response['text'].strip()

            if self._is_valid_text(result) and result.lower() != 'none':
                return result
        except Exception as e:
            self.logger.warning(f"MedGemma recommendations extraction failed: {e}")

        return ""

    async def _check_critical_findings(
        self,
        context: ProcessingContext
    ) -> bool:
        """
        Check for critical findings using MedGemma.

        Args:
            context: Processing context

        Returns:
            True if critical findings detected
        """
        prompt = f"""Does this radiology report contain any critical or urgent findings?

Report text:
{context.raw_text[:1000]}

Answer: YES or NO"""

        try:
            response = await self.medgemma.generate(
                prompt=prompt,
                max_tokens=10,
                temperature=0.1
            )
            return 'YES' in response['text'].upper()
        except Exception as e:
            self.logger.warning(f"Critical findings check failed: {e}")
            return False

    async def _store_in_vector_store(
        self,
        context: ProcessingContext,
        confidence: float
    ):
        """Store successful extraction in vector store for future reference."""
        try:
            extracted_values = {
                'impression': context.clinical_summary or '',
                'findings': context.sections.get('findings', ''),
                'comparison': context.sections.get('comparison', ''),
                'recommendations': context.sections.get('recommendations', ''),
            }

            await self.vector_store.store(
                text=context.raw_text[:2000],
                extracted_values=extracted_values,
                template_id='radiology',
                source_file=str(context.document_path),
                confidence=confidence
            )
            self.logger.info(f"Stored radiology extraction in vector store")
        except Exception as e:
            self.logger.debug(f"Failed to store in vector store: {e}")
