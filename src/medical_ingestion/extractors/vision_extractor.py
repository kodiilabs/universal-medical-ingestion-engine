# # src/medical_ingestion/extractors/vision_extractor.py
# """
# Vision-Based Document Extraction

# Uses MedGemma's multimodal capabilities to:
# 1. Visual Template Recognition - Identify document vendor/type from appearance
# 2. Visual Structure Analysis - Understand layout without flattening text
# 3. Visual Data Extraction - Extract values directly from document image

# This is the fallback when text extraction produces poor results
# (flattened text, missing structure, scanned documents).

# Ollama API supports images via base64 encoding in the 'images' parameter.
# """

# import base64
# import json
# import logging
# from pathlib import Path
# from typing import Dict, Any, List, Optional, Tuple
# from dataclasses import dataclass, field
# from io import BytesIO

# import pypdfium2
# from PIL import Image

# logger = logging.getLogger(__name__)


# @dataclass
# class VisualExtractionResult:
#     """Result from vision-based extraction."""
#     vendor: Optional[str] = None
#     document_type: Optional[str] = None
#     template_suggestion: Optional[str] = None
#     confidence: float = 0.0
#     extracted_values: List[Dict[str, Any]] = field(default_factory=list)
#     layout_info: Dict[str, Any] = field(default_factory=dict)
#     raw_response: Optional[str] = None
#     method: str = "vision"
#     pages_analyzed: int = 0
#     errors: List[str] = field(default_factory=list)


# class VisionExtractor:
#     """
#     Vision-based document analysis using MedGemma multimodal.

#     Uses the model's ability to "see" documents to:
#     - Recognize vendor logos and formatting
#     - Understand table structure visually
#     - Extract data even when text extraction fails

#     This is especially useful for:
#     - LabCorp, Quest, etc. branded reports
#     - Complex table layouts
#     - Documents with mixed text/graphics
#     """

#     # Image settings for optimal model performance
#     MAX_IMAGE_WIDTH = 1024
#     MAX_IMAGE_HEIGHT = 1024
#     JPEG_QUALITY = 85

#     def __init__(self, medgemma_client=None):
#         """
#         Initialize vision extractor.

#         Args:
#             medgemma_client: Optional pre-configured MedGemma client.
#                              If not provided, will create one when needed.
#         """
#         self._medgemma = medgemma_client
#         self.logger = logging.getLogger(__name__)

#     @property
#     def medgemma(self):
#         """Lazy-load MedGemma client."""
#         if self._medgemma is None:
#             from ..medgemma.client import create_client
#             self._medgemma = create_client({})
#         return self._medgemma

#     def pdf_to_images(
#         self,
#         pdf_path: Path,
#         pages: Optional[List[int]] = None,
#         dpi: int = 150
#     ) -> List[Tuple[int, Image.Image]]:
#         """
#         Convert PDF pages to PIL Images.

#         Args:
#             pdf_path: Path to PDF file
#             pages: Specific pages to convert (0-indexed). None = all pages.
#             dpi: Resolution for rendering

#         Returns:
#             List of (page_number, PIL.Image) tuples
#         """
#         pdf = pypdfium2.PdfDocument(str(pdf_path))
#         images = []

#         page_indices = pages if pages else range(len(pdf))

#         for page_idx in page_indices:
#             if page_idx >= len(pdf):
#                 continue

#             page = pdf[page_idx]

#             # Render at specified DPI
#             scale = dpi / 72  # PDF default is 72 DPI
#             bitmap = page.render(scale=scale)
#             pil_image = bitmap.to_pil()

#             # Resize if too large (for model efficiency)
#             pil_image = self._resize_image(pil_image)

#             images.append((page_idx, pil_image))

#         pdf.close()
#         return images

#     def _resize_image(self, image: Image.Image) -> Image.Image:
#         """Resize image to fit within max dimensions while preserving aspect ratio."""
#         width, height = image.size

#         if width <= self.MAX_IMAGE_WIDTH and height <= self.MAX_IMAGE_HEIGHT:
#             return image

#         ratio = min(self.MAX_IMAGE_WIDTH / width, self.MAX_IMAGE_HEIGHT / height)
#         new_size = (int(width * ratio), int(height * ratio))

#         return image.resize(new_size, Image.Resampling.LANCZOS)

#     def _image_to_base64(self, image: Image.Image) -> str:
#         """Convert PIL Image to base64 string for Ollama API."""
#         buffer = BytesIO()
#         image.save(buffer, format='JPEG', quality=self.JPEG_QUALITY)
#         return base64.b64encode(buffer.getvalue()).decode('utf-8')

#     async def identify_document_visually(
#         self,
#         pdf_path: Path,
#         page: int = 0
#     ) -> Dict[str, Any]:
#         """
#         Identify document type and vendor using visual analysis.

#         This is much more reliable than text pattern matching because:
#         - Recognizes logos and visual branding
#         - Understands layout structure
#         - Works even with scanned/image PDFs

#         Args:
#             pdf_path: Path to PDF file
#             page: Page to analyze (default: first page)

#         Returns:
#             {
#                 "vendor": "LabCorp" | "Quest Diagnostics" | etc.,
#                 "document_type": "lab_report" | "radiology" | etc.,
#                 "subtype": "CBC" | "CMP" | "CD4/CD8" | etc.,
#                 "template_id": suggested template ID,
#                 "confidence": float,
#                 "reasoning": str
#             }
#         """
#         images = self.pdf_to_images(pdf_path, pages=[page])

#         if not images:
#             return {
#                 "vendor": None,
#                 "document_type": "unknown",
#                 "confidence": 0.0,
#                 "error": "Failed to convert PDF to image"
#             }

#         _, image = images[0]
#         image_b64 = self._image_to_base64(image)

#         prompt = """Analyze this medical document image and identify the source.

# CRITICAL INSTRUCTIONS:
# - ONLY report what you can CLEARLY SEE in the image
# - DO NOT GUESS or assume the vendor based on document format
# - If you cannot find a clear vendor name, logo, or company identification, set vendor to "Unknown"
# - Look for EXPLICIT text like company names, addresses, phone numbers, URLs, or logos

# 1. VENDOR: Who produced this document?
#    - Look for: Company name in header/footer, logo, address, phone number, website URL
#    - If NO clear vendor identification is visible, use "Unknown"
#    - DO NOT infer vendor from test types or document format alone

# 2. DOCUMENT TYPE: What kind of medical document is this?
#    Options: lab_report, radiology_report, pathology_report, prescription, clinical_note, other

# 3. SUBTYPE: If it's a lab report, what specific tests are included?
#    Examples: Complete Blood Count, Lipid Panel, Thyroid Panel, etc.

# 4. TEMPLATE: Only suggest a template ID if you positively identified a vendor.
#    If vendor is "Unknown", set template_id to null.

# Return ONLY a JSON object:
# {
#     "vendor": "COMPANY NAME HERE or Unknown",
#     "document_type": "lab_report",
#     "subtype": "Complete Blood Count",
#     "template_id": null,
#     "confidence": 0.0-1.0,
#     "reasoning": "Explain what text/logos you found that identify the vendor, or explain why vendor is Unknown"
# }

# IMPORTANT: High confidence (>0.8) requires VISIBLE vendor name/logo. If guessing, use low confidence (<0.5).

# JSON:"""

#         try:
#             response = await self._generate_with_image(prompt, image_b64)
#             result = self.medgemma.extract_json(response['text'])

#             if result:
#                 # Validate the result - reject if vendor is Unknown or confidence is too low
#                 vendor = result.get('vendor', '')
#                 confidence = result.get('confidence', 0.0)

#                 # Normalize vendor - treat variations of "unknown" as None
#                 if vendor and vendor.lower() in ['unknown', 'none', 'n/a', 'not found', 'not identified', '']:
#                     result['vendor'] = None
#                     result['template_id'] = None
#                     # Cap confidence when vendor is unknown
#                     result['confidence'] = min(confidence, 0.5)
#                     self.logger.info(f"Visual ID: vendor unknown, confidence capped at {result['confidence']:.2f}")

#                 # If confidence is very low, don't trust the identification
#                 elif confidence < 0.6:
#                     self.logger.info(f"Visual ID: low confidence ({confidence:.2f}), treating as unreliable")
#                     result['vendor'] = None
#                     result['template_id'] = None

#                 # Log what was found
#                 self.logger.info(
#                     f"Visual ID result: vendor={result.get('vendor')}, "
#                     f"type={result.get('document_type')}, "
#                     f"confidence={result.get('confidence', 0):.2f}, "
#                     f"reasoning={result.get('reasoning', 'N/A')[:100]}"
#                 )

#                 return result
#             else:
#                 return {
#                     "vendor": None,
#                     "document_type": "unknown",
#                     "confidence": 0.3,
#                     "raw_response": response['text'],
#                     "error": "Failed to parse model response"
#                 }

#         except Exception as e:
#             self.logger.error(f"Visual identification failed: {e}")
#             return {
#                 "vendor": None,
#                 "document_type": "unknown",
#                 "confidence": 0.0,
#                 "error": str(e)
#             }

#     async def extract_lab_values_visually(
#         self,
#         pdf_path: Path,
#         pages: Optional[List[int]] = None
#     ) -> VisualExtractionResult:
#         """
#         Extract lab values directly from document images.

#         Uses the model's vision to read values from tables,
#         even when text extraction produces garbled output.

#         Args:
#             pdf_path: Path to PDF file
#             pages: Pages to analyze (default: all)

#         Returns:
#             VisualExtractionResult with extracted values
#         """
#         result = VisualExtractionResult()

#         images = self.pdf_to_images(pdf_path, pages=pages)

#         if not images:
#             result.errors.append("Failed to convert PDF to images")
#             return result

#         all_values = []

#         for page_idx, image in images:
#             image_b64 = self._image_to_base64(image)

#             prompt = """Extract ALL lab test values from this medical document image.

# For each test result you can see, extract:
# - test_name: The name of the test
# - value: The numeric result
# - unit: The unit of measurement
# - reference_min: Lower bound of reference range (if shown)
# - reference_max: Upper bound of reference range (if shown)
# - flag: "H" for high, "L" for low, null for normal

# Return a JSON object with this structure:
# {
#     "page": 0,
#     "values": [
#         {
#             "test_name": "Hemoglobin",
#             "value": 14.5,
#             "unit": "g/dL",
#             "reference_min": 12.0,
#             "reference_max": 16.0,
#             "flag": null
#         },
#         {
#             "test_name": "WBC",
#             "value": 11.2,
#             "unit": "x10E3/uL",
#             "reference_min": 3.4,
#             "reference_max": 10.8,
#             "flag": "H"
#         }
#     ]
# }

# Extract EVERY test result visible. Use null for missing values.

# JSON:"""

#             try:
#                 response = await self._generate_with_image(prompt, image_b64)
#                 page_result = self.medgemma.extract_json(response['text'])

#                 if page_result and 'values' in page_result:
#                     for val in page_result['values']:
#                         val['source_page'] = page_idx
#                         val['extraction_method'] = 'vision'
#                         all_values.append(val)

#                 result.pages_analyzed += 1

#             except Exception as e:
#                 self.logger.warning(f"Visual extraction failed for page {page_idx}: {e}")
#                 result.errors.append(f"Page {page_idx}: {e}")

#         result.extracted_values = all_values
#         result.confidence = 0.85 if all_values else 0.0

#         self.logger.info(f"Visual extraction: {len(all_values)} values from {result.pages_analyzed} pages")

#         return result

#     async def analyze_layout(
#         self,
#         pdf_path: Path,
#         page: int = 0
#     ) -> Dict[str, Any]:
#         """
#         Analyze document layout structure visually.

#         Identifies:
#         - Header location and content
#         - Table structure (rows, columns)
#         - Section divisions
#         - Key field locations

#         This helps with template creation and matching.
#         """
#         images = self.pdf_to_images(pdf_path, pages=[page])

#         if not images:
#             return {"error": "Failed to convert PDF to image"}

#         _, image = images[0]
#         image_b64 = self._image_to_base64(image)

#         prompt = """Analyze the layout structure of this medical document.

# Describe:
# 1. HEADER: Where is it? What info does it contain?
# 2. PATIENT INFO: Where is patient name, DOB, ID located?
# 3. TABLES: How many tables? How many columns each? What are column headers?
# 4. SECTIONS: Are there distinct sections? What are they?
# 5. FOOTER: Any footer info? Page numbers?

# Return a JSON object:
# {
#     "header": {
#         "location": "top",
#         "contains": ["vendor_logo", "vendor_name", "address", "phone"]
#     },
#     "patient_info": {
#         "location": "top_left",
#         "fields": ["name", "dob", "patient_id", "sex", "age"]
#     },
#     "tables": [
#         {
#             "location": "center",
#             "columns": ["Test", "Result", "Flag", "Reference Range"],
#             "approximate_rows": 20
#         }
#     ],
#     "sections": ["header", "patient_info", "results", "footer"],
#     "footer": {
#         "location": "bottom",
#         "contains": ["page_number", "report_date", "disclaimer"]
#     }
# }

# JSON:"""

#         try:
#             response = await self._generate_with_image(prompt, image_b64)
#             return self.medgemma.extract_json(response['text']) or {"error": "Failed to parse response"}

#         except Exception as e:
#             return {"error": str(e)}

#     async def _generate_with_image(
#         self,
#         prompt: str,
#         image_base64: str,
#         max_tokens: int = 1500
#     ) -> Dict[str, Any]:
#         """
#         Generate response using Ollama with image input.

#         Ollama's /api/generate endpoint accepts images via the 'images' parameter.
#         """
#         import aiohttp

#         # Get Ollama host and model from client config
#         host = getattr(self.medgemma, 'host', 'http://localhost:11434')
#         model = getattr(self.medgemma, '_model_name', 'MedAIBase/MedGemma1.5:4b-it-q8_0')

#         payload = {
#             "model": model,
#             "prompt": prompt,
#             "images": [image_base64],
#             "stream": False,
#             "options": {
#                 "num_predict": max_tokens,
#                 "temperature": 0.1,
#             }
#         }

#         async with aiohttp.ClientSession() as session:
#             async with session.post(
#                 f"{host}/api/generate",
#                 json=payload,
#                 timeout=aiohttp.ClientTimeout(total=120)
#             ) as response:
#                 if response.status != 200:
#                     error_text = await response.text()
#                     raise RuntimeError(f"Ollama error ({response.status}): {error_text}")

#                 data = await response.json()
#                 return {
#                     "text": data.get('response', ''),
#                     "model": model
#                 }


# class HybridExtractor:
#     """
#     Hybrid extraction combining text and vision approaches.

#     Strategy:
#     1. Try text extraction first (fast)
#     2. If confidence is low or structure is lost, use vision
#     3. Merge results with confidence weighting
#     """

#     # Threshold below which we trigger vision extraction
#     TEXT_CONFIDENCE_THRESHOLD = 0.7

#     def __init__(self, use_gpu: bool = False):
#         from .pdf_extractor import PDFExtractor
#         self.pdf_extractor = PDFExtractor(use_gpu=use_gpu)
#         self.vision_extractor = VisionExtractor()
#         self.logger = logging.getLogger(__name__)

#     async def extract(
#         self,
#         pdf_path: Path,
#         password: Optional[str] = None
#     ) -> Dict[str, Any]:
#         """
#         Extract using hybrid text + vision approach.

#         Returns:
#             {
#                 "text": str,
#                 "tables": list,
#                 "extracted_values": list,
#                 "vendor": str,
#                 "document_type": str,
#                 "template_id": str,
#                 "confidence": float,
#                 "method": "text" | "vision" | "hybrid",
#                 "warnings": list
#             }
#         """
#         result = {
#             "text": "",
#             "tables": [],
#             "extracted_values": [],
#             "vendor": None,
#             "document_type": None,
#             "template_id": None,
#             "confidence": 0.0,
#             "method": "text",
#             "warnings": []
#         }

#         # Step 1: Try text extraction
#         text_result = self.pdf_extractor.extract_best_effort(
#             pdf_path, password=password, enable_ocr=True
#         )

#         result["text"] = text_result.text
#         result["tables"] = text_result.tables
#         result["warnings"].extend(text_result.warnings)

#         text_confidence = text_result.confidence
#         self.logger.info(f"Text extraction confidence: {text_confidence:.2f}")

#         # Step 2: Visual identification (always do this for vendor/template)
#         try:
#             visual_id = await self.vision_extractor.identify_document_visually(pdf_path)
#             result["vendor"] = visual_id.get("vendor")
#             result["document_type"] = visual_id.get("document_type")
#             result["template_id"] = visual_id.get("template_id")

#             self.logger.info(
#                 f"Visual ID: vendor={result['vendor']}, "
#                 f"type={result['document_type']}, "
#                 f"template={result['template_id']}"
#             )

#         except Exception as e:
#             result["warnings"].append(f"Visual identification failed: {e}")
#             self.logger.warning(f"Visual identification failed: {e}")

#         # Step 3: If text confidence is low, use vision extraction
#         if text_confidence < self.TEXT_CONFIDENCE_THRESHOLD:
#             self.logger.info(
#                 f"Text confidence ({text_confidence:.2f}) below threshold "
#                 f"({self.TEXT_CONFIDENCE_THRESHOLD}), using vision extraction"
#             )

#             try:
#                 visual_result = await self.vision_extractor.extract_lab_values_visually(pdf_path)

#                 if visual_result.extracted_values:
#                     result["extracted_values"] = visual_result.extracted_values
#                     result["method"] = "hybrid" if result["text"] else "vision"
#                     result["confidence"] = max(text_confidence, visual_result.confidence)
#                     self.logger.info(
#                         f"Vision extracted {len(visual_result.extracted_values)} values"
#                     )

#             except Exception as e:
#                 result["warnings"].append(f"Vision extraction failed: {e}")
#                 self.logger.warning(f"Vision extraction failed: {e}")

#         else:
#             result["confidence"] = text_confidence

#         return result
