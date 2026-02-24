# ============================================================================
# FILE 2: src/medical_ingestion/classifiers/schema_detector.py
# ============================================================================
"""
Schema Detection - Medical Document Structure Analysis

Determines the specific schema/format within a document type:
- Lab: CBC, CMP, Lipid Panel, Thyroid Panel, etc.
- Radiology: Chest X-Ray, CT, MRI, Ultrasound, etc.
- Pathology: Biopsy, Surgical Path, Cytology, etc.

This enables schema-specific processing and validation.
"""

from typing import Dict, Any, List, Optional
import logging
import re


class SchemaDetector:
    """
    Detect specific medical document schema/subtype.
    
    After document type is classified (lab, radiology, etc.),
    this determines the specific format to enable schema-aware
    processing.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.schemas = self._load_schemas()
    
    def _load_schemas(self) -> Dict[str, Dict]:
        """
        Load schema definitions for each document type.
        
        Each schema has:
        - Test-specific keywords
        - Expected fields
        - Structural patterns
        """
        return {
            # LAB SCHEMAS
            "lab": {
                "cbc": {
                    "name": "Complete Blood Count",
                    "keywords": ["cbc", "complete blood count", "hemoglobin", "wbc", "rbc", "platelets", "hematocrit"],
                    "required_fields": ["wbc", "hemoglobin", "hematocrit", "platelets"],
                    "typical_test_count": [8, 15]
                },
                "cmp": {
                    "name": "Comprehensive Metabolic Panel",
                    "keywords": ["cmp", "comprehensive metabolic", "basic metabolic", "bmp", "glucose", "creatinine", "sodium", "potassium"],
                    "required_fields": ["glucose", "sodium", "potassium", "creatinine"],
                    "typical_test_count": [8, 18]
                },
                "lipid": {
                    "name": "Lipid Panel",
                    "keywords": ["lipid panel", "cholesterol", "hdl", "ldl", "triglycerides"],
                    "required_fields": ["total_cholesterol", "hdl", "ldl"],
                    "typical_test_count": [4, 7]
                },
                "thyroid": {
                    "name": "Thyroid Panel",
                    "keywords": ["thyroid", "tsh", "t4", "t3", "free t4", "free t3"],
                    "required_fields": ["tsh"],
                    "typical_test_count": [1, 5]
                },
                "liver": {
                    "name": "Liver Function Tests",
                    "keywords": ["liver function", "lft", "alt", "ast", "alkaline phosphatase", "bilirubin", "albumin"],
                    "required_fields": ["alt", "ast"],
                    "typical_test_count": [5, 10]
                },
                "coagulation": {
                    "name": "Coagulation Panel",
                    "keywords": ["coagulation", "pt", "ptt", "inr", "prothrombin", "activated partial thromboplastin"],
                    "required_fields": ["pt", "ptt"],
                    "typical_test_count": [2, 5]
                }
            },
            
            # RADIOLOGY SCHEMAS
            "radiology": {
                "chest_xray": {
                    "name": "Chest X-Ray",
                    "keywords": ["chest x-ray", "chest radiograph", "cxr", "pa and lateral"],
                    "sections": ["indication", "technique", "findings", "impression"]
                },
                "ct": {
                    "name": "CT Scan",
                    "keywords": ["ct scan", "computed tomography", "ct abdomen", "ct chest", "ct head"],
                    "sections": ["indication", "technique", "findings", "impression"]
                },
                "mri": {
                    "name": "MRI",
                    "keywords": ["mri", "magnetic resonance", "mr imaging"],
                    "sections": ["indication", "technique", "findings", "impression"]
                },
                "ultrasound": {
                    "name": "Ultrasound",
                    "keywords": ["ultrasound", "sonography", "doppler", "us abdomen"],
                    "sections": ["indication", "findings", "impression"]
                },
                "mammogram": {
                    "name": "Mammogram",
                    "keywords": ["mammogram", "mammography", "breast imaging"],
                    "sections": ["indication", "technique", "findings", "impression", "birads"]
                }
            },
            
            # PATHOLOGY SCHEMAS
            "pathology": {
                "biopsy": {
                    "name": "Biopsy Report",
                    "keywords": ["biopsy", "tissue", "core biopsy", "needle biopsy"],
                    "sections": ["clinical", "gross", "microscopic", "diagnosis"]
                },
                "surgical_pathology": {
                    "name": "Surgical Pathology",
                    "keywords": ["surgical pathology", "excision", "resection"],
                    "sections": ["clinical", "gross", "microscopic", "diagnosis", "margin"]
                },
                "cytology": {
                    "name": "Cytology",
                    "keywords": ["cytology", "pap smear", "fine needle aspiration", "fna"],
                    "sections": ["clinical", "microscopic", "diagnosis"]
                }
            }
        }
    
    def detect_schema(
        self,
        document_type: str,
        text: str,
        structural_hints: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Detect specific schema within document type.
        
        Args:
            document_type: Classified type (lab, radiology, pathology)
            text: Document text
            structural_hints: Optional hints from fingerprinting
            
        Returns:
            {
                "schema": str,
                "confidence": float,
                "reasoning": str,
                "all_scores": Dict[str, float]
            }
        """
        if document_type not in self.schemas:
            return {
                "schema": "unknown",
                "confidence": 0.0,
                "reasoning": f"No schemas defined for {document_type}",
                "all_scores": {}
            }
        
        # Score each schema
        schemas = self.schemas[document_type]
        scores = {}
        
        text_lower = text.lower()
        
        for schema_id, schema_def in schemas.items():
            score = self._score_schema(text_lower, schema_def)
            scores[schema_id] = score
        
        # Get best match
        if scores:
            best_schema = max(scores.keys(), key=lambda k: scores[k])
            best_score = scores[best_schema]
            
            return {
                "schema": best_schema,
                "confidence": best_score,
                "reasoning": f"Best match: {schemas[best_schema]['name']} (score: {best_score:.2f})",
                "all_scores": scores
            }
        
        return {
            "schema": "unknown",
            "confidence": 0.0,
            "reasoning": "No schema matched",
            "all_scores": {}
        }
    
    def _score_schema(self, text: str, schema_def: Dict) -> float:
        """
        Score how well text matches a schema definition.
        
        Scoring:
        - Keyword matches (60%)
        - Required field presence (30%)
        - Section structure (10%)
        """
        signals = {}
        
        # Keyword matching
        keywords = schema_def['keywords']
        keyword_matches = sum(1 for kw in keywords if kw in text)
        signals['keywords'] = keyword_matches / len(keywords) if keywords else 0.0
        
        # Required fields (for lab schemas)
        if 'required_fields' in schema_def:
            required_fields = schema_def['required_fields']
            field_matches = sum(1 for field in required_fields if field.replace('_', ' ') in text)
            signals['fields'] = field_matches / len(required_fields) if required_fields else 0.0
        
        # Section structure (for radiology/pathology)
        if 'sections' in schema_def:
            sections = schema_def['sections']
            section_matches = sum(1 for section in sections if section in text)
            signals['sections'] = section_matches / len(sections) if sections else 0.0
        
        # Calculate weighted score
        weights = {
            'keywords': 0.6,
            'fields': 0.3,
            'sections': 0.3  # Will auto-normalize
        }
        
        # Only use weights for signals that exist
        active_weights = {k: v for k, v in weights.items() if k in signals}
        total_weight = sum(active_weights.values())
        
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in active_weights.items()}
            score = sum(signals[k] * normalized_weights[k] for k in signals.keys())
            return score
        
        return 0.0