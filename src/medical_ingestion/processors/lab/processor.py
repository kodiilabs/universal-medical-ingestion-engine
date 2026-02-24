# ============================================================================
# FILE 12: src/medical_ingestion/processors/lab/processor.py
# ============================================================================
"""
Lab Report Processor - The Deep Intelligence Processor

This is the MOST COMPLEX processor with 7 specialized agents:
1. Template Matching Agent - Route to template vs MedGemma
2. Extraction Agent - Extract lab values with provenance
3. Validation Agent - Dual validation (rules + AI)
4. Temporal Analysis Agent - Detect trends over time
5. Specimen Quality Agent - Detect pre-analytical errors
6. Clinical Reasoning Agent - Generate summaries & reflex tests
7. Orchestration Agent - Build review queue, aggregate confidence

This processor showcases the full power of the agentic architecture.

OPTIMIZATION: After extraction, Validation/Temporal/SpecimenQuality run in PARALLEL
since they all read from extracted_values independently.
"""

from typing import Dict, Any, List
import logging
import asyncio

from ..base_processor import BaseProcessor
from ...core.context.processing_context import ProcessingContext


class LabProcessor(BaseProcessor):
    """
    Lab report processor with 7-agent deep intelligence pipeline.

    Processing flow (with parallelization):

    SEQUENTIAL (Phase 1 - must run in order):
    1. Template Matching → determine extraction strategy
    2. Extraction → get lab values from PDF

    PARALLEL (Phase 2 - can run simultaneously):
    3. Validation → dual check (rules + AI)
    4. Temporal Analysis → check trends vs patient history
    5. Specimen Quality → detect hemolysis, contamination, etc.

    SEQUENTIAL (Phase 3 - needs results from Phase 2):
    6. Clinical Reasoning → generate summaries, reflex protocols
    7. Orchestration → aggregate results, build review queue

    This is where the "clinical intelligence" lives.
    """

    def get_name(self) -> str:
        return "LabProcessor"

    def _get_agents(self) -> List:
        """
        Initialize all 8 lab processing agents.

        Note: For parallel execution, agents are organized into phases.
        This method returns all agents for compatibility with base class.
        """
        from .agents.template_matcher import TemplateMatchingAgent
        from .agents.metadata_extractor import MetadataExtractionAgent
        from .agents.extractor import ExtractionAgent
        from .agents.validator import ValidationAgent
        from .agents.temporal_analyzer import TemporalAnalysisAgent
        from .agents.specimen_quality import SpecimenQualityAgent
        from .agents.clinical_reasoner import ClinicalReasoningAgent
        from .agents.orchestrator import LabOrchestrationAgent

        return [
            TemplateMatchingAgent(self.config),
            MetadataExtractionAgent(self.config),
            ExtractionAgent(self.config),
            ValidationAgent(self.config),
            TemporalAnalysisAgent(self.config),
            SpecimenQualityAgent(self.config),
            ClinicalReasoningAgent(self.config),
            LabOrchestrationAgent(self.config)
        ]

    def _get_agent_phases(self) -> Dict[str, List]:
        """
        Organize agents into execution phases for parallel processing.

        Returns:
            Dict with phase names and agent lists

        Phase 1 (Sequential - Extraction):
            1. TemplateMatchingAgent - Identify document format
            2. MetadataExtractionAgent - Extract patient/provider/org info
            3. ExtractionAgent - Extract lab values

        Phase 2 (Parallel - Analysis):
            - ValidationAgent, TemporalAnalysisAgent, SpecimenQualityAgent
            All run concurrently as they read from extracted_values independently

        Phase 3 (Sequential - Synthesis):
            1. ClinicalReasoningAgent - Generate summaries
            2. LabOrchestrationAgent - Final aggregation
        """
        from .agents.template_matcher import TemplateMatchingAgent
        from .agents.metadata_extractor import MetadataExtractionAgent
        from .agents.extractor import ExtractionAgent
        from .agents.validator import ValidationAgent
        from .agents.temporal_analyzer import TemporalAnalysisAgent
        from .agents.specimen_quality import SpecimenQualityAgent
        from .agents.clinical_reasoner import ClinicalReasoningAgent
        from .agents.orchestrator import LabOrchestrationAgent

        return {
            # Phase 1: Sequential - Template matching, metadata, and extraction
            "extraction": [
                TemplateMatchingAgent(self.config),
                MetadataExtractionAgent(self.config),
                ExtractionAgent(self.config),
            ],
            # Phase 2: Parallel - Independent validation/analysis agents
            "analysis": [
                ValidationAgent(self.config),
                TemporalAnalysisAgent(self.config),
                SpecimenQualityAgent(self.config),
            ],
            # Phase 3: Sequential - Reasoning and orchestration
            "synthesis": [
                ClinicalReasoningAgent(self.config),
                LabOrchestrationAgent(self.config),
            ]
        }

    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Execute full lab processing pipeline with parallel optimization.

        Phases:
        1. SEQUENTIAL: Template Matching → Extraction
        2. PARALLEL: Validation, Temporal, SpecimenQuality (run concurrently)
        3. SEQUENTIAL: Clinical Reasoning → Orchestration

        Returns:
            {
                "success": bool,
                "agent_results": List[Dict],
                "extracted_values": int,
                "specimen_status": "accepted" | "rejected",
                "requires_review": bool
            }
        """
        self.logger.info(f"Processing lab report: {context.document_id}")

        try:
            phases = self._get_agent_phases()
            all_results = []

            # ================================================================
            # PHASE 1: Sequential extraction (Template Matching → Extraction)
            # ================================================================
            self.logger.info("Phase 1: Sequential extraction agents")
            for agent in phases["extraction"]:
                self.logger.info(f"  Executing: {agent.get_name()}")
                result = await agent.run(context)
                all_results.append(result)

                # Check for critical errors
                if result.get('decision') == 'error' and result.get('critical', False):
                    self.logger.error(f"Critical error from {agent.get_name()}, stopping")
                    raise RuntimeError(f"Critical error in {agent.get_name()}")

            # Populate any missing bounding boxes from table/OCR data
            context.populate_missing_bboxes()
            self.logger.debug(f"Populated bboxes for {len([v for v in context.extracted_values if v.bbox])} values")

            # Sort values to match document order for side-by-side comparison
            context.sort_by_document_order()
            self.logger.debug("Sorted extracted values by document order")

            # ================================================================
            # PHASE 2: Parallel analysis (Validation, Temporal, SpecimenQuality)
            # ================================================================
            self.logger.info("Phase 2: Parallel analysis agents")

            async def run_agent(agent):
                """Run a single agent and return its result."""
                self.logger.info(f"  Starting: {agent.get_name()}")
                result = await agent.run(context)
                self.logger.info(f"  Completed: {agent.get_name()}")
                return result

            # Run all analysis agents concurrently
            parallel_tasks = [run_agent(agent) for agent in phases["analysis"]]
            parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)

            # Process results, handling any exceptions
            for i, result in enumerate(parallel_results):
                agent_name = phases["analysis"][i].get_name()
                if isinstance(result, Exception):
                    self.logger.error(f"Agent {agent_name} failed: {result}")
                    all_results.append({
                        "decision": "error",
                        "agent": agent_name,
                        "error": str(result)
                    })
                else:
                    all_results.append(result)

            # ================================================================
            # PHASE 3: Sequential synthesis (Clinical Reasoning → Orchestration)
            # ================================================================
            self.logger.info("Phase 3: Sequential synthesis agents")
            for agent in phases["synthesis"]:
                self.logger.info(f"  Executing: {agent.get_name()}")
                result = await agent.run(context)
                all_results.append(result)

            # Check if specimen was rejected
            specimen_status = "rejected" if context.specimen_rejected else "accepted"

            # Log summary
            self.logger.info(
                f"Lab processing complete: "
                f"{len(context.extracted_values)} values extracted, "
                f"specimen {specimen_status}, "
                f"review required: {context.requires_review}"
            )

            return {
                "success": True,
                "agent_results": all_results,
                "extracted_values": len(context.extracted_values),
                "specimen_status": specimen_status,
                "requires_review": context.requires_review,
                "confidence": context.overall_confidence
            }

        except Exception as e:
            self.logger.error(f"Lab processing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "agent_results": []
            }