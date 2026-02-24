# ============================================================================
# FILE 11: src/medical_ingestion/processors/base_processor.py
# ============================================================================
"""
Base Processor Class

All document processors (Lab, Radiology, Pathology, Prescription) inherit from this.

Defines the standard processor interface and common functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging
from pathlib import Path

from ..core.context.processing_context import ProcessingContext
from ..core.agent_base import Agent
from ..core.config import get_config


class BaseProcessor(ABC):
    """
    Abstract base class for all document processors.

    Each processor:
    1. Has multiple specialized agents
    2. Orchestrates agent execution in sequence
    3. Builds processor-specific FHIR resources
    4. Handles processor-specific validation

    Subclasses must implement:
    - get_name(): Processor identifier
    - _get_agents(): List of agents to execute
    - process(): Main processing logic
    """

    def __init__(self, config: Dict[str, Any] = None):
        # Merge env config with passed config (passed config takes precedence)
        env_config = get_config()
        self.config = {**env_config, **(config or {})}
        self.logger = logging.getLogger(f"{__name__}.{self.get_name()}")
        
        # Initialize agents (lazy loading)
        self._agents: List[Agent] = []
        self._agents_loaded = False
    
    @abstractmethod
    def get_name(self) -> str:
        """Return processor name (e.g., 'LabProcessor')"""
        pass
    
    @abstractmethod
    def _get_agents(self) -> List[Agent]:
        """
        Return list of agents for this processor.
        
        Agents are executed in order.
        
        Example:
            return [
                TemplateMatchingAgent(self.config),
                ExtractionAgent(self.config),
                ValidationAgent(self.config),
                ...
            ]
        """
        pass
    
    @abstractmethod
    async def process(self, context: ProcessingContext) -> Dict[str, Any]:
        """
        Main processing method - orchestrates all agents.
        
        Args:
            context: Shared processing context
            
        Returns:
            Processing result dictionary
        """
        pass
    
    def _load_agents(self):
        """Load agents (lazy initialization)"""
        if not self._agents_loaded:
            self._agents = self._get_agents()
            self._agents_loaded = True
            self.logger.info(f"Loaded {len(self._agents)} agents")
    
    async def _execute_agents(self, context: ProcessingContext) -> List[Dict[str, Any]]:
        """
        Execute all agents in sequence.
        
        Each agent reads from and writes to the shared context.
        
        Returns:
            List of agent results
        """
        self._load_agents()
        
        results = []
        
        for agent in self._agents:
            self.logger.info(f"Executing agent: {agent.get_name()}")
            
            # Execute agent (using agent.run() wrapper for error handling)
            result = await agent.run(context)
            results.append(result)
            
            # Check if agent flagged an error that should stop processing
            if result.get('decision') == 'error' and result.get('critical', False):
                self.logger.error(f"Critical error from {agent.get_name()}, stopping pipeline")
                break
        
        return results
    
    def get_agent_count(self) -> int:
        """Return number of agents in this processor"""
        self._load_agents()
        return len(self._agents)
    
    def get_agent_names(self) -> List[str]:
        """Return names of all agents"""
        self._load_agents()
        return [agent.get_name() for agent in self._agents]
