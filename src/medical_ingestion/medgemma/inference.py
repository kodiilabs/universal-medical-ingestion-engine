# ============================================================================
# src/medgemma/inference.py
# ============================================================================
"""
MedGemma Inference Engine

High-level inference interface for medical reasoning tasks.
Provides:
- Batch inference
- Async inference
- Structured output parsing
- Retry logic
- Performance monitoring
"""

import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .client import create_client


class InferenceMode(Enum):
    """Inference modes"""
    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class InferenceRequest:
    """Single inference request"""
    prompt: str
    max_tokens: int = 1000
    temperature: float = 0.1
    use_cache: bool = True
    extract_json: bool = False
    json_mode: bool = False  # If True, constrain output to valid JSON (Ollama format:"json")
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Single inference result"""
    text: str
    prompt_tokens: int
    generated_tokens: int
    model: str
    device: str
    inference_time: float
    cached: bool
    json_output: Optional[Dict] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if inference succeeded"""
        return self.error is None

    @property
    def total_tokens(self) -> int:
        """Total tokens used"""
        return self.prompt_tokens + self.generated_tokens


@dataclass
class BatchInferenceResult:
    """Batch inference result"""
    results: List[InferenceResult]
    total_time: float
    successful: int
    failed: int
    cache_hits: int

    @property
    def success_rate(self) -> float:
        """Success rate"""
        total = self.successful + self.failed
        return self.successful / total if total > 0 else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate"""
        total = len(self.results)
        return self.cache_hits / total if total > 0 else 0.0


class MedGemmaInference:
    """
    High-level inference engine for MedGemma.

    Features:
    - Single and batch inference
    - Automatic retry on failure
    - Structured output extraction
    - Performance monitoring
    - Cache management
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        use_gpu: bool = True,
        max_retries: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to MedGemma model
            cache_dir: Path to cache directory
            use_cache: Enable response caching
            use_gpu: Enable GPU acceleration
            max_retries: Maximum retry attempts on failure
            config: Additional configuration
        """
        self.logger = logging.getLogger(__name__)
        self.max_retries = max_retries

        # Build client config
        client_config = config or {}
        if model_path:
            client_config['model_path'] = model_path
        if cache_dir:
            client_config['cache_dir'] = cache_dir
        client_config['use_cache'] = use_cache
        client_config['use_gpu'] = use_gpu

        # Initialize client
        self.client = create_client(client_config)

        # Statistics
        self._total_requests = 0
        self._total_failures = 0
        self._total_retries = 0

    async def infer(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        use_cache: bool = True,
        extract_json: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InferenceResult:
        """
        Run single inference.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_cache: Use cached responses
            extract_json: Extract JSON from response
            metadata: Additional metadata to attach

        Returns:
            InferenceResult
        """
        request = InferenceRequest(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            use_cache=use_cache,
            extract_json=extract_json,
            metadata=metadata or {},
        )

        return await self._execute_request(request)

    async def infer_batch(
        self,
        requests: List[InferenceRequest],
        max_concurrent: int = 5,
        fail_fast: bool = False,
    ) -> BatchInferenceResult:
        """
        Run batch inference.

        Args:
            requests: List of inference requests
            max_concurrent: Maximum concurrent inferences
            fail_fast: Stop on first failure

        Returns:
            BatchInferenceResult
        """
        start_time = datetime.now()

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_execute(request: InferenceRequest) -> InferenceResult:
            async with semaphore:
                return await self._execute_request(request)

        # Execute all requests
        tasks = [bounded_execute(req) for req in requests]

        if fail_fast:
            results = []
            for task in asyncio.as_completed(tasks):
                result = await task
                if not result.success:
                    # Cancel remaining tasks
                    for t in tasks:
                        t.cancel()
                    results.append(result)
                    break
                results.append(result)
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Convert exceptions to failed results
            results = [
                r if isinstance(r, InferenceResult) else InferenceResult(
                    text="",
                    prompt_tokens=0,
                    generated_tokens=0,
                    model="medgemma-local",
                    device="unknown",
                    inference_time=0.0,
                    cached=False,
                    error=str(r),
                )
                for r in results
            ]

        total_time = (datetime.now() - start_time).total_seconds()

        # Compute statistics
        successful = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        cache_hits = sum(1 for r in results if r.cached)

        return BatchInferenceResult(
            results=results,
            total_time=total_time,
            successful=successful,
            failed=failed,
            cache_hits=cache_hits,
        )

    async def infer_with_retry(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        max_retries: Optional[int] = None,
        retry_delay: float = 1.0,
    ) -> InferenceResult:
        """
        Run inference with automatic retry on failure.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            max_retries: Maximum retry attempts (uses instance default if None)
            retry_delay: Delay between retries in seconds

        Returns:
            InferenceResult
        """
        max_retries = max_retries or self.max_retries
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                result = await self.infer(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    use_cache=True,
                )

                if result.success:
                    if attempt > 0:
                        self.logger.info(f"Inference succeeded after {attempt} retries")
                        self._total_retries += attempt
                    return result

                last_error = result.error

            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Inference attempt {attempt + 1} failed: {e}")

            # Wait before retry (except on last attempt)
            if attempt < max_retries:
                await asyncio.sleep(retry_delay)

        # All retries exhausted
        self._total_failures += 1
        return InferenceResult(
            text="",
            prompt_tokens=0,
            generated_tokens=0,
            model="medgemma-local",
            device=self.client.device,
            inference_time=0.0,
            cached=False,
            error=f"Failed after {max_retries} retries: {last_error}",
        )

    async def infer_structured(
        self,
        prompt: str,
        output_schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> InferenceResult:
        """
        Run inference expecting structured JSON output.

        Args:
            prompt: Input prompt
            output_schema: Expected JSON schema (for validation)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            InferenceResult with json_output populated
        """
        result = await self.infer(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            extract_json=True,
        )

        # Validate against schema if provided
        if result.success and output_schema and result.json_output:
            try:
                self._validate_schema(result.json_output, output_schema)
            except Exception as e:
                result.error = f"Schema validation failed: {e}"

        return result

    async def infer_parallel(
        self,
        prompts: List[str],
        max_tokens: int = 1000,
        temperature: float = 0.1,
        max_concurrent: int = 5,
    ) -> List[InferenceResult]:
        """
        Run multiple prompts in parallel.

        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            max_concurrent: Maximum concurrent inferences

        Returns:
            List of InferenceResults
        """
        requests = [
            InferenceRequest(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for prompt in prompts
        ]

        batch_result = await self.infer_batch(
            requests=requests,
            max_concurrent=max_concurrent,
        )

        return batch_result.results

    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics"""
        client_stats = self.client.get_statistics()

        return {
            **client_stats,
            "total_requests": self._total_requests,
            "total_failures": self._total_failures,
            "total_retries": self._total_retries,
            "failure_rate": (
                self._total_failures / self._total_requests
                if self._total_requests > 0
                else 0.0
            ),
        }

    def cleanup_cache(self) -> int:
        """Cleanup expired cache entries"""
        return self.client.cleanup_cache()

    def clear_cache(self):
        """Clear all cache entries"""
        self.client.clear_cache()

    def warm_cache(self, entries: Dict[str, Dict[str, Any]]) -> int:
        """Preload cache with common queries"""
        return self.client.warm_cache(entries)

    async def _execute_request(self, request: InferenceRequest) -> InferenceResult:
        """Execute single inference request"""
        self._total_requests += 1

        try:
            # Use json_mode if explicitly requested OR if extract_json is True
            use_json_mode = request.json_mode or request.extract_json

            # Run inference
            response = await self.client.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                use_cache=request.use_cache,
                json_mode=use_json_mode,
            )

            # Extract JSON if requested
            json_output = None
            if request.extract_json:
                json_output = self.client.extract_json(response['text'])

            # Build result
            result = InferenceResult(
                text=response['text'],
                prompt_tokens=response['prompt_tokens'],
                generated_tokens=response['generated_tokens'],
                model=response['model'],
                device=response['device'],
                inference_time=response['inference_time'],
                cached=response['cached'],
                json_output=json_output,
                metadata=request.metadata,
            )

            return result

        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            self._total_failures += 1

            return InferenceResult(
                text="",
                prompt_tokens=0,
                generated_tokens=0,
                model="medgemma-local",
                device=self.client.device,
                inference_time=0.0,
                cached=False,
                error=str(e),
                metadata=request.metadata,
            )

    def _validate_schema(self, data: Dict, schema: Dict):
        """
        Simple schema validation.

        Validates that:
        - Required keys are present
        - Types match expected types
        """
        if "required" in schema:
            for key in schema["required"]:
                if key not in data:
                    raise ValueError(f"Missing required field: {key}")

        if "properties" in schema:
            for key, prop_schema in schema["properties"].items():
                if key in data:
                    expected_type = prop_schema.get("type")
                    if expected_type:
                        actual_value = data[key]
                        if not self._check_type(actual_value, expected_type):
                            raise TypeError(
                                f"Field '{key}' expected type {expected_type}, "
                                f"got {type(actual_value).__name__}"
                            )

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON schema type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, skip validation

        return isinstance(value, expected_python_type)


# Singleton instance
_default_inference: Optional[MedGemmaInference] = None


def get_default_inference(
    model_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    use_cache: bool = True,
    use_gpu: bool = True,
) -> MedGemmaInference:
    """
    Get or create default inference engine.

    Args:
        model_path: Path to MedGemma model
        cache_dir: Cache directory
        use_cache: Enable caching
        use_gpu: Enable GPU

    Returns:
        Shared MedGemmaInference instance
    """
    global _default_inference

    if _default_inference is None:
        _default_inference = MedGemmaInference(
            model_path=model_path,
            cache_dir=cache_dir,
            use_cache=use_cache,
            use_gpu=use_gpu,
        )

    return _default_inference


# Convenience functions
async def infer(prompt: str, **kwargs) -> InferenceResult:
    """Convenience function for single inference"""
    engine = get_default_inference()
    return await engine.infer(prompt, **kwargs)


async def infer_batch(prompts: List[str], **kwargs) -> List[InferenceResult]:
    """Convenience function for batch inference"""
    engine = get_default_inference()
    return await engine.infer_parallel(prompts, **kwargs)
