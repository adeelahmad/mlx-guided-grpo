"""Remote rollout generation via Ollama or llama.cpp server.

This module enables distributed GRPO training by offloading rollout generation
to a remote GPU running Ollama or llama.cpp server with a quantized model.

Architecture:
    - Remote: Ollama/llama.cpp with GGUF model (fast generation via CUDA)
    - Local: MLX model for policy log probs, loss computation, gradients

Supported backends:
    - Ollama (recommended): Native API with logprobs support
    - llama.cpp server: OpenAI-compatible API

Usage:
    from mlx_grpo.trainer.remote_rollout import RemoteRolloutClient

    # Ollama (recommended)
    client = RemoteRolloutClient("http://192.168.1.100:11434", backend="ollama")
    client.set_model("qwen2.5:4b")
    rollouts = client.generate(prompts, n=4, max_tokens=512)

    # llama.cpp
    client = RemoteRolloutClient("http://192.168.1.100:8080", backend="llamacpp")
    rollouts = client.generate(prompts, n=4, max_tokens=512)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urljoin

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class RolloutResult:
    """Result from a single rollout generation."""

    prompt: str
    completion: str
    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    finish_reason: str = ""
    generation_time: float = 0.0


@dataclass
class RemoteRolloutConfig:
    """Configuration for remote rollout client."""

    base_url: str = "http://localhost:11434"
    backend: str = "ollama"  # "ollama" or "llamacpp"
    model: str = ""  # Required for Ollama (e.g., "qwen2.5:4b")
    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0
    # Generation parameters
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    # Whether to request logprobs (increases response size)
    return_logprobs: bool = True  # Default True for GRPO
    logprobs_count: int = 1


class RemoteRolloutClient:
    """Client for generating rollouts via remote Ollama or llama.cpp server.

    Supports both sync and async operation. Works with Ollama's native API
    or llama.cpp's OpenAI-compatible API.

    Example:
        # Ollama (recommended)
        client = RemoteRolloutClient("http://192.168.1.100:11434", backend="ollama")
        client.set_model("qwen2.5:4b")
        results = client.generate(
            prompts=["What is 2+2?", "Explain gravity"],
            n=4,  # 4 completions per prompt
            max_tokens=512
        )

        # llama.cpp
        client = RemoteRolloutClient("http://192.168.1.100:8080", backend="llamacpp")
        results = client.generate(prompts, n=4, max_tokens=512)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        config: Optional[RemoteRolloutConfig] = None,
        backend: str = "ollama",
        model: str = "",
    ):
        """Initialize the remote rollout client.

        Args:
            base_url: URL of the server (Ollama: port 11434, llama.cpp: port 8080)
            config: Optional configuration. If None, uses defaults.
            backend: "ollama" or "llamacpp"
            model: Model name for Ollama (e.g., "qwen2.5:4b")
        """
        self.config = config or RemoteRolloutConfig(base_url=base_url)
        self.config.base_url = base_url.rstrip("/")
        self.config.backend = backend
        if model:
            self.config.model = model
        self._session: Optional[aiohttp.ClientSession] = None

    def set_model(self, model: str) -> None:
        """Set the model name for Ollama backend.

        Args:
            model: Model name (e.g., "qwen2.5:4b", "llama3:8b")
        """
        self.config.model = model

    @property
    def completion_url(self) -> str:
        """Get the completion API endpoint URL."""
        if self.config.backend == "ollama":
            return f"{self.config.base_url}/api/generate"
        return f"{self.config.base_url}/v1/completions"

    @property
    def chat_url(self) -> str:
        """Get the chat completion API endpoint URL."""
        if self.config.backend == "ollama":
            return f"{self.config.base_url}/api/chat"
        return f"{self.config.base_url}/v1/chat/completions"

    @property
    def health_url(self) -> str:
        """Get the health check endpoint URL."""
        if self.config.backend == "ollama":
            return f"{self.config.base_url}/api/tags"
        return f"{self.config.base_url}/health"

    def check_connection(self) -> bool:
        """Check if the remote server is reachable.

        Returns:
            True if server responds, False otherwise.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required: pip install requests")

        try:
            response = requests.get(self.health_url, timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models on the remote server.

        Returns:
            List of model names.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required: pip install requests")

        try:
            if self.config.backend == "ollama":
                response = requests.get(
                    f"{self.config.base_url}/api/tags", timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    return [m["name"] for m in data.get("models", [])]
            else:
                response = requests.get(
                    f"{self.config.base_url}/v1/models", timeout=5.0
                )
                if response.status_code == 200:
                    data = response.json()
                    return [m["id"] for m in data.get("data", [])]
        except Exception:
            pass
        return []

    def get_server_info(self) -> dict[str, Any]:
        """Get information about the remote server and loaded model.

        Returns:
            Server info dict with model details.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required: pip install requests")

        try:
            if self.config.backend == "ollama":
                response = requests.get(
                    f"{self.config.base_url}/api/tags", timeout=5.0
                )
                if response.status_code == 200:
                    return response.json()
            else:
                # Try props endpoint (llama.cpp specific)
                response = requests.get(
                    f"{self.config.base_url}/props", timeout=5.0
                )
                if response.status_code == 200:
                    return response.json()

                # Fallback to models endpoint (OpenAI compatible)
                response = requests.get(
                    f"{self.config.base_url}/v1/models", timeout=5.0
                )
                if response.status_code == 200:
                    return response.json()
        except Exception as e:
            return {"error": str(e)}

        return {}

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        max_tokens: int = 512,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[list[RolloutResult]]:
        """Generate rollouts synchronously.

        Args:
            prompts: List of prompts to generate completions for.
            n: Number of completions per prompt (group_size).
            max_tokens: Maximum tokens to generate per completion.
            stop: Optional stop sequences.
            **kwargs: Additional generation parameters.

        Returns:
            List of lists - outer list is per prompt, inner list is n completions.
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required: pip install requests")

        results: list[list[RolloutResult]] = []

        for prompt in prompts:
            prompt_results = self._generate_single_sync(
                prompt=prompt,
                n=n,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs,
            )
            results.append(prompt_results)

        return results

    def _generate_single_sync(
        self,
        prompt: str,
        n: int,
        max_tokens: int,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[RolloutResult]:
        """Generate n completions for a single prompt (sync)."""
        all_results: list[RolloutResult] = []

        # For Ollama/llama.cpp, we need to make n separate requests
        # since they don't support n>1 in a single request
        for i in range(n):
            payload = self._build_payload(prompt, 1, max_tokens, stop, **kwargs)

            for attempt in range(self.config.max_retries):
                try:
                    start_time = time.perf_counter()
                    response = requests.post(
                        self.completion_url,
                        json=payload,
                        timeout=self.config.timeout,
                    )
                    elapsed = time.perf_counter() - start_time

                    if response.status_code == 200:
                        results = self._parse_response(
                            prompt, response.json(), elapsed
                        )
                        all_results.extend(results)
                        break
                    else:
                        error_msg = (
                            f"Server returned {response.status_code}: {response.text}"
                        )
                        if attempt < self.config.max_retries - 1:
                            time.sleep(self.config.retry_delay)
                            continue
                        raise RuntimeError(error_msg)

                except requests.exceptions.Timeout:
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                        continue
                    raise TimeoutError(
                        f"Request timed out after {self.config.timeout}s"
                    )
                except requests.exceptions.ConnectionError as e:
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                        continue
                    raise ConnectionError(f"Failed to connect to server: {e}")

        return all_results

    async def generate_async(
        self,
        prompts: list[str],
        n: int = 1,
        max_tokens: int = 512,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[list[RolloutResult]]:
        """Generate rollouts asynchronously (parallel requests).

        Args:
            prompts: List of prompts to generate completions for.
            n: Number of completions per prompt (group_size).
            max_tokens: Maximum tokens to generate per completion.
            stop: Optional stop sequences.
            **kwargs: Additional generation parameters.

        Returns:
            List of lists - outer list is per prompt, inner list is n completions.
        """
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp library required: pip install aiohttp")

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._generate_single_async(
                    session=session,
                    prompt=prompt,
                    n=n,
                    max_tokens=max_tokens,
                    stop=stop,
                    **kwargs,
                )
                for prompt in prompts
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results: list[list[RolloutResult]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Warning: Generation failed for prompt {i}: {result}")
                # Return empty results for failed prompts
                processed_results.append([])
            else:
                processed_results.append(result)

        return processed_results

    async def _generate_single_async(
        self,
        session: aiohttp.ClientSession,
        prompt: str,
        n: int,
        max_tokens: int,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[RolloutResult]:
        """Generate n completions for a single prompt (async)."""
        payload = self._build_payload(prompt, n, max_tokens, stop, **kwargs)

        for attempt in range(self.config.max_retries):
            try:
                start_time = time.perf_counter()
                async with session.post(
                    self.completion_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                ) as response:
                    elapsed = time.perf_counter() - start_time

                    if response.status == 200:
                        data = await response.json()
                        return self._parse_response(prompt, data, elapsed)
                    else:
                        error_text = await response.text()
                        error_msg = f"Server returned {response.status}: {error_text}"
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay)
                            continue
                        raise RuntimeError(error_msg)

            except asyncio.TimeoutError:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                raise TimeoutError(
                    f"Request timed out after {self.config.timeout}s"
                )
            except aiohttp.ClientError as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                raise ConnectionError(f"Failed to connect to server: {e}")

        return []

    def _build_payload(
        self,
        prompt: str,
        n: int,
        max_tokens: int,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the API request payload."""
        if self.config.backend == "ollama":
            # Ollama native API format
            payload: dict[str, Any] = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "top_p": kwargs.get("top_p", self.config.top_p),
                    "top_k": kwargs.get("top_k", self.config.top_k),
                    "repeat_penalty": kwargs.get(
                        "repeat_penalty", self.config.repeat_penalty
                    ),
                    "num_predict": max_tokens,
                },
            }
            if stop:
                payload["options"]["stop"] = stop
        else:
            # llama.cpp / OpenAI-compatible format
            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                "repeat_penalty": kwargs.get(
                    "repeat_penalty", self.config.repeat_penalty
                ),
                "n_probs": (
                    self.config.logprobs_count if self.config.return_logprobs else 0
                ),
            }
            if n > 1:
                payload["n"] = n
            if stop:
                payload["stop"] = stop

        # Add any extra kwargs
        for key, value in kwargs.items():
            if key not in payload and key not in ["temperature", "top_p", "top_k", "repeat_penalty"]:
                payload[key] = value

        return payload

    def _parse_response(
        self,
        prompt: str,
        response_data: dict[str, Any],
        elapsed: float,
    ) -> list[RolloutResult]:
        """Parse the API response into RolloutResult objects."""
        results: list[RolloutResult] = []

        if self.config.backend == "ollama":
            # Ollama native format
            completion = response_data.get("response", "")
            # Ollama returns context as token IDs
            tokens = response_data.get("context", [])
            # Extract completion tokens (after prompt tokens)
            eval_count = response_data.get("eval_count", 0)
            if tokens and eval_count > 0:
                # Last eval_count tokens are the completion
                completion_tokens = tokens[-eval_count:] if eval_count <= len(tokens) else tokens
            else:
                completion_tokens = []

            results.append(
                RolloutResult(
                    prompt=prompt,
                    completion=completion,
                    tokens=completion_tokens,
                    logprobs=[],  # Ollama doesn't return logprobs in standard API
                    finish_reason="stop" if response_data.get("done") else "length",
                    generation_time=elapsed,
                )
            )
        elif "choices" in response_data:
            # OpenAI-compatible format (llama.cpp)
            for choice in response_data["choices"]:
                completion = choice.get("text", "")
                tokens = choice.get("tokens", [])
                logprobs_data = choice.get("logprobs", {})

                logprobs: list[float] = []
                if logprobs_data and "token_logprobs" in logprobs_data:
                    logprobs = logprobs_data["token_logprobs"]

                results.append(
                    RolloutResult(
                        prompt=prompt,
                        completion=completion,
                        tokens=tokens,
                        logprobs=logprobs,
                        finish_reason=choice.get("finish_reason", ""),
                        generation_time=elapsed / len(response_data["choices"]),
                    )
                )
        elif "content" in response_data:
            # llama.cpp native format
            results.append(
                RolloutResult(
                    prompt=prompt,
                    completion=response_data["content"],
                    tokens=response_data.get("tokens", []),
                    logprobs=[],
                    finish_reason=response_data.get("stop_type", ""),
                    generation_time=elapsed,
                )
            )

        return results

    def generate_batch(
        self,
        prompts: list[str],
        n: int = 1,
        max_tokens: int = 512,
        stop: Optional[list[str]] = None,
        use_async: bool = True,
        **kwargs: Any,
    ) -> list[list[RolloutResult]]:
        """Generate rollouts for a batch of prompts.

        This is the main entry point for GRPO integration. Uses async
        by default for better throughput.

        Args:
            prompts: List of prompts.
            n: Completions per prompt (group_size).
            max_tokens: Max tokens per completion.
            stop: Stop sequences.
            use_async: Use async requests (default True for throughput).
            **kwargs: Additional generation params.

        Returns:
            Nested list of RolloutResults.
        """
        if use_async and AIOHTTP_AVAILABLE:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(
                    self.generate_async(
                        prompts=prompts,
                        n=n,
                        max_tokens=max_tokens,
                        stop=stop,
                        **kwargs,
                    )
                )
            finally:
                loop.close()
        else:
            return self.generate(
                prompts=prompts,
                n=n,
                max_tokens=max_tokens,
                stop=stop,
                **kwargs,
            )


def create_remote_client(
    url: str,
    timeout: float = 120.0,
    temperature: float = 0.8,
    return_logprobs: bool = False,
) -> RemoteRolloutClient:
    """Factory function to create a configured remote rollout client.

    Args:
        url: Server URL (e.g., http://192.168.1.100:8080)
        timeout: Request timeout in seconds.
        temperature: Generation temperature.
        return_logprobs: Whether to request log probabilities.

    Returns:
        Configured RemoteRolloutClient instance.
    """
    config = RemoteRolloutConfig(
        base_url=url,
        timeout=timeout,
        temperature=temperature,
        return_logprobs=return_logprobs,
    )
    return RemoteRolloutClient(base_url=url, config=config)


# Convenience function for GRPO integration
def generate_remote_rollouts(
    client: RemoteRolloutClient,
    prompts: list[str],
    group_size: int,
    max_tokens: int,
    temperature: float = 0.8,
    stop_sequences: Optional[list[str]] = None,
) -> tuple[list[str], list[list[str]]]:
    """Generate rollouts using remote server, formatted for GRPO.

    Args:
        client: Configured RemoteRolloutClient.
        prompts: Batch of prompts.
        group_size: Number of completions per prompt.
        max_tokens: Max completion length.
        temperature: Sampling temperature.
        stop_sequences: Optional stop sequences.

    Returns:
        Tuple of (flat_prompts, completions_per_prompt) where:
        - flat_prompts: Prompts repeated group_size times each
        - completions_per_prompt: List of completion lists
    """
    results = client.generate_batch(
        prompts=prompts,
        n=group_size,
        max_tokens=max_tokens,
        stop=stop_sequences,
        temperature=temperature,
    )

    # Flatten prompts (repeat each prompt group_size times)
    flat_prompts: list[str] = []
    for prompt in prompts:
        flat_prompts.extend([prompt] * group_size)

    # Extract completions
    completions_per_prompt: list[list[str]] = []
    for prompt_results in results:
        completions = [r.completion for r in prompt_results]
        # Pad if we didn't get enough completions
        while len(completions) < group_size:
            completions.append("")
        completions_per_prompt.append(completions[:group_size])

    return flat_prompts, completions_per_prompt
