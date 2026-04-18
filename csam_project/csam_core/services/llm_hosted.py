"""
Hosted LLM Service - Drop-in replacement for LLMService using cloud providers.

Supports OpenAI-compatible APIs:
- Groq       (api.groq.com)          — fast inference, generous free tier
- Cerebras   (api.cerebras.ai)        — wafer-scale, very high throughput
- SambaNova  (api.sambanova.ai)       — fast inference, good free limits
- Fireworks  (api.fireworks.ai)       — generous free tier, many models
- Together   (api.together.xyz)       — $1 free credit, then affordable
- NVIDIA NIM (integrate.api.nvidia.com) — $50 free credit, OpenAI-compat
- OpenRouter (openrouter.ai)          — aggregator, free model tier

Same interface as LLMService: generate(), summarize(), extract_entities(), generate_response()
"""

import requests
import json
import logging
import time
import os
import re
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry — all OpenAI-compatible chat/completions endpoints
# ---------------------------------------------------------------------------
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "rate_limit_rpm": 30,
        # Free-tier approximate daily token limits (varies by model)
        "notes": "30 RPM free. ~6k TPM. Strong speed. Best free option.",
        "models": {
            "llama-3.1-8b":   "llama-3.1-8b-instant",
            "llama-3.3-70b":  "llama-3.3-70b-versatile",
            "llama-4-scout":  "meta-llama/llama-4-scout-17b-16e-instruct",
            "gemma2-9b":      "gemma2-9b-it",
            "gemma-7b":       "gemma-7b-it",
            "mixtral-8x7b":   "mixtral-8x7b-32768",
            "qwen-32b":       "qwen/qwen3-32b",
        },
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "env_key": "CEREBRAS_API_KEY",      # was CEREBAS_API_KEY (typo fixed)
        "rate_limit_rpm": 60,
        "notes": "60 RPM free. Very high throughput (1800+ tok/s). Best for bulk runs.",
        "models": {
            "llama-3.1-8b":  "llama3.1-8b",
            "llama-3.3-70b": "llama-3.3-70b",
            "llama-3.1-70b": "llama3.1-70b",
        },
    },
    "sambanova": {
        "base_url": "https://api.sambanova.ai/v1",
        "env_key": "SAMBANOVA_API_KEY",
        "rate_limit_rpm": 30,
        "notes": "30 RPM free. Fast RDU inference. Good Llama 70B/405B support.",
        "models": {
            "llama-3.1-8b":   "Meta-Llama-3.1-8B-Instruct",
            "llama-3.2-3b":   "Meta-Llama-3.2-3B-Instruct",
            "llama-3.3-70b":  "Meta-Llama-3.3-70B-Instruct",
            "llama-3.1-405b": "Meta-Llama-3.1-405B-Instruct",
        },
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_key": "FIREWORKS_API_KEY",
        "rate_limit_rpm": 60,
        "notes": "60 RPM free tier. Generous daily limits. Many open models. Good for bulk.",
        "models": {
            "llama-3.1-8b":  "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "llama-3.1-70b": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "llama-3.3-70b": "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "gemma2-9b":     "accounts/fireworks/models/gemma2-9b-it",
            "mixtral-8x7b":  "accounts/fireworks/models/mixtral-8x7b-instruct",
            "qwen2.5-72b":   "accounts/fireworks/models/qwen2p5-72b-instruct",
        },
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "rate_limit_rpm": 60,
        "notes": "60 RPM. $1 free credit, then ~$0.18/M tok for 70B. Best model variety.",
        "models": {
            "llama-3.1-8b":  "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "llama-3.3-70b": "meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo",
            "gemma2-9b":     "google/gemma-2-9b-it",
            "gemma2-27b":    "google/gemma-2-27b-it",
            "qwen2.5-72b":   "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "mixtral-8x7b":  "mistralai/Mixtral-8x7B-Instruct-v0.1",
        },
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_API_KEY",
        "rate_limit_rpm": 40,
        "notes": "$50 free credit. OpenAI-compat. Widest model selection incl. Nemotron.",
        "models": {
            "llama-3.1-8b":    "meta/llama-3.1-8b-instruct",
            "llama-3.1-70b":   "meta/llama-3.1-70b-instruct",
            "llama-3.3-70b":   "meta/llama-3.3-70b-instruct",
            "llama-3.1-405b":  "meta/llama-3.1-405b-instruct",
            "gemma2-9b":       "google/gemma-2-9b-it",
            "gemma2-27b":      "google/gemma-2-27b-it",
            "mixtral-8x7b":    "mistralai/mixtral-8x7b-instruct-v0.1",
            "nemotron-70b":    "nvidia/llama-3.1-nemotron-70b-instruct",
        },
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "rate_limit_rpm": 20,
        "notes": "20 RPM on free models. 200 RPD on free tier. Aggregates many providers.",
        "models": {
            "llama-3.1-8b":   "meta-llama/llama-3.1-8b-instruct:free",
            "llama-3.3-70b":  "meta-llama/llama-3.3-70b-instruct:free",
            "gemma2-9b":      "google/gemma-2-9b-it:free",
            "phi-3-mini":     "microsoft/phi-3-mini-128k-instruct:free",
            "mistral-7b":     "mistralai/mistral-7b-instruct:free",
        },
    },
}

# ---------------------------------------------------------------------------
# Canonical publication model set
# Chosen for: size diversity, architecture diversity, all on Groq (same provider
# = controlled comparison), all free tier.
# ---------------------------------------------------------------------------
PUBLICATION_MODELS = [
    {
        "provider": "groq",
        "model":    "llama-3.1-8b-instant",
        "label":    "Llama-3.1-8B",
        "size_b":   8,
        "family":   "llama",
    },
    {
        "provider": "groq",
        "model":    "meta-llama/llama-4-scout-17b-16e-instruct",
        "label":    "Llama-4-Scout-17B",
        "size_b":   17,
        "family":   "llama",
    },
    {
        "provider": "groq",
        "model":    "llama-3.3-70b-versatile",
        "label":    "Llama-3.3-70B",
        "size_b":   70,
        "family":   "llama",
    },
    {
        "provider": "groq",
        "model":    "openai/gpt-oss-120b",
        "label":    "GPT-OSS-120B",
        "size_b":   120,
        "family":   "openai",
    },
]

# Fallback providers per model size (used when primary hits rate limit)
FALLBACK_PROVIDERS = {
    "8b":  [
        {"provider": "cerebras",  "model": "llama3.1-8b"},
        {"provider": "sambanova", "model": "Meta-Llama-3.1-8B-Instruct"},
        {"provider": "fireworks", "model": "accounts/fireworks/models/llama-v3p1-8b-instruct"},
    ],
    "70b": [
        {"provider": "cerebras",  "model": "llama-3.3-70b"},
        {"provider": "sambanova", "model": "Meta-Llama-3.3-70B-Instruct"},
        {"provider": "fireworks", "model": "accounts/fireworks/models/llama-v3p3-70b-instruct"},
    ],
}


class RateLimitExceeded(Exception):
    """Raised when all retry attempts are exhausted."""


class HostedLLMService:
    """
    Drop-in replacement for LLMService using hosted OpenAI-compatible APIs.

    Supports Groq, Cerebras, SambaNova, Fireworks, Together, NVIDIA NIM,
    and OpenRouter. Implements key rotation, rate-limit backoff with a
    configurable retry cap, and per-request seed forwarding.

    Usage:
        llm = HostedLLMService(provider="groq",     model="llama-3.1-8b-instant")
        llm = HostedLLMService(provider="cerebras", model="llama-3.3-70b")
        llm = HostedLLMService(provider="fireworks",model="accounts/fireworks/models/llama-v3p3-70b-instruct")
    """

    MAX_RETRIES: int = 5      # maximum 429-retry attempts before raising
    DEFAULT_BACKOFF: int = 30  # fallback sleep seconds when Retry-After header is absent

    def __init__(
        self,
        provider: str = "groq",
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        timeout: int = 120,
        rate_limit_rpm: Optional[int] = None,
        max_retries: int = MAX_RETRIES,
    ):
        if provider not in PROVIDERS:
            raise ValueError(
                f"Unknown provider: '{provider}'. "
                f"Available: {sorted(PROVIDERS.keys())}"
            )

        self.provider = provider
        self.provider_config = PROVIDERS[provider]
        self.base_url = self.provider_config["base_url"]
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

        # Rate limiting
        self.rate_limit_rpm = rate_limit_rpm or self.provider_config["rate_limit_rpm"]
        self.min_interval = 60.0 / self.rate_limit_rpm
        self._last_request_time: float = 0.0

        # API key rotation (supports GROQ_API_KEY, GROQ_API_KEY_2, …)
        self._api_keys: List[str] = []
        self._key_index: int = 0
        if api_key:
            self._api_keys = [api_key]
        else:
            env_key = self.provider_config["env_key"]
            base = os.environ.get(env_key, "")
            if base:
                self._api_keys.append(base)
            for suffix in range(2, 20):
                extra = os.environ.get(f"{env_key}_{suffix}", "")
                if extra:
                    self._api_keys.append(extra)
            if not self._api_keys:
                logger.warning(
                    "No API key found for '%s'. Set $%s environment variable.",
                    provider, env_key,
                )
            else:
                logger.info(
                    "Loaded %d API key(s) for provider '%s'.",
                    len(self._api_keys), provider,
                )

        self.api_key: str = self._api_keys[0] if self._api_keys else ""

        # Usage counters
        self.total_requests: int = 0
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0
        self.total_latency_ms: float = 0.0

        logger.info("HostedLLMService ready: provider=%s  model=%s", provider, model)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Remove <think>…</think> reasoning blocks (Qwen3, DeepSeek-R1, etc.)."""
        if not text:
            return text
        stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if "<think>" in stripped:
            stripped = re.sub(r"<think>.*", "", stripped, flags=re.DOTALL).strip()
        return stripped if stripped else text

    def _rate_limit_wait(self) -> None:
        """Sleep until the minimum inter-request interval has elapsed."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()

    def _rotate_key(self) -> None:
        """Advance to the next API key in the rotation pool."""
        if len(self._api_keys) > 1:
            self._key_index = (self._key_index + 1) % len(self._api_keys)
            self.api_key = self._api_keys[self._key_index]
            logger.info(
                "Rotated to API key %d/%d for %s.",
                self._key_index + 1, len(self._api_keys), self.provider,
            )

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter requires these extra headers
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/csam-project"
            headers["X-Title"] = "CSAM Research"
        return headers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the provider endpoint is reachable with the current key."""
        if not self.api_key:
            return False
        try:
            r = requests.get(
                f"{self.base_url}/models",
                headers=self._build_headers(),
                timeout=10,
            )
            return r.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: Optional[int] = None,
        _retry_count: int = 0,
    ) -> str:
        """
        Generate a chat completion.  Retries on HTTP 429 up to ``max_retries``
        times with key rotation; raises ``RateLimitExceeded`` if exhausted.

        Args:
            prompt:        User message content.
            system_prompt: Optional system message.
            temperature:   Sampling temperature.
            max_tokens:    Max output tokens.
            seed:          Optional integer seed for reproducibility.
            _retry_count:  Internal counter — callers should not set this.

        Returns:
            Generated text string (empty string on non-429 errors).

        Raises:
            RateLimitExceeded: When ``max_retries`` 429 retries are exhausted.
        """
        self._rate_limit_wait()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model":       self.model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      False,
        }
        if seed is not None:
            payload["seed"] = seed

        try:
            t0 = time.time()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self._build_headers(),
                timeout=self.timeout,
            )
            latency_ms = (time.time() - t0) * 1000

            if response.status_code == 200:
                data = response.json()
                result: str = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                self.total_requests += 1
                self.total_tokens_in  += usage.get("prompt_tokens", 0)
                self.total_tokens_out += usage.get("completion_tokens", 0)
                self.total_latency_ms += latency_ms
                return self._strip_thinking_tags(result)

            elif response.status_code == 429:
                if _retry_count >= self.max_retries:
                    raise RateLimitExceeded(
                        f"{self.provider} returned HTTP 429 after "
                        f"{self.max_retries} retries. "
                        "Consider adding more API keys (GROQ_API_KEY_2, …) "
                        "or switching provider."
                    )
                retry_after = int(
                    response.headers.get("Retry-After", self.DEFAULT_BACKOFF)
                )
                self._rotate_key()
                logger.warning(
                    "Rate limited by %s (attempt %d/%d). "
                    "Waiting %ds then retrying with key %d/%d.",
                    self.provider, _retry_count + 1, self.max_retries,
                    retry_after, self._key_index + 1, len(self._api_keys),
                )
                time.sleep(retry_after)
                return self.generate(
                    prompt, system_prompt, temperature, max_tokens,
                    seed=seed, _retry_count=_retry_count + 1,
                )

            else:
                logger.error(
                    "%s error %d: %s",
                    self.provider, response.status_code, response.text[:300],
                )
                return ""

        except requests.exceptions.Timeout:
            logger.error("Timeout waiting for %s (%ds).", self.provider, self.timeout)
            return ""
        except requests.exceptions.RequestException as exc:
            logger.error("Connection error to %s: %s", self.provider, exc)
            return ""

    def summarize(self, memories: List[str]) -> str:
        """Summarize a list of memory strings into one concise statement."""
        memories_text = "\n".join(f"- {m}" for m in memories)
        prompt = (
            "Summarize the following memories into a single concise statement.\n"
            "Focus on the key facts. Be brief.\n\n"
            f"Memories:\n{memories_text}\n\nSummary:"
        )
        system = "You are a memory consolidation system. Produce concise, factual summaries."
        return self.generate(prompt, system_prompt=system, temperature=0.3, max_tokens=100)

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities and relationships from text as JSON."""
        prompt = (
            f'Extract entities and relationships from this text.\n\n'
            f'Text: {text}\n\n'
            'Respond in this exact JSON format:\n'
            '{"entities": [{"name": "...", "type": "Person|Place|Object|Concept"}], '
            '"relationships": [{"source": "...", "target": "...", "type": "..."}]}\n\nJSON:'
        )
        system = "You are an entity extraction system. Output valid JSON only."
        response = self.generate(prompt, system_prompt=system, temperature=0.1, max_tokens=500)

        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                first_nl = cleaned.find("\n")
                if first_nl > 0:
                    cleaned = cleaned[first_nl + 1:]
                if cleaned.rstrip().endswith("```"):
                    cleaned = cleaned.rstrip()[:-3].rstrip()

            json_start = cleaned.find("{")
            if json_start >= 0:
                depth, json_end = 0, -1
                for i in range(json_start, len(cleaned)):
                    if cleaned[i] == "{":
                        depth += 1
                    elif cleaned[i] == "}":
                        depth -= 1
                        if depth == 0:
                            json_end = i + 1
                            break
                if json_end > json_start:
                    return json.loads(cleaned[json_start:json_end])
                # Truncated — try to close open structures
                partial = cleaned[json_start:].rstrip().rstrip(",")
                partial += "]" * (partial.count("[") - partial.count("]"))
                partial += "}" * (partial.count("{") - partial.count("}"))
                return json.loads(partial)
        except json.JSONDecodeError:
            logger.warning("Entity extraction parse error: %s", response[:200])

        return {"entities": [], "relationships": []}

    def generate_response(
        self,
        context: str,
        user_message: str,
        persona: Optional[str] = None,
        mode: str = "chat",
        seed: Optional[int] = None,
    ) -> str:
        """Generate an NPC response given context. Same interface as LLMService."""
        if mode == "qa":
            prompt = (
                "Answer the question based ONLY on the context below. "
                "Be extremely concise.\n\n"
                f"Context:\n{context}\n\nQuestion: {user_message}\n\nAnswer:"
            )
            system = (
                "You are a precise database. Output only the requested date, "
                "name, or fact. Do not use full sentences unless necessary."
            )
            temperature = 0.1
        else:
            prompt = (
                "Based on the following context from your memory, respond to the user.\n\n"
                f"Context from memory:\n{context}\n\nUser says: {user_message}\n\nYour response:"
            )
            system = (
                persona or
                "You are a helpful NPC with a good memory. "
                "Be friendly and reference past conversations when relevant."
            )
            temperature = 0.7

        return self.generate(
            prompt, system_prompt=system,
            temperature=temperature, max_tokens=150, seed=seed,
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return accumulated API usage counters."""
        return {
            "provider":        self.provider,
            "model":           self.model,
            "total_requests":  self.total_requests,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens":    self.total_tokens_in + self.total_tokens_out,
            "avg_latency_ms":  self.total_latency_ms / max(1, self.total_requests),
            "num_api_keys":    len(self._api_keys),
        }

    def __repr__(self) -> str:
        return f"HostedLLMService(provider={self.provider!r}, model={self.model!r})"
