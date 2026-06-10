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
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-model free-tier rate limits (Groq, 2026)
# Source: https://console.groq.com/docs/rate-limits
# ---------------------------------------------------------------------------
MODEL_RATE_LIMITS: Dict[str, Dict[str, int]] = {
    # model_id -> {rpm, tpm, rpd, tpd}
    "llama-3.1-8b-instant":                           {"rpm": 30, "tpm": 30_000, "rpd": 14_400, "tpd": 500_000},
    "llama-3.3-70b-versatile":                        {"rpm": 30, "tpm": 12_000, "rpd": 1_000,  "tpd": 100_000},
    "openai/gpt-oss-120b":                            {"rpm": 30, "tpm": 8_000,  "rpd": 1_000,  "tpd": 200_000},
    "meta-llama/llama-4-scout-17b-16e-instruct":      {"rpm": 30, "tpm": 6_000,  "rpd": 1_000,  "tpd": 100_000},
    "qwen/qwen3-32b":                                 {"rpm": 30, "tpm": 6_000,  "rpd": 1_000,  "tpd": 100_000},
}
_DEFAULT_LIMITS: Dict[str, int] = {"rpm": 30, "tpm": 6_000, "rpd": 1_000, "tpd": 100_000}

# Fraction of a limit at which we proactively rotate to the next key
_TPM_ROTATION_THRESHOLD = 0.85   # rotate when 85% of TPM consumed
_RPM_ROTATION_THRESHOLD = 0.85


@dataclass
class _KeySlot:
    """Per-key rate-limit state tracking."""
    key: str
    # Rolling 60-second window counters
    window_start: float = field(default_factory=time.time)
    rpm_used: int = 0
    tpm_used: int = 0
    # Rolling 24-hour window counters
    day_start: float = field(default_factory=time.time)
    rpd_used: int = 0
    tpd_used: int = 0
    # Backoff — epoch timestamp until which this key is unavailable
    cooldown_until: float = 0.0

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

    MAX_RETRIES: int = 8      # maximum 429-retry attempts before raising
    DEFAULT_BACKOFF: int = 60  # fallback sleep seconds when Retry-After header is absent

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

        # Per-model rate limits (fall back to provider default)
        self._limits = MODEL_RATE_LIMITS.get(model, _DEFAULT_LIMITS).copy()
        if rate_limit_rpm:
            self._limits["rpm"] = rate_limit_rpm

        # Build key slots — supports GROQ_API_KEY, GROQ_API_KEY_2 … GROQ_API_KEY_19
        raw_keys: List[str] = []
        if api_key:
            raw_keys = [api_key]
        else:
            env_key = self.provider_config["env_key"]
            base = os.environ.get(env_key, "")
            if base:
                raw_keys.append(base)
            for suffix in range(2, 20):
                extra = os.environ.get(f"{env_key}_{suffix}", "")
                if extra:
                    raw_keys.append(extra)

        if not raw_keys:
            logger.warning(
                "No API key found for '%s'. Set $%s environment variable.",
                provider, env_key if not api_key else "api_key",
            )
        else:
            logger.info(
                "Loaded %d API key(s) for provider '%s' (model=%s).",
                len(raw_keys), provider, model,
            )

        self._slots: List[_KeySlot] = [_KeySlot(key=k) for k in raw_keys]
        self._slot_index: int = 0

        # Convenience alias kept for back-compat (points at active slot's key)
        self.api_key: str = raw_keys[0] if raw_keys else ""

        # Usage counters (aggregate across all keys)
        self.total_requests: int = 0
        self.total_tokens_in: int = 0
        self.total_tokens_out: int = 0
        self.total_latency_ms: float = 0.0

        logger.info(
            "HostedLLMService ready: provider=%s  model=%s  keys=%d  limits=%s",
            provider, model, len(self._slots), self._limits,
        )

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

    def _refresh_slot(self, slot: "_KeySlot") -> None:
        """Reset per-minute and per-day counters if their windows have expired."""
        now = time.time()
        if now - slot.window_start >= 60.0:
            slot.window_start = now
            slot.rpm_used = 0
            slot.tpm_used = 0
        if now - slot.day_start >= 86_400.0:
            slot.day_start = now
            slot.rpd_used = 0
            slot.tpd_used = 0

    def _pick_slot(self) -> int:
        """
        Return the index of the best available key slot.

        Selection order:
        1. Not in cooldown (429 backoff).
        2. Below RPM and TPM rotation thresholds.
        3. Below daily RPD/TPD limits.
        Among candidates, prefer the slot with lowest tpm_used this minute.
        If all slots are exhausted, sleep until the soonest cooldown expires.
        """
        now = time.time()
        rpm_lim = self._limits["rpm"]
        tpm_lim = self._limits["tpm"]
        rpd_lim = self._limits["rpd"]
        tpd_lim = self._limits["tpd"]

        for slot in self._slots:
            self._refresh_slot(slot)

        available = [
            i for i, s in enumerate(self._slots)
            if now >= s.cooldown_until
            and s.rpm_used < int(rpm_lim * _RPM_ROTATION_THRESHOLD)
            and s.tpm_used < int(tpm_lim * _TPM_ROTATION_THRESHOLD)
            and s.rpd_used < rpd_lim
            and s.tpd_used < int(tpd_lim * 0.99)
        ]

        if available:
            # Prefer key with the most remaining minute-token budget
            return min(available, key=lambda i: self._slots[i].tpm_used)

        # All slots near/at limits — find shortest wait
        soonest_idx = 0
        soonest_wake = float("inf")
        for i, s in enumerate(self._slots):
            # Slot becomes free either when its cooldown expires or its minute window resets
            wake = max(s.cooldown_until, s.window_start + 60.0)
            if wake < soonest_wake:
                soonest_wake = wake
                soonest_idx = i

        wait = max(0.0, soonest_wake - now)
        if wait > 0:
            logger.warning(
                "All %d key(s) rate-limited. Sleeping %.1fs before retrying.",
                len(self._slots), wait,
            )
            time.sleep(wait)
            # Refresh after sleep
            for slot in self._slots:
                self._refresh_slot(slot)

        return soonest_idx

    def _mark_used(self, slot: "_KeySlot", tokens_in: int, tokens_out: int) -> None:
        """Update per-key counters after a successful request."""
        slot.rpm_used += 1
        slot.tpm_used += tokens_in + tokens_out
        slot.rpd_used += 1
        slot.tpd_used += tokens_in + tokens_out

    def _mark_cooldown(self, slot: "_KeySlot", retry_after: int) -> None:
        """Put a slot into cooldown after receiving a 429."""
        slot.cooldown_until = time.time() + retry_after
        if retry_after > 3_600:
            logger.warning(
                "Key ...%s has likely hit its DAILY limit (retry_after=%ds). "
                "It will be skipped until the window resets.",
                slot.key[-6:], retry_after,
            )
        else:
            logger.warning(
                "Key ...%s in cooldown for %ds (TPM/RPM limit).",
                slot.key[-6:], retry_after,
            )

    def _rotate_key(self) -> None:
        """Legacy shim — advance slot index by one (used in log messages)."""
        self._slot_index = self._pick_slot()
        self.api_key = self._slots[self._slot_index].key

    def _build_headers(self, key: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {key or self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/csam-project"
            headers["X-Title"] = "CSAM Research"
        return headers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if any loaded key can reach the provider endpoint."""
        for slot in self._slots:
            try:
                r = requests.get(
                    f"{self.base_url}/models",
                    headers=self._build_headers(key=slot.key),
                    timeout=10,
                )
                if r.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                continue
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
        Generate a chat completion with multi-key rotation and per-key rate tracking.

        On HTTP 429:
          - Marks the current key in cooldown (using Retry-After header).
          - Immediately switches to the next available key (no sleep if one exists).
          - Only sleeps when ALL keys are simultaneously in cooldown/at limits.
          - Raises RateLimitExceeded after max_retries exhausted attempts.

        Returns:
            Generated text string (empty string on non-429 errors).
        """
        if not self._slots:
            logger.error("No API keys configured for %s.", self.provider)
            return ""

        # Pick best available slot (handles per-key RPM/TPM/RPD/TPD checks)
        slot_idx = self._pick_slot()
        slot = self._slots[slot_idx]
        self._slot_index = slot_idx
        self.api_key = slot.key

        messages: List[Dict[str, str]] = []
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
                headers=self._build_headers(key=slot.key),
                timeout=self.timeout,
            )
            latency_ms = (time.time() - t0) * 1000

            if response.status_code == 200:
                data = response.json()
                result: str = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tok_in  = usage.get("prompt_tokens", 0)
                tok_out = usage.get("completion_tokens", 0)
                # Update per-key and aggregate counters
                self._mark_used(slot, tok_in, tok_out)
                self.total_requests   += 1
                self.total_tokens_in  += tok_in
                self.total_tokens_out += tok_out
                self.total_latency_ms += latency_ms
                logger.debug(
                    "OK key=...%s  tok=%d+%d  slot_tpm=%d/%d",
                    slot.key[-6:], tok_in, tok_out,
                    slot.tpm_used, self._limits["tpm"],
                )
                return self._strip_thinking_tags(result)

            elif response.status_code == 429:
                if _retry_count >= self.max_retries:
                    raise RateLimitExceeded(
                        f"{self.provider} returned HTTP 429 after "
                        f"{self.max_retries} retries across {len(self._slots)} key(s). "
                        "Add more API keys (GROQ_API_KEY_2 … GROQ_API_KEY_8) or "
                        "reduce request rate."
                    )
                retry_after = int(
                    response.headers.get("Retry-After", self.DEFAULT_BACKOFF)
                )
                self._mark_cooldown(slot, retry_after)
                logger.warning(
                    "429 on key ...%s (attempt %d/%d). "
                    "Switching key and retrying immediately.",
                    slot.key[-6:], _retry_count + 1, self.max_retries,
                )
                # Immediate retry — _pick_slot() will choose a fresh key or sleep
                return self.generate(
                    prompt, system_prompt, temperature, max_tokens,
                    seed=seed, _retry_count=_retry_count + 1,
                )

            elif response.status_code in (401, 400):
                # 401 = invalid/revoked key; 400 can mean org restricted
                body = response.text[:200]
                is_key_error = (
                    response.status_code == 401
                    or "organization_restricted" in body
                    or "invalid_api_key" in body
                )
                if is_key_error:
                    logger.warning(
                        "Key ...%s permanently disabled (%d: %s). Rotating.",
                        slot.key[-6:], response.status_code, body[:80],
                    )
                    self._mark_cooldown(slot, retry_after=86_400 * 365)
                    if _retry_count >= self.max_retries:
                        raise RateLimitExceeded(
                            f"All keys exhausted (key error on ...{slot.key[-6:]})."
                        )
                    return self.generate(
                        prompt, system_prompt, temperature, max_tokens,
                        seed=seed, _retry_count=_retry_count + 1,
                    )
                logger.error(
                    "%s error %d: %s",
                    self.provider, response.status_code, body,
                )
                return ""

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
        """Return accumulated API usage counters plus per-key breakdown."""
        per_key = [
            {
                "key_suffix":   s.key[-6:],
                "rpd_used":     s.rpd_used,
                "tpd_used":     s.tpd_used,
                "cooldown_sec": max(0.0, round(s.cooldown_until - time.time(), 1)),
            }
            for s in self._slots
        ]
        return {
            "provider":         self.provider,
            "model":            self.model,
            "total_requests":   self.total_requests,
            "total_tokens_in":  self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens":     self.total_tokens_in + self.total_tokens_out,
            "avg_latency_ms":   self.total_latency_ms / max(1, self.total_requests),
            "num_api_keys":     len(self._slots),
            "limits":           self._limits,
            "per_key":          per_key,
        }

    def __repr__(self) -> str:
        return f"HostedLLMService(provider={self.provider!r}, model={self.model!r})"
