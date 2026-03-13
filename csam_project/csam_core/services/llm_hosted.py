"""
Hosted LLM Service - Drop-in replacement for LLMService using cloud providers.

Supports OpenAI-compatible APIs:
- Groq (https://api.groq.com/openai/v1)
- Cerebras (https://api.cerebras.ai/v1)
- SambaNova (https://api.sambanova.ai/v1)

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

# Provider configurations
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "rate_limit_rpm": 30,
        "models": {
            "llama-8b": "llama-3.1-8b-instant",
            "llama-70b": "llama-3.3-70b-versatile",
            "qwen-32b": "qwen/qwen3-32b",
            "gpt-oss-120b": "openai/gpt-oss-120b",
            "gpt-oss-20b": "openai/gpt-oss-20b",
            "llama-4-maverick": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
        }
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "env_key": "CEREBAS_API_KEY",  # Note: typo preserved from .env
        "rate_limit_rpm": 30,
        "models": {
            "llama-70b": "llama-3.3-70b",
        }
    },
    "sambanova": {
        "base_url": "https://api.sambanova.ai/v1",
        "env_key": "SAMBANOVA_API_KEY",
        "rate_limit_rpm": 20,
        "models": {
            "llama-8b": "Meta-Llama-3.1-8B-Instruct",
            "llama-70b": "Meta-Llama-3.3-70B-Instruct",
        }
    }
}


class HostedLLMService:
    """
    Drop-in replacement for LLMService using hosted OpenAI-compatible APIs.
    
    Usage:
        llm = HostedLLMService(provider="groq", model="llama-3.1-8b-instant")
        llm = HostedLLMService(provider="groq", model="llama-3.3-70b-versatile")
        llm = HostedLLMService(provider="cerebras", model="llama-3.3-70b")
    """
    
    def __init__(
        self,
        provider: str = "groq",
        model: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        timeout: int = 120,
        rate_limit_rpm: Optional[int] = None
    ):
        """
        Initialize the hosted LLM service.
        
        Args:
            provider: Provider name ("groq", "cerebras", "sambanova")
            model: Model ID (provider-specific)
            api_key: API key (if None, reads from environment)
            timeout: Request timeout in seconds
            rate_limit_rpm: Override rate limit (requests per minute)
        """
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Options: {list(PROVIDERS.keys())}")
        
        self.provider = provider
        self.provider_config = PROVIDERS[provider]
        self.base_url = self.provider_config["base_url"]
        self.model = model
        self.timeout = timeout
        
        # Rate limiting
        self.rate_limit_rpm = rate_limit_rpm or self.provider_config["rate_limit_rpm"]
        self.min_interval = 60.0 / self.rate_limit_rpm  # seconds between requests
        self._last_request_time = 0
        
        # API key(s) — supports rotation through multiple keys
        self._api_keys: List[str] = []
        self._key_index = 0
        if api_key:
            self._api_keys = [api_key]
        else:
            env_key = self.provider_config["env_key"]
            # Collect all matching keys: GROQ_API_KEY, GROQ_API_KEY_2, ...
            base = os.environ.get(env_key, "")
            if base:
                self._api_keys.append(base)
            for suffix in range(2, 20):
                extra = os.environ.get(f"{env_key}_{suffix}", "")
                if extra:
                    self._api_keys.append(extra)
            if not self._api_keys:
                logger.warning(f"No API key found for {provider}. Set ${env_key} environment variable.")
            else:
                logger.info(f"Loaded {len(self._api_keys)} API key(s) for {provider}")
        self.api_key = self._api_keys[0] if self._api_keys else ""
        
        # Stats tracking
        self.total_requests = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_latency_ms = 0
        
        logger.info(f"HostedLLMService initialized: provider={provider}, model={model}")
    
    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """Remove thinking/reasoning tags from model output (e.g., Qwen3's <think>...</think>)."""
        if not text:
            return text
        # Remove closed <think>...</think> blocks (including multi-line)
        stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        # Handle unclosed <think>... (truncated by max_tokens before </think>)
        if '<think>' in stripped:
            stripped = re.sub(r'<think>.*', '', stripped, flags=re.DOTALL).strip()
        # If stripping removed everything, return original
        return stripped if stripped else text
    
    def _rate_limit_wait(self):
        """Wait if necessary to respect rate limits."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()
    
    def is_available(self) -> bool:
        """Check if the hosted API is reachable."""
        if not self.api_key:
            return False
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate text completion using OpenAI-compatible chat API.
        
        Same interface as LLMService.generate().
        """
        self._rate_limit_wait()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        if seed is not None:
            payload["seed"] = seed
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            t0 = time.time()
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            latency_ms = (time.time() - t0) * 1000
            
            if response.status_code == 200:
                data = response.json()
                result = data["choices"][0]["message"]["content"]
                
                # Track usage
                usage = data.get("usage", {})
                self.total_requests += 1
                self.total_tokens_in += usage.get("prompt_tokens", 0)
                self.total_tokens_out += usage.get("completion_tokens", 0)
                self.total_latency_ms += latency_ms
                
                # Strip thinking tags (e.g., Qwen3's <think>...</think>)
                result = self._strip_thinking_tags(result)
                return result
            elif response.status_code == 429:
                # Rate limited — rotate key and retry
                retry_after = int(response.headers.get("Retry-After", 2))
                self._rotate_key()
                logger.warning(f"Rate limited by {self.provider}. Waiting {retry_after}s... (rotated to key {self._key_index + 1}/{len(self._api_keys)})")
                time.sleep(retry_after)
                return self.generate(prompt, system_prompt, temperature, max_tokens, seed=seed)
            else:
                logger.error(f"{self.provider} error: {response.status_code} - {response.text[:200]}")
                return ""
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout waiting for {self.provider} ({self.timeout}s)")
            return ""
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to {self.provider}: {e}")
            return ""
    
    def summarize(self, memories: List[str]) -> str:
        """Summarize a list of memories into a concise summary. Same as LLMService."""
        memories_text = "\n".join([f"- {m}" for m in memories])
        
        prompt = f"""Summarize the following memories into a single concise statement.
Focus on the key facts and patterns. Be brief.

Memories:
{memories_text}

Summary:"""

        system = "You are a memory consolidation system. Produce concise, factual summaries."
        return self.generate(prompt, system_prompt=system, temperature=0.3, max_tokens=100)
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities and relationships from text. Same as LLMService."""
        prompt = f"""Extract entities and relationships from this text.

Text: {text}

Respond in this exact JSON format:
{{"entities": [{{"name": "...", "type": "Person|Place|Object|Concept"}}], "relationships": [{{"source": "...", "target": "...", "type": "..."}}]}}

JSON:"""

        system = "You are an entity extraction system. Output valid JSON only."
        response = self.generate(prompt, system_prompt=system, temperature=0.1, max_tokens=500)
        
        try:
            # Strip markdown code fences if present
            cleaned = response.strip()
            if cleaned.startswith('```'):
                first_newline = cleaned.find('\n')
                if first_newline > 0:
                    cleaned = cleaned[first_newline + 1:]
                if cleaned.rstrip().endswith('```'):
                    cleaned = cleaned.rstrip()[:-3].rstrip()
            
            # Find the outermost JSON object
            json_start = cleaned.find('{')
            if json_start >= 0:
                depth = 0
                json_end = -1
                for i in range(json_start, len(cleaned)):
                    if cleaned[i] == '{':
                        depth += 1
                    elif cleaned[i] == '}':
                        depth -= 1
                        if depth == 0:
                            json_end = i + 1
                            break
                
                if json_end > json_start:
                    json_str = cleaned[json_start:json_end]
                    return json.loads(json_str)
                else:
                    # Truncated JSON -- close open structures
                    partial = cleaned[json_start:].rstrip().rstrip(',')
                    open_brackets = partial.count('[') - partial.count(']')
                    open_braces = partial.count('{') - partial.count('}')
                    partial += ']' * open_brackets + '}' * open_braces
                    return json.loads(partial)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse entity extraction response: {response[:200]}...")
        
        return {"entities": [], "relationships": []}
    
    def generate_response(
        self,
        context: str,
        user_message: str,
        persona: Optional[str] = None,
        mode: str = "chat",
        seed: Optional[int] = None
    ) -> str:
        """Generate NPC response given context. Same as LLMService."""
        if mode == "qa":
            prompt = f"""Answer the question based ONLY on the context below. Be extremely concise.
            
Context:
{context}

Question: {user_message}

Answer:"""
            system = "You are a precise database. Output only the requested date, name, or fact. Do not use full sentences unless necessary."
            temperature = 0.1
        else:
            prompt = f"""Based on the following context from your memory, respond to the user.

Context from memory:
{context}

User says: {user_message}

Your response:"""
            system = persona or "You are a helpful NPC with a good memory. Be friendly and reference past conversations when relevant."
            temperature = 0.7
        
        return self.generate(prompt, system_prompt=system, temperature=temperature, max_tokens=150, seed=seed)
    
    def _rotate_key(self):
        """Rotate to the next API key."""
        if len(self._api_keys) > 1:
            self._key_index = (self._key_index + 1) % len(self._api_keys)
            self.api_key = self._api_keys[self._key_index]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            "provider": self.provider,
            "model": self.model,
            "total_requests": self.total_requests,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens": self.total_tokens_in + self.total_tokens_out,
            "avg_latency_ms": self.total_latency_ms / max(1, self.total_requests),
            "num_api_keys": len(self._api_keys),
        }
    
    def __repr__(self):
        return f"HostedLLMService(provider={self.provider}, model={self.model})"
