"""
Async LLM Service - Async wrapper for Ollama API (CPU-only)

Keeps LLM on CPU to reduce GPU memory pressure while embeddings run on GPU.
Provides concurrent request handling for multi-agent scenarios.
"""

import aiohttp
import asyncio
import json
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class AsyncLLMService:
    """
    Async wrapper for Ollama API with concurrent request support.
    
    Features:
    - Async HTTP requests using aiohttp
    - Connection pooling for efficiency
    - Concurrent request limiting (semaphore)
    - CPU-only to reduce GPU load
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:1b",
        timeout: int = 60,
        max_concurrent: int = 4
    ):
        """
        Initialize async LLM service.
        
        Args:
            base_url: Ollama server URL
            model: Model name to use
            timeout: Request timeout in seconds
            max_concurrent: Max concurrent LLM requests (limited by Ollama)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self._session
    
    async def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False
    
    async def list_models(self) -> List[str]:
        """List available models on Ollama server."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return [m["name"] for m in data.get("models", [])]
                return []
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return []
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        Generate text completion (async).
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # Limit concurrent requests
        async with self._semaphore:
            try:
                session = await self._get_session()
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "")
                    else:
                        text = await response.text()
                        logger.error(f"Ollama error: {response.status} - {text}")
                        return ""
                        
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"Failed to connect to Ollama: {e}")
                return ""
    
    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> List[str]:
        """
        Generate responses for multiple prompts concurrently.
        
        Args:
            prompts: List of prompts
            system_prompt: System prompt for all
            temperature: Sampling temperature
            max_tokens: Max tokens per generation
            
        Returns:
            List of generated responses (same order as prompts)
        """
        tasks = [
            self.generate(p, system_prompt, temperature, max_tokens)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)
    
    async def summarize(self, memories: List[str]) -> str:
        """
        Summarize memories into concise summary.
        
        Args:
            memories: List of memory texts
            
        Returns:
            Concise summary
        """
        memories_text = "\n".join([f"- {m}" for m in memories])
        
        prompt = f"""Summarize the following memories into a single concise statement.
Focus on the key facts and patterns. Be brief.

Memories:
{memories_text}

Summary:"""

        system = "You are a memory consolidation system. Produce concise, factual summaries."
        
        return await self.generate(
            prompt,
            system_prompt=system,
            temperature=0.3,
            max_tokens=100
        )
    
    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Text to extract from
            
        Returns:
            Dict with 'entities' and 'relationships' lists
        """
        prompt = f"""Extract entities and relationships from this text.

Text: {text}

Respond in this exact JSON format:
{{"entities": [{{"name": "...", "type": "Person|Place|Object|Concept"}}], "relationships": [{{"source": "...", "target": "...", "type": "..."}}]}}

JSON:"""

        system = "You are an entity extraction system. Output valid JSON only."
        
        response = await self.generate(
            prompt,
            system_prompt=system,
            temperature=0.1,
            max_tokens=200
        )
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse entity extraction: {response}")
        
        return {"entities": [], "relationships": []}
    
    async def generate_response(
        self,
        context: str,
        user_message: str,
        persona: Optional[str] = None
    ) -> str:
        """
        Generate NPC response.
        
        Args:
            context: Retrieved memories/knowledge
            user_message: What user said
            persona: Optional NPC persona
            
        Returns:
            NPC's response
        """
        prompt = f"""Based on the following context from your memory, respond to the user.

Context from memory:
{context}

User says: {user_message}

Your response:"""

        system = persona or "You are a helpful NPC with a good memory. Be friendly and reference past conversations when relevant."
        
        return await self.generate(
            prompt,
            system_prompt=system,
            temperature=0.7,
            max_tokens=150
        )
    
    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._session and not self._session.closed:
            # Try to close gracefully
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.create_task(self.close())
            except RuntimeError:
                pass


# Backward-compatible synchronous wrapper
class SyncLLMService:
    """
    Synchronous wrapper around AsyncLLMService.
    
    Provides same API as original LLM Service but with async internals.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:1b",
        timeout: int = 60
    ):
        self._async_service = AsyncLLMService(
            base_url=base_url,
            model=model,
            timeout=timeout
        )
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
    
    def is_available(self) -> bool:
        """Check if Ollama is available (sync)."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self._async_service.is_available())
            return result
        finally:
            loop.close()
    
    def list_models(self) -> List[str]:
        """List models (sync)."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self._async_service.list_models())
            return result
        finally:
            loop.close()
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """Generate text (sync)."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self._async_service.generate(prompt, system_prompt, temperature, max_tokens)
            )
            return result
        finally:
            loop.close()
    
    def summarize(self, memories: List[str]) -> str:
        """Summarize memories (sync)."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self._async_service.summarize(memories))
            return result
        finally:
            loop.close()
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities (sync)."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self._async_service.extract_entities(text))
            return result
        finally:
            loop.close()
    
    def generate_response(
        self,
        context: str,
        user_message: str,
        persona: Optional[str] = None
    ) -> str:
        """Generate response (sync)."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self._async_service.generate_response(context, user_message, persona)
            )
            return result
        finally:
            loop.close()
