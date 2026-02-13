"""
LLM Service - Wrapper for local LLM via Ollama.

This module provides access to a local LLM for:
- Summarization (consolidating memories)
- Entity extraction (building knowledge graph)
- Response generation (NPC dialogue)

Uses Ollama which runs entirely locally.
"""

import requests
import json
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class LLMService:
    """
    Wrapper for Ollama API.
    
    Ollama is a local LLM server that runs models like llama3.2, mistral, phi-3.
    No cloud API keys needed - everything runs on your machine.
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:1b",
        timeout: int = 60
    ):
        """
        Initialize the LLM service.
        
        Args:
            base_url: Ollama server URL (default localhost:11434)
            model: Model name to use (default llama3.2:1b for max speed)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """List available models on Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
            return []
        except requests.exceptions.RequestException:
            return []
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
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
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return ""
    
    def summarize(self, memories: List[str]) -> str:
        """
        Summarize a list of memories into a concise summary.
        
        Args:
            memories: List of memory texts to summarize
            
        Returns:
            Concise summary of the memories
        """
        memories_text = "\n".join([f"- {m}" for m in memories])
        
        prompt = f"""Summarize the following memories into a single concise statement.
Focus on the key facts and patterns. Be brief.

Memories:
{memories_text}

Summary:"""

        system = "You are a memory consolidation system. Produce concise, factual summaries."
        
        return self.generate(prompt, system_prompt=system, temperature=0.3, max_tokens=100)
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary with 'entities' and 'relationships' lists
        """
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
                # Remove opening fence (```json or ```)
                first_newline = cleaned.find('\n')
                if first_newline > 0:
                    cleaned = cleaned[first_newline + 1:]
                # Remove closing fence
                if cleaned.rstrip().endswith('```'):
                    cleaned = cleaned.rstrip()[:-3].rstrip()
            
            # Find the outermost JSON object
            json_start = cleaned.find('{')
            if json_start >= 0:
                # Count braces to find matching close
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
                    # Truncated JSON -- try to repair by closing open structures
                    partial = cleaned[json_start:]
                    # Close any open arrays and the root object
                    partial = partial.rstrip().rstrip(',')
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
        mode: str = "chat"
    ) -> str:
        """
        Generate NPC response given context and user message.
        
        Args:
            context: Retrieved memories/knowledge as context
            user_message: What the user/player said
            persona: Optional persona description for the NPC
            
        Returns:
            NPC's response
        """
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
        
        return self.generate(prompt, system_prompt=system, temperature=temperature, max_tokens=150)


# Singleton instance
_default_service = None

def get_llm_service(
    base_url: str = "http://localhost:11434",
    model: str = "llama3.2:1b"
) -> LLMService:
    """Get the default LLM service instance."""
    global _default_service
    if _default_service is None:
        _default_service = LLMService(base_url, model)
    return _default_service
