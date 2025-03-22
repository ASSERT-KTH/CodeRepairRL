"""Minimal API for batched LLM completions from Anthropic/OpenRouter."""
import os
import asyncio
from typing import List

import httpx
from dotenv import load_dotenv; load_dotenv()
from tqdm.auto import tqdm


class LLM:
    """LLM client for making batched API requests to Anthropic/OpenRouter."""
    
    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-5-haiku-latest",
        max_tokens: int = 4096,
        temperature: float = 0.2,
        batch_size: int = 10,
        retry_limit: int = 3
    ):
        """Initialize the LLM client with configuration."""
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.retry_limit = retry_limit
        
        # Get API key
        self.api_key = os.environ.get(f"{provider.upper()}_API_KEY")
        if not self.api_key:
            raise ValueError(f"{provider.upper()}_API_KEY not found in environment")
            
        # Set API URL
        self.url = {
            "anthropic": "https://api.anthropic.com/v1/messages",
            "openrouter": "https://openrouter.ai/api/v1/chat/completions"
        }.get(provider)
        
        if not self.url:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Set headers
        self.headers = {
            "anthropic": {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            },
            "openrouter": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/ASSERT-KTH/TTC"
            }
        }.get(provider)
    
    async def request(self, client: httpx.AsyncClient, prompt: str, system_prompt: str) -> str:
        """Make a single API request with retries."""
        payload = {
            "anthropic": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system_prompt,
                "messages": [{"role": "user", "content": prompt}]
            },
            "openrouter": {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }
        }.get(self.provider)
        
        for attempt in range(self.retry_limit):
            try:
                response = await client.post(self.url, headers=self.headers, json=payload, timeout=60.0)
                response.raise_for_status()
                data = response.json()
                
                if self.provider == "anthropic":
                    return data.get("content", [{}])[0].get("text", "")
                else:  # openrouter
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
            except Exception as e:
                if attempt == self.retry_limit - 1:  # Last attempt
                    raise RuntimeError(f"API request failed after {self.retry_limit} attempts: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def batch_process(self, prompts: List[str], system_prompt: str) -> List[str]:
        """Process a batch of requests concurrently."""
        async with httpx.AsyncClient() as client:
            tasks = [self.request(client, prompt, system_prompt) for prompt in prompts]
            return await asyncio.gather(*tasks)
    
    def generate_completions(self, prompts: List[str], system_prompt: str) -> List[str]:
        """Generate completions for multiple prompts in efficient batches."""
        if not prompts:
            return []
        
        results = []
        
        with tqdm(total=len(prompts), desc="Generating completions") as pbar:
            for i in range(0, len(prompts), self.batch_size):
                batch = prompts[i:i + self.batch_size]
                batch_results = asyncio.run(self.batch_process(batch, system_prompt))
                print(batch_results)
                results.extend(batch_results)
                pbar.update(len(batch))
        
        return results