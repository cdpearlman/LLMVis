"""
OpenRouter API Client

Wrapper for OpenRouter API providing text generation and embedding capabilities
for the AI chatbot feature.

Uses the OpenAI-compatible API via requests.
"""

import os
import requests
from typing import List, Dict, Optional


# =============================================================================
# GLOBAL MODEL CONFIGURATION
# =============================================================================
# Change these to switch models across the entire application

# Chat model: Gemini 2.5 Flash Lite - $0.10/$0.40 per 1M tokens, 1M context
DEFAULT_CHAT_MODEL = "google/gemini-2.5-flash-lite"

# Embedding model: text-embedding-3-small - $0.02 per 1M tokens, 1536 dimensions
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

# =============================================================================

# OpenRouter API endpoint
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# System prompt for the chatbot
SYSTEM_PROMPT = """You are a helpful AI assistant integrated into a Transformer Explanation Dashboard. 
Your role is to help users understand how transformer models work, explain the experiments 
available in the dashboard, and answer questions about machine learning concepts.

You have access to:
1. RAG documents containing information about transformers and the dashboard
2. The current state of the dashboard (selected model, prompt, analysis results)

When answering:
- Be extremely concise. Directly answer the user's question.
- Do not provide exhaustive explanations unless explicitly asked.
- At the end of your concise answer, offer to go into more detail (e.g., "Let me know if you'd like me to explain [TOPIC] in more detail.")
- Be clear and educational
- Use examples when helpful
- Reference specific dashboard features when relevant
- Format code snippets properly with markdown code blocks
- If you don't know something, say so honestly

Dashboard context will be provided in the user's messages when available."""


class OpenRouterClient:
    """Client for interacting with OpenRouter API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._initialized = False
        
        if self.api_key:
            self._initialize()
    
    def _initialize(self):
        """Initialize the OpenRouter API client."""
        if not self.api_key:
            return
        
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://transformer-dashboard.local",  # Optional: for rankings
            "X-Title": "Transformer Explanation Dashboard"  # Optional: for rankings
        }
        self._initialized = True
    
    @property
    def is_available(self) -> bool:
        """Check if the OpenRouter API is available and configured."""
        return self._initialized and self.api_key is not None
    
    def generate_response(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        rag_context: Optional[str] = None,
        dashboard_context: Optional[Dict] = None
    ) -> str:
        """
        Generate a response using OpenRouter.
        
        Args:
            user_message: The user's message
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            rag_context: Retrieved context from RAG documents
            dashboard_context: Current dashboard state (model, prompt, results)
            
        Returns:
            Generated response text
        """
        if not self.is_available:
            return "Sorry, the AI assistant is not available. Please check that the OPENROUTER_API_KEY environment variable is set."
        
        try:
            # Build the full prompt with context
            full_message = self._build_prompt(user_message, rag_context, dashboard_context)
            
            # Build messages array with system prompt and history
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            
            # Add chat history
            if chat_history:
                for msg in chat_history[-10:]:  # Keep last 10 messages for context
                    role = "user" if msg.get("role") == "user" else "assistant"
                    messages.append({
                        "role": role,
                        "content": msg.get("content", "")
                    })
            
            # Add the current user message
            messages.append({"role": "user", "content": full_message})
            
            # Make API request
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=self._headers,
                json={
                    "model": DEFAULT_CHAT_MODEL,
                    "messages": messages
                },
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get("message", str(e))
                except:
                    pass
            
            if "rate" in error_msg.lower() or "429" in error_msg:
                return f"The AI service is currently rate limited. Please try again in a moment. {error_msg}"
            elif "401" in error_msg or "invalid" in error_msg.lower():
                return "Invalid API key. Please check your OPENROUTER_API_KEY configuration."
            else:
                print(f"OpenRouter API error: {e}")
                return f"Sorry, I encountered an error: {error_msg}"
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
            
    def generate_stream(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        rag_context: Optional[str] = None,
        dashboard_context: Optional[Dict] = None
    ):
        """
        Generate a streaming response using OpenRouter.
        Yields text chunks as they arrive.
        """
        if not self.is_available:
            yield "Sorry, the AI assistant is not available. Please check that the OPENROUTER_API_KEY environment variable is set."
            return
            
        try:
            full_message = self._build_prompt(user_message, rag_context, dashboard_context)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if chat_history:
                for msg in chat_history[-10:]:
                    role = "user" if msg.get("role") == "user" else "assistant"
                    messages.append({"role": role, "content": msg.get("content", "")})
            messages.append({"role": "user", "content": full_message})
            
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=self._headers,
                json={
                    "model": DEFAULT_CHAT_MODEL,
                    "messages": messages,
                    "stream": True
                },
                timeout=60,
                stream=True
            )
            response.raise_for_status()
            
            import json
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: ') and line != 'data: [DONE]':
                        try:
                            data = json.loads(line[6:])
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            continue
                            
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if e.response is not None:
                try:
                    error_data = e.response.json()
                    error_msg = error_data.get("error", {}).get("message", str(e))
                except:
                    pass
            if "rate" in error_msg.lower() or "429" in error_msg:
                yield f"The AI service is currently rate limited. Please try again in a moment. {error_msg}"
            elif "401" in error_msg or "invalid" in error_msg.lower():
                yield "Invalid API key. Please check your OPENROUTER_API_KEY configuration."
            else:
                print(f"OpenRouter API stream error: {e}")
                yield f"Sorry, I encountered an error: {error_msg}"
        except Exception as e:
            print(f"OpenRouter API stream error: {e}")
            yield f"Sorry, I encountered an error: {str(e)}"
    
    def _build_prompt(
        self,
        user_message: str,
        rag_context: Optional[str] = None,
        dashboard_context: Optional[Dict] = None
    ) -> str:
        """Build the full prompt with context."""
        parts = []
        
        # Add dashboard context if available
        if dashboard_context:
            context_str = self._format_dashboard_context(dashboard_context)
            if context_str:
                parts.append(f"**Current Dashboard State:**\n{context_str}\n")
        
        # Add RAG context if available
        if rag_context:
            parts.append(f"**Relevant Documentation:**\n{rag_context}\n")
        
        # Add the user's message
        parts.append(f"**User Question:**\n{user_message}")
        
        return "\n".join(parts)
    
    def _format_dashboard_context(self, context: Dict) -> str:
        """Format dashboard context for the prompt."""
        lines = []
        
        if context.get("model"):
            lines.append(f"- Selected Model: {context['model']}")
        
        if context.get("prompt"):
            lines.append(f"- Input Prompt: \"{context['prompt']}\"")
        
        if context.get("predicted_token"):
            prob = context.get("predicted_probability", 0)
            lines.append(f"- Predicted Next Token: \"{context['predicted_token']}\" (probability: {prob:.1%})")
        
        if context.get("top_predictions"):
            top = context["top_predictions"][:5]
            tokens_str = ", ".join([f"{t['token']} ({t['probability']:.1%})" for t in top])
            lines.append(f"- Top Predictions: {tokens_str}")
        
        if context.get("ablated_heads"):
            heads_str = ", ".join([f"L{h['layer']}H{h['head']}" for h in context["ablated_heads"]])
            lines.append(f"- Ablated Attention Heads: {heads_str}")
        
        return "\n".join(lines)
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding vector for text using OpenRouter Embedding API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not self.is_available:
            return None
        
        try:
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/embeddings",
                headers=self._headers,
                json={
                    "model": DEFAULT_EMBEDDING_MODEL,
                    "input": text
                },
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            return data["data"][0]["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def get_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Get embedding vector for a query.
        
        Note: OpenRouter doesn't have separate task types for embeddings,
        so this calls the same endpoint as get_embedding.
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        return self.get_embedding(query)


# Singleton instance
_client_instance: Optional[OpenRouterClient] = None


def get_openrouter_client() -> OpenRouterClient:
    """Get or create the singleton OpenRouter client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = OpenRouterClient()
    return _client_instance


def generate_response(
    user_message: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    rag_context: Optional[str] = None,
    dashboard_context: Optional[Dict] = None
) -> str:
    """
    Convenience function to generate a response.
    
    Args:
        user_message: The user's message
        chat_history: Previous chat messages
        rag_context: Retrieved RAG context
        dashboard_context: Current dashboard state
        
    Returns:
        Generated response text
    """
    client = get_openrouter_client()
    return client.generate_response(user_message, chat_history, rag_context, dashboard_context)


def generate_stream(
    user_message: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
    rag_context: Optional[str] = None,
    dashboard_context: Optional[Dict] = None
):
    """
    Convenience function to generate a streaming response.
    
    Args:
        user_message: The user's message
        chat_history: Previous chat messages
        rag_context: Retrieved RAG context
        dashboard_context: Current dashboard state
        
    Returns:
        Generator yielding text chunks
    """
    client = get_openrouter_client()
    return client.generate_stream(user_message, chat_history, rag_context, dashboard_context)


def get_embedding(text: str) -> Optional[List[float]]:
    """Convenience function to get document embedding."""
    client = get_openrouter_client()
    return client.get_embedding(text)


def get_query_embedding(query: str) -> Optional[List[float]]:
    """Convenience function to get query embedding."""
    client = get_openrouter_client()
    return client.get_query_embedding(query)


# Backward compatibility aliases (for gradual migration)
GeminiClient = OpenRouterClient
get_gemini_client = get_openrouter_client
