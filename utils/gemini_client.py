"""
Gemini API Client

Wrapper for Google Gemini API providing text generation and embedding capabilities
for the AI chatbot feature.

Uses the new google-genai SDK (migrated from deprecated google-generativeai).
"""

import os
from typing import List, Dict, Optional
from google import genai
from google.genai import types


# Default model configuration
DEFAULT_GENERATION_MODEL = "gemini-2.0-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

# System prompt for the chatbot
SYSTEM_PROMPT = """You are a helpful AI assistant integrated into a Transformer Explanation Dashboard. 
Your role is to help users understand how transformer models work, explain the experiments 
available in the dashboard, and answer questions about machine learning concepts.

You have access to:
1. RAG documents containing information about transformers and the dashboard
2. The current state of the dashboard (selected model, prompt, analysis results)

When answering:
- Be clear and educational
- Use examples when helpful
- Reference specific dashboard features when relevant
- Format code snippets properly with markdown code blocks
- If you don't know something, say so honestly

Dashboard context will be provided in the user's messages when available."""


class GeminiClient:
    """Client for interacting with Google Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._initialized = False
        self._client = None
        
        if self.api_key:
            self._initialize()
    
    def _initialize(self):
        """Initialize the Gemini API client."""
        if not self.api_key:
            return
        
        try:
            # Create the centralized client object (new SDK architecture)
            self._client = genai.Client(api_key=self.api_key)
            self._initialized = True
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
            self._initialized = False
    
    @property
    def is_available(self) -> bool:
        """Check if the Gemini API is available and configured."""
        return self._initialized and self.api_key is not None
    
    def generate_response(
        self,
        user_message: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        rag_context: Optional[str] = None,
        dashboard_context: Optional[Dict] = None
    ) -> str:
        """
        Generate a response using Gemini.
        
        Args:
            user_message: The user's message
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            rag_context: Retrieved context from RAG documents
            dashboard_context: Current dashboard state (model, prompt, results)
            
        Returns:
            Generated response text
        """
        if not self.is_available:
            return "Sorry, the AI assistant is not available. Please check that the GEMINI_API_KEY environment variable is set."
        
        try:
            # Build the full prompt with context
            full_message = self._build_prompt(user_message, rag_context, dashboard_context)
            
            # Convert chat history to new SDK format
            history = []
            if chat_history:
                for msg in chat_history[-10:]:  # Keep last 10 messages for context
                    role = "user" if msg.get("role") == "user" else "model"
                    history.append({
                        "role": role,
                        "parts": [{"text": msg.get("content", "")}]
                    })
            
            # Create chat session with system instruction and send message
            chat = self._client.chats.create(
                model=DEFAULT_GENERATION_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                ),
                history=history
            )
            response = chat.send_message(message=full_message)
            
            return response.text
            
        except Exception as e:
            error_msg = str(e)
            if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                return f"The AI service is currently rate limited. Please try again in a moment. {error_msg}"
            elif "invalid" in error_msg.lower() and "key" in error_msg.lower():
                return "Invalid API key. Please check your GEMINI_API_KEY configuration."
            else:
                print(f"Gemini API error: {e}")
                return f"Sorry, I encountered an error: {error_msg}"
    
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
        Get embedding vector for text using Gemini Embedding API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not self.is_available:
            return None
        
        try:
            result = self._client.models.embed_content(
                model=DEFAULT_EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )
            # New SDK returns embeddings as a list, get the first one
            return result.embeddings[0].values
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def get_query_embedding(self, query: str) -> Optional[List[float]]:
        """
        Get embedding vector for a query (uses different task type for better retrieval).
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as list of floats, or None if failed
        """
        if not self.is_available:
            return None
        
        try:
            result = self._client.models.embed_content(
                model=DEFAULT_EMBEDDING_MODEL,
                contents=query,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY"
                )
            )
            # New SDK returns embeddings as a list, get the first one
            return result.embeddings[0].values
        except Exception as e:
            print(f"Query embedding error: {e}")
            return None


# Singleton instance
_client_instance: Optional[GeminiClient] = None


def get_gemini_client() -> GeminiClient:
    """Get or create the singleton Gemini client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = GeminiClient()
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
    client = get_gemini_client()
    return client.generate_response(user_message, chat_history, rag_context, dashboard_context)


def get_embedding(text: str) -> Optional[List[float]]:
    """Convenience function to get document embedding."""
    client = get_gemini_client()
    return client.get_embedding(text)


def get_query_embedding(query: str) -> Optional[List[float]]:
    """Convenience function to get query embedding."""
    client = get_gemini_client()
    return client.get_query_embedding(query)
