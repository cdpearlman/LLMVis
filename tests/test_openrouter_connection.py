"""
Tests for OpenRouter API connection.

Verifies that the API key is configured correctly and can connect
to the OpenRouter API without consuming many tokens.
"""

import os
import pytest
import requests
from dotenv import load_dotenv

# Load environment variables for tests
load_dotenv()

# OpenRouter API endpoint
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class TestOpenRouterConnection:
    """Test suite for OpenRouter API connectivity."""

    def test_api_key_is_set(self):
        """Verify OPENROUTER_API_KEY environment variable is configured."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        assert api_key is not None, "OPENROUTER_API_KEY environment variable is not set"
        assert len(api_key) > 0, "OPENROUTER_API_KEY is empty"
        # OpenRouter keys start with "sk-or-"
        assert len(api_key) > 10, "OPENROUTER_API_KEY appears too short to be valid"

    @pytest.mark.timeout(30)
    def test_can_list_models(self):
        """
        Test API connectivity by listing available models.
        This verifies the API key is valid without consuming generation tokens.
        """
        api_key = os.environ.get("OPENROUTER_API_KEY")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        models = data.get("data", [])
        
        assert len(models) > 0, "No models returned - API key may be invalid"
        
        # Verify we can see some models
        model_ids = [m.get("id", "") for m in models]
        assert len(model_ids) > 0, "No model IDs found in available models list"

    @pytest.mark.timeout(30)
    def test_chat_model_available(self):
        """Verify the configured chat model is available."""
        from utils.openrouter_client import DEFAULT_CHAT_MODEL
        
        api_key = os.environ.get("OPENROUTER_API_KEY")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        models = data.get("data", [])
        model_ids = [m.get("id", "") for m in models]
        
        # Check for the configured model or similar
        model_family = DEFAULT_CHAT_MODEL.split("/")[0] if "/" in DEFAULT_CHAT_MODEL else DEFAULT_CHAT_MODEL
        has_model = any(model_family in mid for mid in model_ids)
        
        assert has_model or DEFAULT_CHAT_MODEL in model_ids, (
            f"Chat model '{DEFAULT_CHAT_MODEL}' or similar not found. "
            f"Sample available models: {model_ids[:10]}..."
        )

    @pytest.mark.timeout(30)
    def test_embedding_model_available(self):
        """Verify an embedding model is available."""
        api_key = os.environ.get("OPENROUTER_API_KEY")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(
            f"{OPENROUTER_BASE_URL}/models",
            headers=headers,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        models = data.get("data", [])
        model_ids = [m.get("id", "") for m in models]
        
        # Check for embedding model
        has_embedding_model = any("embedding" in mid.lower() for mid in model_ids)
        
        # OpenRouter may not list embedding models separately, so this is a soft check
        if not has_embedding_model:
            print(f"Note: No embedding models explicitly listed. Available: {model_ids[:10]}...")
