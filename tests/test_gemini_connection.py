"""
Tests for Gemini API connection.

Verifies that the API key is configured correctly and can connect
to the Gemini API without consuming generation tokens.

Note: These tests avoid importing utils.gemini_client where possible
to prevent slow tensorflow/jax imports from the google-generativeai package.
"""

import os
import pytest
from dotenv import load_dotenv

# Load environment variables for tests
load_dotenv()


class TestGeminiConnection:
    """Test suite for Gemini API connectivity."""

    def test_api_key_is_set(self):
        """Verify GEMINI_API_KEY environment variable is configured."""
        api_key = os.environ.get("GEMINI_API_KEY")
        assert api_key is not None, "GEMINI_API_KEY environment variable is not set"
        assert len(api_key) > 0, "GEMINI_API_KEY is empty"
        # Basic format check (Gemini keys are typically 39+ characters)
        assert len(api_key) > 10, "GEMINI_API_KEY appears too short to be valid"

    @pytest.mark.timeout(30)
    def test_can_list_models(self):
        """
        Test API connectivity by listing available models.
        This verifies the API key is valid without consuming generation tokens.
        """
        import google.generativeai as genai
        
        api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        # List models - this is a read-only API call that validates the key
        models = list(genai.list_models())
        
        assert len(models) > 0, "No models returned - API key may be invalid"
        
        # Verify we can see generation models
        model_names = [m.name for m in models]
        has_gemini_model = any("gemini" in name.lower() for name in model_names)
        assert has_gemini_model, "No Gemini models found in available models list"

    @pytest.mark.timeout(30)
    def test_flash_model_available(self):
        """Verify a Gemini Flash model (used by default) is available."""
        import google.generativeai as genai
        
        api_key = os.environ.get("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        models = list(genai.list_models())
        model_names = [m.name for m in models]
        
        # Check for flash model variants (our default is gemini-2.0-flash)
        has_flash_model = any("flash" in name.lower() for name in model_names)
        assert has_flash_model, (
            f"No Gemini Flash models available. "
            f"Available models: {model_names[:10]}..."
        )
