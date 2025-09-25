"""
Caching service using dash-extensions ServersideOutputTransform.
Provides session-scoped caching for heavy computation results.
"""

import hashlib
import json
import logging
import uuid
from typing import Dict, Any, Optional, Tuple
from dash_extensions.enrich import ServersideOutputTransform

from ..constants import DEFAULT_CACHE_KEY_PREFIX, TOP_K

logger = logging.getLogger(__name__)

# Global session registry
_session_registry: Dict[str, str] = {}

def get_session_id() -> str:
    """
    Get or create a session ID for the current session.
    
    Returns:
        Session ID string
    """
    # In a real deployment, this would be tied to actual session management
    # For now, we'll use a simple global session
    if 'current_session' not in _session_registry:
        _session_registry['current_session'] = str(uuid.uuid4())
        logger.info(f"Created new session: {_session_registry['current_session']}")
    
    return _session_registry['current_session']

def compose_cache_key(params: Dict[str, Any]) -> str:
    """
    Create a deterministic cache key from parameters.
    
    Args:
        params: Dictionary with caching parameters
        
    Returns:
        Hex-encoded hash string for use as cache key
    """
    # Extract and normalize parameters
    cache_params = {
        'model_name': params.get('model_name', ''),
        'prompt': params.get('prompt', '').strip().lower(),
        'attention_pattern': params.get('attention_pattern', ''),
        'mlp_pattern': params.get('mlp_pattern', ''),
        'norm_param_name': params.get('norm_param_name', ''),
        'logit_lens_param_name': params.get('logit_lens_param_name', ''),
        'top_k': params.get('top_k', TOP_K),
        'session_id': get_session_id()
    }
    
    # Create deterministic string representation
    cache_string = json.dumps(cache_params, sort_keys=True, separators=(',', ':'))
    
    # Generate hash
    cache_hash = hashlib.sha256(cache_string.encode('utf-8')).hexdigest()[:16]
    
    # Combine with prefix
    cache_key = f"{DEFAULT_CACHE_KEY_PREFIX}_{cache_hash}"
    
    logger.info(f"Generated cache key: {cache_key}")
    return cache_key

def create_serverside_output(data: Any, cache_key: str) -> ServersideOutputTransform:
    """
    Create a ServersideOutputTransform for caching data.
    
    Args:
        data: Data to cache
        cache_key: Cache key for the data
        
    Returns:
        ServersideOutputTransform instance
    """
    logger.info(f"Creating serverside output with key: {cache_key}")
    return ServersideOutputTransform(
        data,
        id=cache_key,
        backend='memory',  # Use in-memory backend as specified
    )

def extract_from_serverside(serverside_output: ServersideOutputTransform) -> Any:
    """
    Extract data from a ServersideOutputTransform.
    
    Args:
        serverside_output: ServersideOutputTransform instance
        
    Returns:
        Cached data
    """
    try:
        data = serverside_output.data
        logger.info("Successfully extracted data from serverside cache")
        return data
    except Exception as e:
        logger.error(f"Failed to extract serverside data: {e}")
        raise

class CacheManager:
    """
    Manager for handling caching operations in the dashboard.
    """
    
    def __init__(self):
        self._cache_keys: Dict[str, str] = {}
        self._session_id = get_session_id()
    
    def cache_activation_data(
        self,
        model_name: str,
        prompt: str,
        attention_pattern: str,
        mlp_pattern: str,
        norm_param_name: str,
        logit_lens_param_name: str,
        activation_data: Dict[str, Any]
    ) -> Tuple[str, ServersideOutputTransform]:
        """
        Cache activation data with proper key generation.
        
        Returns:
            Tuple of (cache_key, serverside_output)
        """
        params = {
            'model_name': model_name,
            'prompt': prompt,
            'attention_pattern': attention_pattern,
            'mlp_pattern': mlp_pattern,
            'norm_param_name': norm_param_name,
            'logit_lens_param_name': logit_lens_param_name
        }
        
        cache_key = compose_cache_key(params)
        serverside_output = create_serverside_output(activation_data, cache_key)
        
        # Track cache key for this session
        self._cache_keys[cache_key] = cache_key
        
        logger.info(f"Cached activation data with key: {cache_key}")
        return cache_key, serverside_output
    
    def cache_bertviz_data(
        self,
        base_cache_key: str,
        bertviz_type: str,
        layer_idx: Optional[int],
        html_content: str
    ) -> Tuple[str, ServersideOutputTransform]:
        """
        Cache BertViz HTML content.
        
        Args:
            base_cache_key: Base cache key from activation data
            bertviz_type: Type of visualization ('thumbnail', 'head_view', 'model_view')
            layer_idx: Layer index (for layer-specific views)
            html_content: HTML content to cache
            
        Returns:
            Tuple of (cache_key, serverside_output)
        """
        # Create specific cache key for BertViz data
        if layer_idx is not None:
            viz_key = f"{base_cache_key}_bertviz_{bertviz_type}_layer_{layer_idx}"
        else:
            viz_key = f"{base_cache_key}_bertviz_{bertviz_type}"
        
        serverside_output = create_serverside_output(html_content, viz_key)
        self._cache_keys[viz_key] = viz_key
        
        logger.info(f"Cached BertViz {bertviz_type} with key: {viz_key}")
        return viz_key, serverside_output
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about current cache state.
        
        Returns:
            Dictionary with cache information
        """
        return {
            'session_id': self._session_id,
            'cached_keys': list(self._cache_keys.keys()),
            'cache_count': len(self._cache_keys)
        }
    
    def clear_session_cache(self) -> None:
        """
        Clear cache for current session.
        """
        self._cache_keys.clear()
        # Generate new session ID
        self._session_id = str(uuid.uuid4())
        _session_registry['current_session'] = self._session_id
        logger.info(f"Cleared session cache, new session: {self._session_id}")

# Global cache manager instance
cache_manager = CacheManager()

def get_cache_manager() -> CacheManager:
    """
    Get the global cache manager instance.
    
    Returns:
        CacheManager instance
    """
    return cache_manager

def create_cache_key_for_visualization(
    model_name: str,
    prompt: str,
    selections: Dict[str, str]
) -> str:
    """
    Convenience function to create cache key for visualization data.
    
    Args:
        model_name: Model name
        prompt: Input prompt
        selections: Dictionary with user selections
        
    Returns:
        Cache key string
    """
    params = {
        'model_name': model_name,
        'prompt': prompt,
        'attention_pattern': selections.get('attention_pattern', ''),
        'mlp_pattern': selections.get('mlp_pattern', ''),
        'norm_param_name': selections.get('norm_param_name', ''),
        'logit_lens_param_name': selections.get('logit_lens_param_name', '')
    }
    
    return compose_cache_key(params)
