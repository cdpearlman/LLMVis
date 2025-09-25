"""
Constants for the visualization dashboard application.
"""

# Model configuration
SUPPORTED_MODELS = ["gpt2", "Qwen/Qwen2.5-0.5B"]
DEVICE = "cpu"  # CPU-only execution as specified

# Visualization parameters
TOP_K = 3  # Number of top tokens to extract per layer
MIN_EDGE_OPACITY = 0.15  # Minimum visibility for probability edges
CURVE_FACTOR = 0.5  # Bezier curve strength for edges

# Layout configuration
TILE_WIDTH = 300
TILE_HEIGHT = 200
TILE_SIZE = (TILE_WIDTH, TILE_HEIGHT)
MAX_ROWS = 4  # Maximum rows for layer grid
MAX_PROMPT_WORDS = 50  # Word limit for prompts

# Cache configuration
CACHE_TTL_MINUTES = 60  # Session-scoped, but with fallback TTL
DEFAULT_CACHE_KEY_PREFIX = "viz_dashboard"

# Component IDs for Dash callbacks
COMPONENT_IDS = {
    # Top bar
    "model_dropdown": "model-dropdown",
    "prompt_input": "prompt-input",
    "find_modules_btn": "find-modules-btn",
    "prompt_status": "prompt-status",
    
    # Sidebar
    "attention_pattern_dropdown": "attention-pattern-dropdown",
    "mlp_pattern_dropdown": "mlp-pattern-dropdown",
    "norm_param_dropdown": "norm-param-dropdown",
    "logit_param_dropdown": "logit-param-dropdown",
    "visualize_btn": "visualize-btn",
    "sidebar_status": "sidebar-status",
    
    # Dashboard
    "main_dashboard": "main-dashboard",
    "dashboard_content": "dashboard-content",
    
    # Modal
    "layer_modal": "layer-modal",
    "modal_tabs": "modal-tabs",
    "modal_content": "modal-content",
    
    # Hidden stores for caching
    "cached_data_store": "cached-data-store",
    "session_store": "session-store",
}

# CSS classes
CSS_CLASSES = {
    "top_bar": "top-bar-container",
    "sidebar": "sidebar-container",
    "dashboard": "dashboard-container",
    "layer_tile": "layer-tile",
    "edge_line": "edge-line",
    "modal": "layer-modal",
}

# Error messages
ERROR_MESSAGES = {
    "model_load_failed": "Failed to load model. Please try again.",
    "no_modules_found": "No modules found for this model.",
    "activation_capture_failed": "Failed to capture activations.",
    "invalid_selection": "Please select all required parameters.",
    "prompt_too_long": "Prompt truncated to {max_words} words.",
    "bertviz_render_failed": "Failed to generate attention visualization.",
}

# Success messages
SUCCESS_MESSAGES = {
    "modules_found": "Found {count} module patterns.",
    "visualization_ready": "Visualization generated successfully.",
    "data_cached": "Results cached for session.",
}
