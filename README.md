# Transformer Explanation Dashboard

A Dash-based interactive application for visualizing and analyzing the internal mechanics of Transformer-based Large Language Models (LLMs). This tool enables users to inspect the generation pipeline step-by-step and perform real-time experiments like ablation and attribution.

## Architecture

The project is structured around a central Dash application with modular components and utility libraries:

### Core Components
*   `app.py`: The main application entry point that orchestrates the layout and callbacks.
*   `components/`: Modular UI elements.
    *   `pipeline.py`: Implements the 5-stage visualization pipeline (Tokenization, Embedding, Attention, MLP, Output).
    *   `investigation_panel.py`: Handles the experimental interfaces (Ablation and Attribution).
    *   `ablation_panel.py`: Logic for the head ablation interface.
    *   `tokenization_panel.py`: Visualization for token processing.
    *   `sidebar.py` & `model_selector.py`: Configuration and navigation controls.

### Utilities (`utils/`)
*   `model_patterns.py`: Core logic for hooking into PyTorch models to capture activations.
*   `model_config.py`: Registry for automatic detection of model families (LLaMA, GPT-2, OPT, etc.).
*   `head_detection.py`: Analysis logic for categorizing attention heads.
*   `beam_search.py`: Implementation of beam search for sequence generation analysis.
*   `token_attribution.py`: Integrated Gradients implementation for feature importance.

## Installation

### Prerequisites
*   Python 3.11+
*   PyTorch

### Steps

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Launch the Dashboard**:
    ```bash
    python app.py
    ```

2.  **Access the Interface**:
    Open a web browser and navigate to `http://127.0.0.1:8050/`.

3.  **Workflow**:
    *   **Model Selection**: Choose a model from the dropdown or enter a HuggingFace model ID. The system automatically detects the architecture.
    *   **Analysis**: Enter a prompt and click "Analyze" to visualize the forward pass.
    *   **Pipeline Exploration**: Interact with the 5 pipeline stages to view detailed activation data.
    *   **Experiments**: Use the Investigation Panel at the bottom to run Ablation (disable heads) or Attribution (analyze token importance) experiments.

## Testing

The project includes a comprehensive test suite located in the `tests/` directory. To run the tests:

```bash
pytest tests/
```

## Documentation

Additional project documentation is available in the `conductor/` directory:
*   [Product Definition](conductor/product.md)
*   [Tech Stack](conductor/tech-stack.md)
*   [Workflow](conductor/workflow.md)
