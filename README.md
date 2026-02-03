# Transformer Explanation Dashboard

A comprehensive, interactive tool for capturing, visualizing, and experimenting with Transformer-based Large Language Models (LLMs). This project demystifies the inner workings of models by transforming abstract architectural concepts into tangible, observable phenomena.

## Vision

To foster a deep, intuitive understanding of how powerful models process information by combining interactive visualizations with hands-on experimentation capabilities.

## Key Features

### 🔍 Interactive Pipeline Visualization
Follow the data flow step-by-step through the model's architecture:
1.  **Tokenization**: See how text is split and assigned IDs.
2.  **Embedding**: Visualize the look-up of semantic vectors.
3.  **Attention**: Explore head-level attention patterns using **BertViz**.
4.  **MLP (Feed-Forward)**: Understand where factual knowledge is stored.
5.  **Output Selection**: View probability distributions and top predictions.

### 🧪 Experiments & Investigation
Go beyond static observation with interactive experiments:
*   **Ablation Studies**: Selectively disable specific attention heads across different layers to observe their impact on generation and probability.
*   **Token Attribution**: Use **Integrated Gradients** to see which input tokens contributed most to a specific prediction.
*   **Beam Search Analysis**: Visualize how multiple generation choices are explored.

### 🤖 Broad Model Support
The dashboard features **Automatic Model Family Detection**, supporting a wide range of architectures without manual configuration:
*   **LLaMA-like**: LLaMA 2/3, Mistral, Mixtral, Qwen2/2.5
*   **GPT-2**: GPT-2 (Small/Medium/Large/XL)
*   **OPT**: Facebook OPT models
*   **GPT-NeoX**: Pythia, GPT-NeoX
*   **BLOOM**: BigScience BLOOM
*   **Falcon**: TII Falcon
*   **MPT**: MosaicML MPT

## Getting Started

### Prerequisites
*   Python 3.11+ recommended
*   PyTorch

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/transformer-dashboard.git
    cd transformer-dashboard
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Dashboard

Launch the application:

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:8050/`.

## Usage Guide

1.  **Select a Model**: Choose from the predefined list or enter a HuggingFace model ID. The system will auto-detect the architecture.
2.  **Enter a Prompt**: Type a sentence to analyze.
3.  **Configure Generation**: Adjust "Number of New Tokens" and "Number of Generation Choices" (Beam Width).
4.  **Run Analysis**: Click "Analyze" to run the forward pass.
5.  **Explore the Pipeline**: Click on the pipeline stages (Tokenization, Attention, etc.) to expand detailed views.
6.  **Run Experiments**:
    *   Use the **Investigation Panel** at the bottom to switch between Ablation and Attribution tabs.
    *   In **Ablation**, select layers and heads to disable, then click "Run Ablation Experiment".
    *   In **Attribution**, select a target token and method to visualize feature importance.

## Project Structure

*   `app.py`: Main application entry point and layout orchestration.
*   `components/`: Modular UI components.
    *   `pipeline.py`: The core 5-stage visualization.
    *   `investigation_panel.py`: Ablation and attribution interfaces.
    *   `ablation_panel.py`: Specific logic for head ablation UI.
    *   `tokenization_panel.py`: Token visualization.
*   `utils/`: Backend logic and helper functions.
    *   `model_patterns.py`: Activation capture and hooking logic.
    *   `model_config.py`: Model family definitions and auto-detection.
    *   `head_detection.py`: Attention head categorization logic.
    *   `beam_search.py`: Beam search implementation.
*   `tests/`: Comprehensive test suite ensuring stability.
*   `conductor/`: Detailed project documentation and product guidelines.

## Documentation

For more detailed information on the project's background and technical details, check the `conductor/` directory:
*   [Product Definition](conductor/product.md)
*   [Tech Stack](conductor/tech-stack.md)
*   [Workflow](conductor/workflow.md)

## Contributing

Contributions are welcome! Please ensure that any new features include appropriate tests in the `tests/` directory. Run the test suite before submitting:

```bash
pytest tests/
```
