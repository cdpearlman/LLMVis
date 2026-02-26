# Hugging Face Spaces Dockerfile for Transformer Explanation Dashboard
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyTorch and transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port HF Spaces expects
EXPOSE 7860

# Run the Dash app
CMD ["python", "app.py"]
