# Hugging Face Spaces Dockerfile for Transformer Explanation Dashboard
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyTorch, transformers, and Playwright
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    # Playwright/Chromium deps
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright + Chromium (for headless testing)
RUN pip install --no-cache-dir playwright && playwright install chromium

# Copy application code
COPY . .

# Expose the port HF Spaces expects
EXPOSE 7860

# Run the Dash app
CMD ["python", "app.py"]
