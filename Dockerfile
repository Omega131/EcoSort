# ==============================================================================
# EcoSort — Dockerfile for Hugging Face Spaces & Cloud Containers
# ==============================================================================
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY starter_kit/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application files and model
COPY model.onnx /app/model.onnx
COPY web_app/ /app/web_app/
COPY run_app.py /app/run_app.py

# Create a non-root user (best practice for Hugging Face Spaces)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

USER user

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "run_app.py", "--host", "0.0.0.0", "--port", "7860"]
