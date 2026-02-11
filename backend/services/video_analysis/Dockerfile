# Dockerfile for FitBalance Biomechanics Backend
# ================================================
# Optimized for Google Cloud Run deployment

# Use Python 3.11 slim image for smaller footprint
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Disable GPU for Cloud Run (CPU-only)
    CUDA_VISIBLE_DEVICES="" \
    # Ultralytics settings
    YOLO_VERBOSE=False

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Download YOLO pose model during build (faster cold starts)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

# Copy application code
COPY app/ ./app/
COPY configs/ ./configs/

# Create temp directory for video uploads
RUN mkdir -p /tmp/uploads

# Expose port (Cloud Run uses PORT env variable)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application with Uvicorn
# Cloud Run sets PORT env variable, default to 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
