# ------------------------------
# Base image
# ------------------------------
FROM python:3.10-slim

# ------------------------------
# System dependencies
# ------------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Set working directory
# ------------------------------
WORKDIR /app

# ------------------------------
# Copy project files
# ------------------------------

COPY requirements.txt ./

# ------------------------------
# Install dependencies
# ------------------------------

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ------------------------------
# Set environment variables
# ------------------------------
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ------------------------------
# Default command
# ------------------------------
CMD ["python", "-m", "emotion-recognition-mlops.src.rnn_pipeline.training.train.py"]
