# Use an NVIDIA CUDA base image to support Unsloth and GPU acceleration
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set up python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install deps
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# HF Spaces uses port 7860
EXPOSE 7860

# Start server - Important: point to prevaluation_env
CMD ["uvicorn", "prevaluation_env.server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
