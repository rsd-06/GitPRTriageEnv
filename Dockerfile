FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install deps first (layer caching — faster rebuilds)
COPY prevaluation_env/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

WORKDIR /app

# HF Spaces uses port 7860
EXPOSE 7860

# Start server from top-level to resolve relative imports (..models)
CMD ["uvicorn", "prevaluation_env.server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
