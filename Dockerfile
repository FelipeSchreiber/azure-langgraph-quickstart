FROM python:3.12-slim

# ---------- system deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- working directory ----------
WORKDIR /app

# ---------- Python dependencies ----------
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------- application code ----------
COPY src/ ./src/

# ---------- application code ----------
COPY agent_config.yaml ./

# Make the src package importable
ENV PYTHONPATH=/app

# ---------- runtime ----------
EXPOSE 8000

# Health-check uses the liveness probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
