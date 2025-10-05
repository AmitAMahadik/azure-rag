# Lightweight base image
FROM python:3.11-slim

# System deps (optional but helpful for some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY . .

# FastAPI will listen on 8080 (match your Azure Container App config later)
EXPOSE 8080

# Start the API (module:function is streaming_chat_rag:app)
CMD ["uvicorn", "streaming_chat_rag:app", "--host", "0.0.0.0", "--port", "8080"]

