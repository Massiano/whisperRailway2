# Use Python 3.11 slim image
FROM python:3.11-slim

# Install system deps for librosa/soundfile
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working dir
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install deps (CPU-only)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port
EXPOSE $PORT

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
