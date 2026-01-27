# Use lightweight Python base image (CPU only is fine for API mode)
FROM python:3.10-slim

# Set Env
ENV PYTHONUNBUFFERED=1

# Install system dependencies (needed for chromadb/building)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Working Directory
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy App Code
COPY app.py .
COPY start.sh .
COPY data/ data/

# Make start script executable
RUN chmod +x start.sh

# Expose Port
EXPOSE 7860

# Start
CMD ["./start.sh"]
