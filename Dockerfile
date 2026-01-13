# Use a lightweight Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some compiled python libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MODEL_DIR=/app/models

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and models
COPY src/ src/
COPY models/ models/
COPY params.yaml .

# Set environment variables
ENV PYTHONPATH=/app

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.credit_risk_model.api.app:app", "--host", "0.0.0.0", "--port", "8000"]