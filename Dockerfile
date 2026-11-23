
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY cattle_breed_project/backend/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model, annotations and backend
COPY cattle_breed_project/models ./cattle_breed_project/models
COPY cattle_breed_project/annotations ./cattle_breed_project/annotations
COPY cattle_breed_project/backend ./cattle_breed_project/backend

# Set backend working directory
WORKDIR /app/cattle_breed_project/backend

# Render provides its own PORT environment variable
ENV PORT=10000

EXPOSE 10000

# Start Flask using Gunicorn
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 app:app"]
