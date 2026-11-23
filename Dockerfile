FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY cattle_breed_project/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts, annotations, and backend source
COPY cattle_breed_project/models ./cattle_breed_project/models
COPY cattle_breed_project/annotations ./cattle_breed_project/annotations
COPY cattle_breed_project/backend ./cattle_breed_project/backend

WORKDIR /app/cattle_breed_project/backend

ENV PORT=5000
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
