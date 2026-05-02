# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Group 1 — Lightweight web packages
RUN pip install --no-cache-dir \
    --timeout=300 \
    --retries=10 \
    fastapi==0.111.0 \
    uvicorn==0.30.0 \
    pydantic==2.7.0 \
    python-multipart==0.0.9 \
    python-dotenv==1.0.1

# Group 2 — AWS
RUN pip install --no-cache-dir \
    --timeout=300 \
    --retries=10 \
    boto3==1.34.0

# Group 3 — ML (heaviest — separate layer)
RUN pip install --no-cache-dir \
    --timeout=300 \
    --retries=10 \
    numpy==1.26.4

RUN pip install --no-cache-dir \
    --timeout=300 \
    --retries=10 \
    scikit-learn==1.4.2

RUN pip install --no-cache-dir \
    --timeout=300 \
    --retries=10 \
    joblib==1.4.2 \
    imbalanced-learn==0.12.2

# Copy app files
COPY main.py model.py schema.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]