# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Dépendances système nécessaires pour ML et OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    tesseract-ocr \
    tesseract-ocr-fra \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers de requirements d'abord pour optimiser le cache Docker
COPY requirements*.txt ./

# Installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    (pip install -r requirements-dev.txt || true) && \
    (pip install -r requirements-ml.txt || true) && \
    (pip install -r requirements-ocr.txt || true)

# Copier le code source
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p logs results data models

# Exposer le port
EXPOSE 8000

# Commande de démarrage corrigée
CMD ["python", "-m", "api.main"]

