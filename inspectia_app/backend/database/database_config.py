"""
Configuration de base de données PostgreSQL pour InspectIA
Lit les configurations depuis les variables d'environnement
"""

import os
from typing import Optional

# Configuration de base de données PostgreSQL pour InspectIA
# Les valeurs par défaut sont utilisées si les variables d'environnement ne sont pas définies
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", 
    "postgresql://maramata:maramata@localhost:5432/INSPECT_IA"
)

# Configuration pour SQLAlchemy
SQLALCHEMY_DATABASE_URL: str = os.getenv(
    "SQLALCHEMY_DATABASE_URL", 
    DATABASE_URL
)

# Configuration pour Alembic (migrations)
ALEMBIC_DATABASE_URL: str = os.getenv(
    "ALEMBIC_DATABASE_URL", 
    DATABASE_URL
)

# Pool de connexions
DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))

# Configuration de logging
DB_ECHO: bool = os.getenv("DB_ECHO", "false").lower() == "true"

# Configuration de l'application
APP_NAME: str = os.getenv("APP_NAME", "InspectIA")
APP_VERSION: str = os.getenv("APP_VERSION", "2.0.0")
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

# Configuration des chapitres
CHAPTERS: list = os.getenv("CHAPTERS", "chap30,chap84,chap85").split(",")

# Configuration des chemins
ML_MODELS_PATH: str = os.getenv("ML_MODELS_PATH", "backend/models")
ML_RESULTS_PATH: str = os.getenv("ML_RESULTS_PATH", "backend/results")
OCR_INPUT_PATH: str = os.getenv("OCR_INPUT_PATH", "backend/data/ocr/input")
OCR_PROCESSED_PATH: str = os.getenv("OCR_PROCESSED_PATH", "backend/data/ocr/processed")
RL_DATA_PATH: str = os.getenv("RL_DATA_PATH", "backend/data/rl")

# Configuration des seuils
DEFAULT_FRAUD_THRESHOLD: float = float(os.getenv("DEFAULT_FRAUD_THRESHOLD", "0.5"))
DEFAULT_CONFORME_THRESHOLD: float = float(os.getenv("DEFAULT_CONFORME_THRESHOLD", "0.2"))
DEFAULT_FRAUDE_THRESHOLD: float = float(os.getenv("DEFAULT_FRAUDE_THRESHOLD", "0.8"))

# Configuration des logs
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = os.getenv("LOG_FILE", "backend/logs/inspectia.log")

# Configuration API
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# Configuration de sécurité
SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-jwt-secret-key-here")

def get_database_config() -> dict:
    """Retourne la configuration de base de données"""
    return {
        "host": "localhost",
        "port": 5432,
        "user": "maramata",
        "password": "maramata",
        "database": "INSPECT_IA",
        "database_url": DATABASE_URL,
        "sqlalchemy_database_url": SQLALCHEMY_DATABASE_URL,
        "alembic_database_url": ALEMBIC_DATABASE_URL,
        "pool_size": DB_POOL_SIZE,
        "max_overflow": DB_MAX_OVERFLOW,
        "pool_timeout": DB_POOL_TIMEOUT,
        "pool_recycle": DB_POOL_RECYCLE,
        "echo": DB_ECHO
    }

def get_app_config() -> dict:
    """Retourne la configuration de l'application"""
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "debug": DEBUG,
        "environment": ENVIRONMENT,
        "chapters": CHAPTERS,
        "log_level": LOG_LEVEL,
        "log_file": LOG_FILE,
        "api_host": API_HOST,
        "api_port": API_PORT
    }
