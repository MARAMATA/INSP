"""
Module principal pour InspectIA
Système d'analyse automatisée des documents douaniers
"""

__version__ = "2.0.0"
__author__ = "InspectIA Team"

# Configuration par défaut
DEFAULT_CONFIG_PATH = "configs/base.yaml"
DEFAULT_DATA_DIR = "data"
DEFAULT_MODELS_DIR = "models"

# Imports des modules principaux
from .shared import (
    run_auto_predict,
    process_ocr_document,
    predict_fraud_from_ocr_data,
    get_chapter_config,
    list_available_chapters,
    get_best_model_for_chapter
)

# Imports des nouvelles fonctions d'agrégation
from .shared.ocr_pipeline import AdvancedOCRPipeline
from .shared.ocr_ingest import process_declaration_file, aggregate_csv_by_declaration

__all__ = [
    # Configuration
    'DEFAULT_CONFIG_PATH',
    'DEFAULT_DATA_DIR', 
    'DEFAULT_MODELS_DIR',
    
    # Shared modules
    'run_auto_predict',
    'process_ocr_document',
    'predict_fraud_from_ocr_data',
    'get_chapter_config',
    'list_available_chapters',
    'get_best_model_for_chapter',
    
    # Nouvelles fonctions d'agrégation
    'AdvancedOCRPipeline',
    'process_declaration_file',
    'aggregate_csv_by_declaration'
]