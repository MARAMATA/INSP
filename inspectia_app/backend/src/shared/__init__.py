"""
Module shared pour InspectIA
Modules partagés entre les différents chapitres
"""

# Imports des modules disponibles
from .ocr_pipeline import (
    run_auto_predict,
    process_ocr_document,
    predict_fraud_from_ocr_data,
    get_chapter_config,
    list_available_chapters,
    get_best_model_for_chapter
)

__all__ = [
    'run_auto_predict',
    'process_ocr_document',
    'predict_fraud_from_ocr_data',
    'get_chapter_config',
    'list_available_chapters',
    'get_best_model_for_chapter'
]


