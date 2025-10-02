"""
Package database pour l'intégration PostgreSQL avec InspectIA
"""

# Import des modules principaux
from .database import get_db, get_db_session, init_database, test_connection
from .backend_integration import InspectIADatabase, get_inspectia_db
from .models import (
    Declaration, Prediction, Feature, Chapter, Model,
    RLDecision, FeedbackHistory, AnalysisResult
)

__all__ = [
    'get_db',
    'get_db_session', 
    'init_database',
    'test_connection',
    'InspectIADatabase',
    'get_inspectia_db',
    'Declaration',
    'Prediction',
    'Feature',
    'Chapter',
    'Model',
    'RLDecision',
    'FeedbackHistory',
    'AnalysisResult'
]










