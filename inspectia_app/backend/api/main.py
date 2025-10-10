from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Ajouter le chemin du backend pour les imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import des routers
from api.routes_predict import router as predict_router, postgresql_router, ml_router

app = FastAPI(
    title="InspectIA API",
    description="API pour l'analyse intelligente des déclarations douanières avec système ML-RL hybride",
    version="2.0.0"
)

# Configuration CORS pour permettre les requêtes depuis Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routers
app.include_router(predict_router)  # Routes principales /predict
app.include_router(postgresql_router)  # Routes PostgreSQL /api/v2
app.include_router(ml_router)  # Routes ML Dashboard /predict/ml

@app.get("/")
async def root():
    return {
        "message": "InspectIA API - Système d'analyse douanière intelligent",
        "version": "2.0.0",
        "features": [
            "Système ML-RL hybride avancé",
            "Traitement OCR avancé avec agrégation",
            "Analyse de fraude en temps réel avec 6 techniques",
            "Support multi-chapitres (30, 84, 85) avec modèles optimisés",
            "Base de données PostgreSQL intégrée",
            "API PostgreSQL native",
            "Détection de fraude avancée (Bienaymé-Tchebychev, TEI, Clustering)",
            "Features business spécifiques par chapitre",
            "Seuils optimaux calculés scientifiquement",
            "Pipeline ML avec convention train/val/test",
            "Système RL avec bandits multi-bras",
            "Analyse SHAP pour interprétabilité"
        ],
        "endpoints": {
            "predict": "/predict (Routes principales)",
            "api": "/api/v2 (PostgreSQL)",
            "ml_dashboard": "/predict/ml"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "InspectIA API",
        "version": "2.0.0",
        "system": "ML-RL Hybrid"
    }

@app.get("/chapters")
async def get_available_chapters():
    """Retourne les chapitres disponibles avec leurs informations mises à jour"""
    return {
        "chapters": [
            {
                "id": "chap30",
                "name": "Produits pharmaceutiques",
                "description": "Médicaments et produits pharmaceutiques",
                "best_model": "XGBoost",
                "model_type": "ML Avancé",
                "performance": {
                    "validation_f1": 0.9815,
                    "f1_score": 0.9796,
                    "auc_score": 0.9995,
                    "precision": 0.9889,
                    "recall": 0.9705
                },
                "features_count": 41,
                "data_size": 25334,
                "fraud_rate": 19.44,
                "decision_thresholds": {
                    "conforme": 0.15,
                    "fraude": 0.45,
                    "optimal_threshold": 0.35
                },
                "system_status": "operational",
                "ml_rl_integration": True,
                "ocr_pipeline": True,
                "advanced_fraud_detection": True
            },
            {
                "id": "chap84", 
                "name": "Machines et appareils mécaniques",
                "description": "Machines et appareils mécaniques",
                "best_model": "XGBoost",
                "model_type": "ML Avancé",
                "performance": {
                    "validation_f1": 0.9891,
                    "f1_score": 0.9887,
                    "auc_score": 0.9997,
                    "precision": 0.9942,
                    "recall": 0.9833
                },
                "features_count": 43,
                "data_size": 264494,
                "fraud_rate": 26.80,
                "decision_thresholds": {
                    "conforme": 0.15,
                    "fraude": 0.25,
                    "optimal_threshold": 0.20
                },
                "system_status": "operational",
                "ml_rl_integration": True,
                "ocr_pipeline": True,
                "advanced_fraud_detection": True
            },
            {
                "id": "chap85",
                "name": "Appareils électriques", 
                "description": "Machines et appareils électriques",
                "best_model": "XGBoost",
                "model_type": "ML Avancé",
                "performance": {
                    "validation_f1": 0.9808,
                    "f1_score": 0.9808,
                    "auc_score": 0.9993,
                    "precision": 0.9894,
                    "recall": 0.9723
                },
                "features_count": 43,
                "data_size": 197402,
                "fraud_rate": 21.32,
                "decision_thresholds": {
                    "conforme": 0.15,
                    "fraude": 0.25,
                    "optimal_threshold": 0.20
                },
                "system_status": "operational",
                "ml_rl_integration": True,
                "ocr_pipeline": True,
                "advanced_fraud_detection": True
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)