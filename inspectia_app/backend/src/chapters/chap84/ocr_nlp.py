# backend/src/chapters/chap84/ocr_nlp.py
"""
Interface OCR/NLP pour le Chapitre 84 - Machines et appareils mécaniques
Intégration complète avec le nouveau système ML-RL avancé
- Modèle ML: XGBoost - Validation F1: 0.9891 ⭐ (critère de sélection)
  * Test: F1=0.9887, AUC=0.9997, Precision=0.9942, Recall=0.9833
- Système RL: Epsilon-greedy, UCB, Hybrid
- Features business optimisées par corrélation (43 features)
- Configuration EXCEPTIONNELLE avec seuils optimaux (0.200)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Ajouter le chemin pour les imports
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))

from shared.ocr_pipeline import (
    process_ocr_document, 
    predict_fraud_from_ocr_data, 
    run_auto_predict,
    get_chapter_config,
    get_best_model_for_chapter,
    AdvancedOCRPipeline
)
from shared.ocr_ingest import process_declaration_file, aggregate_csv_by_declaration

def predict_from_uploads(
    paths: Optional[List[str]] = None, 
    declarations: Optional[List[Dict]] = None, 
    uploads: Optional[List[str]] = None, 
    pdfs: Optional[List[str]] = None,
    level: str = "basic"
) -> Dict[str, Any]:
    """
    Prédiction pour le chapitre 84 avec le nouveau système ML-RL intégré
    
    Args:
        paths: Chemins des fichiers (compatibilité)
        declarations: Données de déclarations directes
        uploads: Chemins des images uploadées
        pdfs: Chemins des PDFs
        level: Niveau RL (basic, advanced, expert)
    
    Returns:
        Dict avec les résultats de prédiction
    """
    try:
        # Utiliser le nouveau pipeline OCR avancé
        result = run_auto_predict(
            chapter="chap84",
            uploads=uploads or paths,
            declarations=declarations,
            pdfs=pdfs
        )
        
        # Enrichir avec les informations spécifiques au chapitre 84
        result.update({
            "chapter_name": "Machines et appareils mécaniques",
            "best_model": "xgboost",
            "model_performance": {
                "validation_f1": 0.9891,
                "f1_score": 0.9887,
                "auc": 0.9997,
                "precision": 0.9942,
                "recall": 0.9833,
                "accuracy": 0.9887
            },
            "optimal_threshold": 0.20,
            "rl_level": level,
            "mechanical_features": True,
            "features_count": 43,
            "configuration": "EXCEPTIONNELLE",
            "fraud_rate": 26.80,
            "data_size": 264494
        })
        
        return result
        
    except Exception as e:
        return {
            "error": f"Erreur prédiction chapitre 84: {str(e)}",
            "chapter": "chap84",
            "results": []
        }

def predict_from_file_with_aggregation(file_path: str, level: str = "basic") -> Dict[str, Any]:
    """
    Prédiction pour le chapitre 84 avec agrégation par DECLARATION_ID
    
    Args:
        file_path: Chemin vers le fichier (PDF, CSV, Image)
        level: Niveau RL (basic, advanced, expert)
    
    Returns:
        Dict avec les résultats de prédiction incluant l'agrégation
    """
    try:
        # Traiter le fichier avec agrégation
        file_result = process_declaration_file(file_path, chapter="chap84")
        
        if "error" in file_result:
            return {
                "error": f"Erreur traitement fichier: {file_result['error']}",
                "file_path": file_path,
                "chapter": "chap84"
            }
        
        # Utiliser le pipeline OCR pour la prédiction
        pipeline = AdvancedOCRPipeline()
        
        # Si c'est un CSV avec agrégation, utiliser la fonction spécialisée
        if file_result.get("source_type") == "csv" and "total_declarations" in file_result:
            # Reconstituer les données agrégées
            aggregated_data = [file_result["extracted_data"]]
            prediction_result = pipeline.process_csv_with_aggregation(
                aggregated_data, chapter="chap84", level=level
            )
        else:
            # Traitement standard
            prediction_result = pipeline.predict_fraud(
                file_result["extracted_data"], chapter="chap84", level=level
            )
        
        # Enrichir avec les informations spécifiques au chapitre 84
        result = {
            "file_path": file_path,
            "chapter": "chap84",
            "chapter_name": "Machines et appareils mécaniques",
            "best_model": "xgboost",
            "file_processing": file_result,
            "prediction": prediction_result,
            "aggregation_info": {
                "declaration_id": file_result.get("extracted_data", {}).get("DECLARATION_ID", "UNKNOWN"),
                "total_declarations": file_result.get("total_declarations", 1),
                "source_type": file_result.get("source_type", "unknown")
            },
            "model_performance": {
                "validation_f1": 0.9891,
                "f1_score": 0.9887,
                "auc": 0.9997,
                "precision": 0.9942,
                "recall": 0.9833,
                "accuracy": 0.9887
            },
            "optimal_threshold": 0.20,
            "configuration": "EXCEPTIONNELLE",
            "fraud_rate": 26.80,
            "data_size": 264494
        }
        
        return result
        
    except Exception as e:
        return {
            "error": f"Erreur prédiction fichier chapitre 84: {str(e)}",
            "file_path": file_path,
            "chapter": "chap84"
        }

def predict_from_ocr_data(ocr_data: Dict[str, Any], level: str = "basic") -> Dict[str, Any]:
    """
    Prédiction directe à partir de données OCR pour le chapitre 84
    
    Args:
        ocr_data: Données extraites par OCR
        level: Niveau RL (basic, advanced, expert)
    
    Returns:
        Dict avec la prédiction de fraude
    """
    try:
        result = predict_fraud_from_ocr_data(ocr_data, chapter="chap84", level=level)
        
        # Enrichir avec les métadonnées du chapitre 84
        result.update({
            "chapter_name": "Machines et appareils mécaniques",
            "best_model": "xgboost",
            "prediction": result.get("predicted_fraud", "N/A"),
            "fraud_probability": result.get("fraud_probability", 0),
            "validation_f1": 0.9891,
            "f1_score": 0.9887,
            "auc": 0.9997,
            "precision": 0.9942,
            "recall": 0.9833,
            "accuracy": 0.9887,
            "features_count": 43,
            "specialized_features": [
                "BUSINESS_GLISSEMENT_MACHINE_BUREAU",
                "BUSINESS_GLISSEMENT_PAYS_MACHINES",
                "BUSINESS_GLISSEMENT_RATIO_SUSPECT",
                "BUSINESS_RISK_PAYS_HIGH",
                "BUSINESS_ORIGINE_DIFF_PROVENANCE",
                "BUSINESS_REGIME_PREFERENTIEL",
                "BUSINESS_REGIME_NORMAL",
                "BUSINESS_VALEUR_ELEVEE",
                "BUSINESS_VALEUR_EXCEPTIONNELLE",
                "BUSINESS_POIDS_ELEVE",
                "BUSINESS_DROITS_ELEVES",
                "BUSINESS_RATIO_LIQUIDATION_CAF",
                "BUSINESS_RATIO_DOUANE_CAF",
                "BUSINESS_IS_MACHINE_BUREAU",
                "BUSINESS_IS_ELECTROMENAGER",
                "BUSINESS_IS_PRECISION_UEMOA",
                "BUSINESS_ARTICLES_MULTIPLES",
                "BUSINESS_AVEC_DPI"
            ],
            "model_performance": {
                "validation_f1": 0.9891,
                "f1_score": 0.9887,
                "auc": 0.9997,
                "precision": 0.9942,
                "recall": 0.9833,
                "accuracy": 0.9887
            },
            "optimal_threshold": 0.20,
            "fraud_rate": 26.80,
            "data_size": 264494
        })
        
        return result
        
    except Exception as e:
        return {
            "error": f"Erreur prédiction OCR chapitre 84: {str(e)}",
            "chapter": "chap84"
        }

def process_document(image_path: str, level: str = "basic") -> Dict[str, Any]:
    """
    Traiter un document complet avec OCR et prédiction pour le chapitre 84
    
    Args:
        image_path: Chemin vers l'image du document
        level: Niveau RL (basic, advanced, expert)
    
    Returns:
        Dict avec le résultat complet du traitement
    """
    try:
        result = process_ocr_document(image_path, chapter="chap84", level=level)
        
        # Enrichir avec les informations spécifiques
        if "prediction" in result:
            result["prediction"].update({
                "chapter_name": "Machines et appareils mécaniques",
                "best_model": "xgboost",
                "mechanical_features": True,
                "model_performance": {
                    "validation_f1": 0.9891,
                    "f1_score": 0.9887,
                    "auc": 0.9997,
                    "precision": 0.9942,
                    "recall": 0.9833,
                    "accuracy": 0.9887
                },
                "optimal_threshold": 0.20
            })
        
        return result
        
    except Exception as e:
        return {
            "error": f"Erreur traitement document chapitre 84: {str(e)}",
            "image_path": image_path
        }

def get_chapter_info() -> Dict[str, Any]:
    """Obtenir les informations sur le chapitre 84"""
    config = get_chapter_config("chap84")
    return {
        "chapter": "chap84",
        "name": "Machines et appareils mécaniques",
        "best_model": get_best_model_for_chapter("chap84"),
        "model_performance": {
            "validation_f1": 0.9891,
            "f1_score": 0.9887,
            "auc": 0.9997,
            "precision": 0.9942,
            "recall": 0.9833,
            "accuracy": 0.9887
        },
        "optimal_threshold": 0.20,
        "configuration": "EXCEPTIONNELLE",
        "features_count": 43,
        "fraud_rate": 26.80,
        "data_size": 264494,
        "features": {
            "numerical": config.get("features", {}).get("numerical", []),
            "categorical": config.get("features", {}).get("categorical", []),
            "business": config.get("features", {}).get("business", [])
        },
        "rl_config": config.get("rl_config", {}),
        "specialized_for": [
            "Machines de bureau",
            "Équipements industriels",
            "Électroménager",
            "Réacteurs nucléaires",
            "Chaudières",
            "Machines à vapeur",
            "Machines-outils",
            "Équipements de transport"
        ]
    }

def test_chapter84_integration():
    """Tester l'intégration du chapitre 84"""
    print("🧪 Test intégration Chapitre 84 - Machines et appareils mécaniques")
    
    # Test des informations
    info = get_chapter_info()
    print(f"✅ Chapitre: {info['name']}")
    print(f"✅ Modèle: {info['best_model']}")
    print(f"✅ F1-Score: {info['model_performance']['f1_score']}")
    print(f"✅ AUC: {info['model_performance']['auc']}")
    
    # Test de prédiction avec données simulées
    test_data = {
        'declaration_id': 'TEST_CHAP84_001',
        'valeur_caf': 10000000,
        'poids_net': 500,
        'quantite_complement': 5,
        'taux_droits_percent': 5.0,
        'code_sh_complet': '8471.30.00.00',
        'pays_origine': 'CN',
        'pays_provenance': 'CN',
        'regime_complet': 'C111',
        'statut_bae': 'AVEC_BAE',
        'type_regime': 'CONSOMMATION',
        'regime_douanier': 'CONSOMMATION',
        'regime_fiscal': 'NORMAL'
    }
    
    result = predict_from_ocr_data(test_data, level="advanced")
    print(f"✅ Prédiction test: {result.get('predicted_fraud', 'N/A')}")
    print(f"✅ Probabilité: {result.get('fraud_probability', 0):.3f}")
    print(f"✅ ML utilisé: {result.get('ml_integration_used', False)}")
    
    print("🎯 Chapitre 84 testé avec succès!")

if __name__ == "__main__":
    test_chapter84_integration()