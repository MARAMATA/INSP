#!/usr/bin/env python3
"""
Système de retraining automatique des modèles ML avec les feedbacks
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import joblib
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import threading
import time

# Ajouter le chemin du backend
backend_root = Path(__file__).resolve().parents[2]
sys.path.append(str(backend_root))

from src.chapters.chap30.ml_model_advanced import Chap30MLAdvanced
from src.chapters.chap84.ml_model_advanced import Chap84MLAdvanced
from src.chapters.chap85.ml_model_advanced import Chap85MLAdvanced

logger = logging.getLogger(__name__)

class MLRetrainingSystem:
    """
    Système de retraining automatique des modèles ML basé sur les feedbacks
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.retraining_lock = threading.Lock()
        self.last_retraining = {}
        self.retraining_interval = 1 * 60 * 60  # 1 heure en secondes (pour les tests)
        self.min_feedbacks_for_retraining = 10  # Seuil réduit pour les tests
        self.models_dir = backend_root / "results"
        
        # Configuration des modèles par chapitre
        self.chapter_models = {
            "chap30": Chap30MLAdvanced,
            "chap84": Chap84MLAdvanced,
            "chap85": Chap85MLAdvanced
        }
        
        logger.info("🚀 Système de retraining ML initialisé")
    
    def should_retrain(self, chapter: str) -> bool:
        """Détermine si un modèle doit être retrainé"""
        try:
            # Vérifier l'intervalle de temps
            last_time = self.last_retraining.get(chapter, 0)
            current_time = time.time()
            
            if current_time - last_time < self.retraining_interval:
                logger.info(f"⏰ Retraining trop récent pour {chapter}")
                return False
            
            # Vérifier le nombre de feedbacks
            feedback_count = self._get_feedback_count(chapter)
            if feedback_count < self.min_feedbacks_for_retraining:
                logger.info(f"📊 Pas assez de feedbacks pour {chapter}: {feedback_count}")
                return False
            
            # Vérifier la qualité des feedbacks
            feedback_quality = self._get_feedback_quality(chapter)
            if feedback_quality < 0.3:  # Seuil de qualité très réduit pour les tests
                logger.info(f"📉 Qualité des feedbacks insuffisante pour {chapter}: {feedback_quality}")
                return False
            
            logger.info(f"✅ Conditions de retraining remplies pour {chapter}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification retraining {chapter}: {e}")
            return False
    
    def retrain_model(self, chapter: str) -> Dict[str, Any]:
        """Retrain un modèle ML avec les nouveaux feedbacks"""
        with self.retraining_lock:
            try:
                logger.info(f"🔄 Début du retraining pour {chapter}")
                
                # 1. Récupérer les données d'entraînement existantes
                existing_data = self._load_existing_training_data(chapter)
                if existing_data is None:
                    return {"success": False, "error": "Données d'entraînement non trouvées"}
                
                # 2. Récupérer les nouveaux feedbacks
                new_feedbacks = self._get_new_feedbacks(chapter)
                if not new_feedbacks:
                    return {"success": False, "error": "Aucun nouveau feedback"}
                
                # 3. Préparer les nouvelles données
                new_data = self._prepare_feedback_data(new_feedbacks, chapter)
                if new_data is None:
                    return {"success": False, "error": "Erreur préparation des données"}
                
                # 4. Combiner les données
                combined_data = self._combine_training_data(existing_data, new_data)
                
                # 5. Entraîner le nouveau modèle
                model_class = self.chapter_models[chapter]
                ml_system = model_class()
                
                # Diviser les données
                X = combined_data.drop('FRAUD_FLAG', axis=1)
                y = combined_data['FRAUD_FLAG']
                
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
                )
                
                splits = {
                    'X_train': X_train, 'y_train': y_train,
                    'X_val': X_val, 'y_val': y_val,
                    'X_test': X_test, 'y_test': y_test
                }
                
                # Créer le pipeline de préprocessing
                preprocessor = ml_system.create_preprocessing_pipeline(X_train)
                
                # Entraîner les modèles
                results, trained_models, validation_results = ml_system.train_models(splits, preprocessor)
                
                # Trouver le meilleur modèle
                best_model_name = ml_system.find_best_model(validation_results)
                best_model = trained_models[best_model_name]
                
                # 6. Évaluer le nouveau modèle
                evaluation = self._evaluate_model(best_model, X_test, y_test, chapter)
                
                # 7. Sauvegarder le nouveau modèle si meilleur
                if evaluation["improvement"]:
                    self._save_retrained_model(best_model, best_model_name, chapter, evaluation)
                    self.last_retraining[chapter] = time.time()
                    
                    logger.info(f"✅ Retraining réussi pour {chapter} - Amélioration: {evaluation['improvement_score']:.4f}")
                    
                    return {
                        "success": True,
                        "chapter": chapter,
                        "best_model": best_model_name,
                        "improvement": evaluation["improvement_score"],
                        "new_metrics": evaluation.get("metrics", {}),
                        "feedbacks_used": len(new_feedbacks),
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    logger.info(f"⚠️ Retraining pour {chapter} - Pas d'amélioration")
                    return {
                        "success": False,
                        "chapter": chapter,
                        "reason": "Pas d'amélioration",
                        "evaluation": evaluation
                    }
                
            except Exception as e:
                logger.error(f"❌ Erreur retraining {chapter}: {e}")
                return {"success": False, "error": str(e)}
    
    def _get_feedback_count(self, chapter: str) -> int:
        """Récupère le nombre de feedbacks pour un chapitre"""
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM advanced_feedbacks 
                WHERE chapter_id = %s AND created_at > %s
            """, (chapter, datetime.now() - timedelta(days=7)))
            
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return count
            
        except Exception as e:
            logger.error(f"Erreur récupération feedbacks {chapter}: {e}")
            return 0
    
    def _get_feedback_quality(self, chapter: str) -> float:
        """Calcule la qualité moyenne des feedbacks"""
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT AVG(feedback_quality_score) FROM advanced_feedbacks 
                WHERE chapter_id = %s AND created_at > %s
            """, (chapter, datetime.now() - timedelta(days=7)))
            
            quality = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return float(quality) if quality else 0.0
            
        except Exception as e:
            logger.error(f"Erreur qualité feedbacks {chapter}: {e}")
            return 0.0
    
    def _load_existing_training_data(self, chapter: str) -> Optional[pd.DataFrame]:
        """Charge les données d'entraînement existantes"""
        try:
            data_path = backend_root / "data" / "processed" / f"{chapter.upper()}_PROCESSED_ADVANCED.csv"
            if data_path.exists():
                data = pd.read_csv(data_path)
                logger.info(f"📊 Données existantes chargées pour {chapter}: {len(data)} lignes")
                return data
            else:
                logger.warning(f"⚠️ Fichier de données non trouvé: {data_path}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur chargement données {chapter}: {e}")
            return None
    
    def _get_new_feedbacks(self, chapter: str) -> List[Dict[str, Any]]:
        """Récupère les nouveaux feedbacks depuis la base de données"""
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()
            
            # Récupérer les feedbacks des 7 derniers jours
            cursor.execute("""
                SELECT declaration_id, inspector_decision, inspector_confidence,
                       predicted_fraud, predicted_probability, context_json,
                       feedback_quality_score, inspector_expertise_level
                FROM advanced_feedbacks 
                WHERE chapter_id = %s 
                AND created_at > %s
                AND feedback_quality_score > 0.6
                ORDER BY created_at DESC
            """, (chapter, datetime.now() - timedelta(days=7)))
            
            feedbacks = []
            for row in cursor.fetchall():
                feedbacks.append({
                    "declaration_id": row[0],
                    "inspector_decision": bool(row[1]),
                    "inspector_confidence": float(row[2]),
                    "predicted_fraud": bool(row[3]),
                    "predicted_probability": float(row[4]),
                    "context": json.loads(row[5]) if row[5] else {},
                    "quality_score": float(row[6]),
                    "expertise_level": row[7]
                })
            
            cursor.close()
            conn.close()
            
            logger.info(f"📥 {len(feedbacks)} nouveaux feedbacks récupérés pour {chapter}")
            return feedbacks
            
        except Exception as e:
            logger.error(f"Erreur récupération feedbacks {chapter}: {e}")
            return []
    
    def _prepare_feedback_data(self, feedbacks: List[Dict[str, Any]], chapter: str) -> Optional[pd.DataFrame]:
        """Prépare les données de feedback pour l'entraînement"""
        try:
            if not feedbacks:
                return None
            
            # Convertir les feedbacks en DataFrame
            data_rows = []
            
            for feedback in feedbacks:
                context = feedback["context"]
                
                # Créer une ligne de données basée sur le contexte
                row = {
                    "ANNEE": context.get("ANNEE", 2024),
                    "BUREAU": context.get("BUREAU", "UNKNOWN"),
                    "NUMERO": context.get("NUMERO", "000"),
                    "VALEUR": context.get("VALEUR", 0.0),
                    "POIDS": context.get("POIDS", 0.0),
                    "PAYS_ORIGINE": context.get("PAYS_ORIGINE", "UNKNOWN"),
                    "CODE_SH": context.get("CODE_SH", "00000000"),
                    "FRAUD_FLAG": int(feedback["inspector_decision"]),  # Utiliser la décision de l'inspecteur
                    "feedback_quality": feedback["quality_score"],
                    "inspector_confidence": feedback["inspector_confidence"],
                    "expertise_level": feedback["expertise_level"]
                }
                
                # Ajouter les features business spécifiques au chapitre
                if chapter == "chap30":
                    row.update({
                        "BUSINESS_GLISSEMENT_COSMETIQUE": context.get("BUSINESS_GLISSEMENT_COSMETIQUE", 0),
                        "BUSINESS_IS_MEDICAMENT": context.get("BUSINESS_IS_MEDICAMENT", 0),
                        "BUSINESS_IS_ANTIPALUDEEN": context.get("BUSINESS_IS_ANTIPALUDEEN", 0)
                    })
                elif chapter == "chap84":
                    row.update({
                        "BUSINESS_GLISSEMENT_MACHINE": context.get("BUSINESS_GLISSEMENT_MACHINE", 0),
                        "BUSINESS_IS_MACHINE": context.get("BUSINESS_IS_MACHINE", 0),
                        "BUSINESS_IS_ELECTRONIQUE": context.get("BUSINESS_IS_ELECTRONIQUE", 0)
                    })
                elif chapter == "chap85":
                    row.update({
                        "BUSINESS_GLISSEMENT_ELECTRONIQUE": context.get("BUSINESS_GLISSEMENT_ELECTRONIQUE", 0),
                        "BUSINESS_IS_ELECTRONIQUE": context.get("BUSINESS_IS_ELECTRONIQUE", 0),
                        "BUSINESS_IS_TELEPHONE": context.get("BUSINESS_IS_TELEPHONE", 0)
                    })
                
                data_rows.append(row)
            
            df = pd.DataFrame(data_rows)
            logger.info(f"📊 Données de feedback préparées pour {chapter}: {len(df)} lignes")
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur préparation données feedback {chapter}: {e}")
            return None
    
    def _combine_training_data(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """Combine les données existantes avec les nouvelles données de feedback"""
        try:
            # S'assurer que les colonnes sont compatibles
            common_columns = list(set(existing_data.columns) & set(new_data.columns))
            
            existing_subset = existing_data[common_columns]
            new_subset = new_data[common_columns]
            
            # Combiner les données
            combined = pd.concat([existing_subset, new_subset], ignore_index=True)
            
            # Supprimer les doublons basés sur les colonnes clés
            key_columns = ["ANNEE", "BUREAU", "NUMERO"]
            if all(col in combined.columns for col in key_columns):
                combined = combined.drop_duplicates(subset=key_columns, keep='last')
            
            logger.info(f"📊 Données combinées: {len(existing_data)} + {len(new_data)} = {len(combined)}")
            
            return combined
            
        except Exception as e:
            logger.error(f"Erreur combinaison données: {e}")
            return existing_data
    
    def _evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, chapter: str) -> Dict[str, Any]:
        """Évalue le nouveau modèle et compare avec l'ancien"""
        try:
            # Prédictions du nouveau modèle
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Métriques du nouveau modèle
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            new_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "auc": roc_auc_score(y_test, y_pred_proba)
            }
            
            # Charger les métriques de l'ancien modèle
            old_metrics = self._load_old_model_metrics(chapter)
            
            # Calculer l'amélioration
            improvement_score = 0.0
            if old_metrics:
                for metric in ["f1", "auc", "precision", "recall"]:
                    if metric in new_metrics and metric in old_metrics:
                        improvement_score += (new_metrics[metric] - old_metrics[metric])
                
                improvement_score /= 4  # Moyenne des améliorations
            
            improvement = improvement_score > 0.01  # Seuil d'amélioration de 1%
            
            return {
                "metrics": new_metrics,
                "old_metrics": old_metrics,
                "improvement": improvement,
                "improvement_score": improvement_score
            }
            
        except Exception as e:
            logger.error(f"Erreur évaluation modèle {chapter}: {e}")
            return {
                "metrics": {},
                "old_metrics": {},
                "improvement": False,
                "improvement_score": 0.0
            }
    
    def _load_old_model_metrics(self, chapter: str) -> Optional[Dict[str, float]]:
        """Charge les métriques de l'ancien modèle"""
        try:
            # Utiliser les métriques de la configuration par défaut
            from src.shared.ocr_pipeline import CHAPTER_CONFIGS
            
            config = CHAPTER_CONFIGS.get(chapter, {})
            if config:
                return {
                    "f1": config.get("f1_score", 0.0),
                    "auc": config.get("auc_score", 0.0),
                    "precision": config.get("precision", 0.0),
                    "recall": config.get("recall", 0.0),
                    "accuracy": config.get("accuracy", 0.0)
                }
            
            # Fallback vers le fichier YAML si disponible
            metrics_file = self.models_dir / chapter / "ml_supervised_report.yaml"
            if metrics_file.exists():
                import yaml
                with open(metrics_file, 'r') as f:
                    data = yaml.safe_load(f)
                    return data.get("test_metrics", {})
            
            return None
            
        except Exception as e:
            logger.error(f"Erreur chargement métriques ancien modèle {chapter}: {e}")
            return None
    
    def _save_retrained_model(self, model, model_name: str, chapter: str, evaluation: Dict[str, Any]):
        """Sauvegarde le nouveau modèle retrainé"""
        try:
            # Créer le répertoire si nécessaire
            chapter_dir = self.models_dir / chapter / "models"
            chapter_dir.mkdir(parents=True, exist_ok=True)
            
            # Sauvegarder le modèle
            model_path = chapter_dir / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            
            # Sauvegarder les métriques
            metrics_path = chapter_dir / f"{model_name}_retraining_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({
                    "model_name": model_name,
                    "chapter": chapter,
                    "retraining_timestamp": datetime.now().isoformat(),
                    "metrics": evaluation["metrics"],
                    "improvement_score": evaluation["improvement_score"],
                    "old_metrics": evaluation["old_metrics"]
                }, f, indent=2)
            
            # Mettre à jour le fichier optimal_thresholds.json
            self._update_optimal_thresholds(chapter, model_name, evaluation["metrics"])
            
            logger.info(f"💾 Modèle retrainé sauvegardé: {model_path}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèle retrainé {chapter}: {e}")
    
    def _update_optimal_thresholds(self, chapter: str, model_name: str, metrics: Dict[str, float]):
        """Met à jour les seuils optimaux après retraining"""
        try:
            thresholds_file = self.models_dir / chapter / "optimal_thresholds.json"
            
            # Calculer le nouveau seuil optimal basé sur la précision
            precision = metrics.get("precision", 0.5)
            optimal_threshold = 1.0 - precision  # Seuil inverse de la précision
            
            # Charger les seuils existants
            if thresholds_file.exists():
                with open(thresholds_file, 'r') as f:
                    thresholds = json.load(f)
            else:
                thresholds = {}
            
            # Mettre à jour
            thresholds[model_name] = {
                "optimal_threshold": optimal_threshold,
                "precision": precision,
                "f1": metrics.get("f1", 0.0),
                "auc": metrics.get("auc", 0.0),
                "last_updated": datetime.now().isoformat()
            }
            
            # Sauvegarder
            with open(thresholds_file, 'w') as f:
                json.dump(thresholds, f, indent=2)
            
            logger.info(f"📊 Seuils optimaux mis à jour pour {chapter}: {optimal_threshold:.4f}")
            
        except Exception as e:
            logger.error(f"Erreur mise à jour seuils {chapter}: {e}")
    
    def check_and_retrain_all(self) -> Dict[str, Any]:
        """Vérifie et retrain tous les modèles si nécessaire"""
        results = {}
        
        for chapter in ["chap30", "chap84", "chap85"]:
            try:
                if self.should_retrain(chapter):
                    logger.info(f"🔄 Lancement du retraining pour {chapter}")
                    result = self.retrain_model(chapter)
                    results[chapter] = result
                else:
                    results[chapter] = {
                        "success": False,
                        "reason": "Conditions non remplies",
                        "chapter": chapter
                    }
                    
            except Exception as e:
                logger.error(f"Erreur retraining {chapter}: {e}")
                results[chapter] = {
                    "success": False,
                    "error": str(e),
                    "chapter": chapter
                }
        
        return results

# Instance globale du système de retraining
_retraining_system = None

def get_retraining_system(database_url: str = None) -> MLRetrainingSystem:
    """Récupère l'instance globale du système de retraining"""
    global _retraining_system
    
    if _retraining_system is None:
        if database_url is None:
            # Configuration par défaut
            database_url = "postgresql://maramata:maramata@localhost:5432/INSPECT_IA"
        
        _retraining_system = MLRetrainingSystem(database_url)
    
    return _retraining_system

def trigger_retraining(chapter: str) -> Dict[str, Any]:
    """Déclenche le retraining pour un chapitre spécifique"""
    system = get_retraining_system()
    return system.retrain_model(chapter)

def check_retraining_status() -> Dict[str, Any]:
    """Vérifie le statut de retraining de tous les chapitres"""
    system = get_retraining_system()
    return system.check_and_retrain_all()


















