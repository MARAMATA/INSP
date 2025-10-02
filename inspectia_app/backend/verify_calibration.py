#!/usr/bin/env python3
"""
Script pour vérifier la calibration des meilleurs modèles de chaque chapitre
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_data(chapter):
    """Charger le modèle et les données de test pour un chapitre"""
    logger.info(f"🔍 Chargement du chapitre {chapter}...")
    
    # Chemins
    models_dir = f"/Users/macbook/Desktop/inspectia_app/backend/models/chap{chapter}"
    splits_dir = f"/Users/macbook/Desktop/inspectia_app/backend/data/ml_splits/chap{chapter}"
    
    # Charger les données de test
    X_test = pd.read_csv(f"{splits_dir}/X_test.csv")
    y_test = pd.read_csv(f"{splits_dir}/y_test.csv").values.ravel()
    
    # Charger les features
    features_path = f"{models_dir}/features.pkl"
    if os.path.exists(features_path):
        features = joblib.load(features_path)
        X_test = X_test[features]
    
    # Identifier le meilleur modèle
    if chapter == 30:
        best_model_name = "XGBoost"
    elif chapter == 84:
        best_model_name = "CatBoost"
    elif chapter == 85:
        best_model_name = "XGBoost"
    else:
        raise ValueError(f"Chapitre {chapter} non supporté")
    
    # Charger le modèle
    model_path = f"{models_dir}/{best_model_name.lower()}_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    model = joblib.load(model_path)
    
    logger.info(f"   ✅ Modèle {best_model_name} chargé")
    logger.info(f"   ✅ Données de test: {X_test.shape}")
    logger.info(f"   ✅ Taux de fraude: {y_test.mean():.3f}")
    
    return model, X_test, y_test, best_model_name

def check_calibration(model, X_test, y_test, model_name, chapter):
    """Vérifier la calibration d'un modèle"""
    logger.info(f"🎯 Vérification de la calibration - {model_name} (Chapitre {chapter})")
    
    # Obtenir les probabilités prédites
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        logger.error(f"   ❌ Erreur prédiction: {e}")
        return None, None, None
    
    # Calculer la calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_prob, n_bins=10, strategy='uniform'
    )
    
    # Calculer le Brier Score
    brier_score = brier_score_loss(y_test, y_prob)
    
    # Calculer l'ECE (Expected Calibration Error)
    bin_boundaries = np.linspace(0, 1, 11)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_test[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    logger.info(f"   📊 Brier Score: {brier_score:.4f}")
    logger.info(f"   📊 ECE (Expected Calibration Error): {ece:.4f}")
    
    # Interprétation
    if brier_score < 0.1:
        brier_interpretation = "Excellent"
    elif brier_score < 0.2:
        brier_interpretation = "Bon"
    elif brier_score < 0.3:
        brier_interpretation = "Moyen"
    else:
        brier_interpretation = "Mauvais"
    
    if ece < 0.05:
        ece_interpretation = "Excellent"
    elif ece < 0.1:
        ece_interpretation = "Bon"
    elif ece < 0.2:
        ece_interpretation = "Moyen"
    else:
        ece_interpretation = "Mauvais"
    
    logger.info(f"   🎯 Brier Score: {brier_interpretation}")
    logger.info(f"   🎯 ECE: {ece_interpretation}")
    
    return (fraction_of_positives, mean_predicted_value), {
        'brier_score': brier_score,
        'ece': ece,
        'brier_interpretation': brier_interpretation,
        'ece_interpretation': ece_interpretation
    }

def plot_calibration_curves(results):
    """Créer un graphique de calibration pour tous les modèles"""
    plt.figure(figsize=(15, 5))
    
    for i, (chapter, data) in enumerate(results.items(), 1):
        if data is None:
            continue
            
        plt.subplot(1, 3, i)
        
        # Courbe de calibration
        fraction_of_positives, mean_predicted_value = data['calibration_curve']
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f"Chapitre {chapter} ({data['model_name']})", 
                linewidth=2, markersize=6)
        
        # Ligne de calibration parfaite
        plt.plot([0, 1], [0, 1], "k:", label="Calibration parfaite", linewidth=2)
        
        plt.xlabel('Probabilité moyenne prédite')
        plt.ylabel('Fraction de positifs')
        plt.title(f'Chapitre {chapter} - {data["model_name"]}\n'
                 f'Brier: {data["metrics"]["brier_score"]:.4f} ({data["metrics"]["brier_interpretation"]})\n'
                 f'ECE: {data["metrics"]["ece"]:.4f} ({data["metrics"]["ece_interpretation"]})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('/Users/macbook/Desktop/inspectia_app/backend/calibration_verification.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("📊 Graphique de calibration sauvegardé: calibration_verification.png")

def main():
    """Fonction principale"""
    logger.info("🚀 VÉRIFICATION DE LA CALIBRATION DES MEILLEURS MODÈLES")
    logger.info("=" * 70)
    
    results = {}
    
    # Vérifier chaque chapitre
    for chapter in [30, 84, 85]:
        try:
            # Charger le modèle et les données
            model, X_test, y_test, model_name = load_model_and_data(chapter)
            
            # Vérifier la calibration
            calibration_curve_data, metrics = check_calibration(
                model, X_test, y_test, model_name, chapter
            )
            
            if calibration_curve_data is not None:
                results[chapter] = {
                    'model_name': model_name,
                    'calibration_curve': calibration_curve_data,
                    'metrics': metrics
                }
            
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Erreur chapitre {chapter}: {e}")
            results[chapter] = None
    
    # Créer le graphique de calibration
    if any(data is not None for data in results.values()):
        plot_calibration_curves(results)
    
    # Résumé final
    logger.info("📋 RÉSUMÉ DE LA CALIBRATION")
    logger.info("=" * 70)
    
    for chapter, data in results.items():
        if data is not None:
            metrics = data['metrics']
            logger.info(f"Chapitre {chapter} ({data['model_name']}):")
            logger.info(f"   Brier Score: {metrics['brier_score']:.4f} ({metrics['brier_interpretation']})")
            logger.info(f"   ECE: {metrics['ece']:.4f} ({metrics['ece_interpretation']})")
            
            # Vérification de la calibration
            if metrics['brier_score'] < 0.1 and metrics['ece'] < 0.05:
                logger.info(f"   ✅ CALIBRATION EXCELLENTE")
            elif metrics['brier_score'] < 0.2 and metrics['ece'] < 0.1:
                logger.info(f"   ✅ CALIBRATION BONNE")
            else:
                logger.info(f"   ⚠️ CALIBRATION À AMÉLIORER")
        else:
            logger.info(f"Chapitre {chapter}: ❌ ERREUR")
        
        logger.info("")

if __name__ == "__main__":
    main()
