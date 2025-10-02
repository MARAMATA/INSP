#!/usr/bin/env python3
"""
Script de validation approfondie de la calibration des modèles
Vérifie les points critiques : split correct, sharpness, BSS, avant/après calibration
"""

import sys
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_splits(chapter):
    """Charger le modèle et TOUS les splits (train, valid, test)"""
    logger.info(f"🔍 Chargement complet du chapitre {chapter}...")
    
    # Chemins
    models_dir = f"/Users/macbook/Desktop/inspectia_app/backend/models/chap{chapter}"
    splits_dir = f"/Users/macbook/Desktop/inspectia_app/backend/data/ml_splits/chap{chapter}"
    
    # Charger tous les splits
    X_train = pd.read_csv(f"{splits_dir}/X_train.csv")
    X_valid = pd.read_csv(f"{splits_dir}/X_valid.csv")
    X_test = pd.read_csv(f"{splits_dir}/X_test.csv")
    
    y_train = pd.read_csv(f"{splits_dir}/y_train.csv").values.ravel()
    y_valid = pd.read_csv(f"{splits_dir}/y_valid.csv").values.ravel()
    y_test = pd.read_csv(f"{splits_dir}/y_test.csv").values.ravel()
    
    # Charger les features
    features_path = f"{models_dir}/features.pkl"
    if os.path.exists(features_path):
        features = joblib.load(features_path)
        X_train = X_train[features]
        X_valid = X_valid[features]
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
    
    # Charger le modèle calibré
    model_path = f"{models_dir}/{best_model_name.lower()}_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
    
    model = joblib.load(model_path)
    
    logger.info(f"   ✅ Modèle {best_model_name} chargé")
    logger.info(f"   ✅ Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
    logger.info(f"   ✅ Taux de fraude - Train: {y_train.mean():.3f}, Valid: {y_valid.mean():.3f}, Test: {y_test.mean():.3f}")
    
    return model, X_train, X_valid, X_test, y_train, y_valid, y_test, best_model_name

def extract_base_model(calibrated_model):
    """Extraire le modèle de base d'un CalibratedClassifierCV"""
    if hasattr(calibrated_model, 'estimator'):
        return calibrated_model.estimator
    else:
        return calibrated_model

def calculate_brier_skill_score(y_true, y_prob, base_rate):
    """Calculer le Brier Skill Score vs un classifieur de base"""
    brier_score = brier_score_loss(y_true, y_prob)
    brier_score_baseline = brier_score_loss(y_true, [base_rate] * len(y_true))
    bss = 1 - (brier_score / brier_score_baseline)
    return bss

def analyze_calibration_robust(model, X_train, X_valid, X_test, y_train, y_valid, y_test, model_name, chapter):
    """Analyse robuste de la calibration"""
    logger.info(f"🎯 ANALYSE ROBUSTE - {model_name} (Chapitre {chapter})")
    logger.info("=" * 60)
    
    # 1. Vérifier si c'est un modèle calibré
    is_calibrated = isinstance(model, CalibratedClassifierCV)
    logger.info(f"📊 Modèle calibré: {is_calibrated}")
    
    if is_calibrated:
        # Extraire le modèle de base
        base_model = extract_base_model(model)
        logger.info(f"📊 Modèle de base: {type(base_model).__name__}")
        
        # Prédictions du modèle de base (non calibré)
        y_prob_base_train = base_model.predict_proba(X_train)[:, 1]
        y_prob_base_valid = base_model.predict_proba(X_valid)[:, 1]
        y_prob_base_test = base_model.predict_proba(X_test)[:, 1]
        
        # Prédictions du modèle calibré
        y_prob_cal_train = model.predict_proba(X_train)[:, 1]
        y_prob_cal_valid = model.predict_proba(X_valid)[:, 1]
        y_prob_cal_test = model.predict_proba(X_test)[:, 1]
        
        # Comparaison avant/après calibration
        logger.info("📊 COMPARAISON AVANT/APRÈS CALIBRATION:")
        
        # Sur validation (où la calibration a été ajustée)
        brier_base_valid = brier_score_loss(y_valid, y_prob_base_valid)
        brier_cal_valid = brier_score_loss(y_valid, y_prob_cal_valid)
        auc_base_valid = roc_auc_score(y_valid, y_prob_base_valid)
        auc_cal_valid = roc_auc_score(y_valid, y_prob_cal_valid)
        
        logger.info(f"   Validation (calibration ajustée ici):")
        logger.info(f"     Brier Score - Base: {brier_base_valid:.4f}, Calibré: {brier_cal_valid:.4f}")
        logger.info(f"     AUC - Base: {auc_base_valid:.4f}, Calibré: {auc_cal_valid:.4f}")
        
        # Sur test (jamais vu)
        brier_base_test = brier_score_loss(y_test, y_prob_base_test)
        brier_cal_test = brier_score_loss(y_test, y_prob_cal_test)
        auc_base_test = roc_auc_score(y_test, y_prob_base_test)
        auc_cal_test = roc_auc_score(y_test, y_prob_cal_test)
        
        logger.info(f"   Test (jamais vu):")
        logger.info(f"     Brier Score - Base: {brier_base_test:.4f}, Calibré: {brier_cal_test:.4f}")
        logger.info(f"     AUC - Base: {auc_base_test:.4f}, Calibré: {auc_cal_test:.4f}")
        
        # Vérifier que la calibration améliore sur validation mais pas forcément sur test
        brier_improvement_valid = brier_base_valid - brier_cal_valid
        brier_improvement_test = brier_base_test - brier_cal_test
        
        logger.info(f"   Amélioration Brier - Valid: {brier_improvement_valid:+.4f}, Test: {brier_improvement_test:+.4f}")
        
        if brier_improvement_valid > 0:
            logger.info("   ✅ Calibration améliore sur validation (normal)")
        else:
            logger.warning("   ⚠️ Calibration n'améliore pas sur validation (suspect)")
        
        # Utiliser les probabilités calibrées pour la suite
        y_prob_final = y_prob_cal_test
    else:
        # Modèle non calibré
        y_prob_final = model.predict_proba(X_test)[:, 1]
        logger.info("   ⚠️ Modèle non calibré détecté")
    
    # 2. Analyse de calibration avec plus de bins
    logger.info("📊 ANALYSE DE CALIBRATION DÉTAILLÉE:")
    
    # Calibration curve avec plus de bins
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_prob_final, n_bins=15, strategy='quantile'
    )
    
    # Calculer l'ECE avec plus de bins
    bin_boundaries = np.linspace(0, 1, 16)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    bin_counts = []
    bin_accuracies = []
    bin_confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob_final > bin_lower) & (y_prob_final <= bin_upper)
        prop_in_bin = in_bin.mean()
        bin_counts.append(prop_in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_test[in_bin].mean()
            avg_confidence_in_bin = y_prob_final[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
    
    brier_score = brier_score_loss(y_test, y_prob_final)
    
    logger.info(f"   Brier Score (15 bins): {brier_score:.4f}")
    logger.info(f"   ECE (15 bins): {ece:.4f}")
    
    # 3. Analyse de sharpness
    logger.info("📊 ANALYSE DE SHARPNESS:")
    
    # Histogramme des probabilités
    prob_hist, prob_bins = np.histogram(y_prob_final, bins=20, range=(0, 1))
    prob_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
    
    # Calculer la sharpness (écart-type des probabilités)
    sharpness = np.std(y_prob_final)
    mean_prob = np.mean(y_prob_final)
    
    logger.info(f"   Sharpness (std): {sharpness:.4f}")
    logger.info(f"   Probabilité moyenne: {mean_prob:.4f}")
    logger.info(f"   Probabilités > 0.8: {(y_prob_final > 0.8).mean():.3f}")
    logger.info(f"   Probabilités < 0.2: {(y_prob_final < 0.2).mean():.3f}")
    
    # 4. Brier Skill Score
    base_rate = y_test.mean()
    bss = calculate_brier_skill_score(y_test, y_prob_final, base_rate)
    
    logger.info(f"📊 BRIER SKILL SCORE:")
    logger.info(f"   Taux de base: {base_rate:.3f}")
    logger.info(f"   BSS: {bss:.4f}")
    
    if bss > 0.5:
        logger.info("   ✅ Excellent BSS (>0.5)")
    elif bss > 0.2:
        logger.info("   ✅ Bon BSS (>0.2)")
    elif bss > 0:
        logger.info("   ⚠️ BSS positif mais faible")
    else:
        logger.warning("   ❌ BSS négatif (pire que le taux de base)")
    
    # 5. Distribution des bins
    logger.info("📊 DISTRIBUTION DES BINS:")
    for i, (count, acc, conf) in enumerate(zip(bin_counts, bin_accuracies, bin_confidences)):
        if count > 0.01:  # Seulement les bins avec >1% des échantillons
            logger.info(f"   Bin {i+1}: {count:.3f} échantillons, acc={acc:.3f}, conf={conf:.3f}")
    
    return {
        'brier_score': brier_score,
        'ece': ece,
        'sharpness': sharpness,
        'bss': bss,
        'base_rate': base_rate,
        'calibration_curve': (fraction_of_positives, mean_predicted_value),
        'prob_hist': (prob_hist, prob_centers),
        'is_calibrated': is_calibrated,
        'brier_improvement_test': brier_improvement_test if is_calibrated else 0
    }

def plot_comprehensive_analysis(results):
    """Créer des graphiques complets d'analyse"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Analyse Complète de la Calibration des Modèles', fontsize=16, fontweight='bold')
    
    chapters = [30, 84, 85]
    colors = ['blue', 'red', 'green']
    
    for i, (chapter, data) in enumerate(results.items()):
        if data is None:
            continue
        
        color = colors[i]
        
        # 1. Courbe de calibration
        ax = axes[0, i]
        fraction_of_positives, mean_predicted_value = data['calibration_curve']
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
                color=color, label=f"Chapitre {chapter}", linewidth=2, markersize=6)
        ax.plot([0, 1], [0, 1], "k:", label="Parfait", linewidth=2)
        ax.set_xlabel('Probabilité moyenne prédite')
        ax.set_ylabel('Fraction de positifs')
        ax.set_title(f'Chapitre {chapter}\nBrier: {data["brier_score"]:.4f}, ECE: {data["ece"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # 2. Histogramme des probabilités (sharpness)
        ax = axes[1, i]
        prob_hist, prob_centers = data['prob_hist']
        ax.bar(prob_centers, prob_hist, width=0.05, alpha=0.7, color=color)
        ax.axvline(data['base_rate'], color='red', linestyle='--', 
                  label=f'Taux de base: {data["base_rate"]:.3f}')
        ax.set_xlabel('Probabilité prédite')
        ax.set_ylabel('Nombre d\'échantillons')
        ax.set_title(f'Sharpness: {data["sharpness"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Métriques comparatives
        ax = axes[2, i]
        metrics = ['Brier', 'ECE', 'Sharpness', 'BSS']
        values = [data['brier_score'], data['ece'], data['sharpness'], data['bss']]
        bars = ax.bar(metrics, values, color=color, alpha=0.7)
        ax.set_ylabel('Valeur')
        ax.set_title(f'Chapitre {chapter} - Métriques')
        ax.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/Users/macbook/Desktop/inspectia_app/backend/calibration_robust_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info("📊 Graphique d'analyse robuste sauvegardé: calibration_robust_analysis.png")

def main():
    """Fonction principale"""
    logger.info("🚀 VALIDATION ROBUSTE DE LA CALIBRATION")
    logger.info("=" * 70)
    
    results = {}
    
    # Analyser chaque chapitre
    for chapter in [30, 84, 85]:
        try:
            # Charger le modèle et tous les splits
            model, X_train, X_valid, X_test, y_train, y_valid, y_test, model_name = load_model_and_splits(chapter)
            
            # Analyse robuste
            analysis = analyze_calibration_robust(
                model, X_train, X_valid, X_test, y_train, y_valid, y_test, model_name, chapter
            )
            
            results[chapter] = analysis
            
            logger.info("")
            
        except Exception as e:
            logger.error(f"❌ Erreur chapitre {chapter}: {e}")
            results[chapter] = None
    
    # Créer les graphiques
    if any(data is not None for data in results.values()):
        plot_comprehensive_analysis(results)
    
    # Résumé final
    logger.info("📋 RÉSUMÉ FINAL DE LA VALIDATION")
    logger.info("=" * 70)
    
    for chapter, data in results.items():
        if data is not None:
            logger.info(f"Chapitre {chapter}:")
            logger.info(f"   Brier Score: {data['brier_score']:.4f}")
            logger.info(f"   ECE: {data['ece']:.4f}")
            logger.info(f"   Sharpness: {data['sharpness']:.4f}")
            logger.info(f"   BSS: {data['bss']:.4f}")
            logger.info(f"   Modèle calibré: {data['is_calibrated']}")
            
            # Évaluation globale
            if (data['brier_score'] < 0.1 and data['ece'] < 0.05 and 
                data['bss'] > 0.2 and data['sharpness'] > 0.1):
                logger.info(f"   ✅ CALIBRATION EXCELLENTE ET ROBUSTE")
            elif (data['brier_score'] < 0.2 and data['ece'] < 0.1 and 
                  data['bss'] > 0.1):
                logger.info(f"   ✅ CALIBRATION BONNE")
            else:
                logger.info(f"   ⚠️ CALIBRATION À AMÉLIORER")
        else:
            logger.info(f"Chapitre {chapter}: ❌ ERREUR")
        
        logger.info("")

if __name__ == "__main__":
    main()
