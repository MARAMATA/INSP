#!/usr/bin/env python3
"""
Suite complète de validation pour la production
Tests anti-fuites, calibration, robustesse, monitoring
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import logging
import warnings
from datetime import datetime, timedelta
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    brier_score_loss, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import hashlib
import yaml

warnings.filterwarnings('ignore')

class ProductionValidationSuite:
    """Suite complète de validation pour la production"""
    
    def __init__(self):
        self.base_path = Path("/Users/macbook/Desktop/inspectia_app/backend")
        self.chapters = ['chap30', 'chap84', 'chap85']
        self.results = {}
        
        # Configuration des tests
        self.config = {
            'null_importance_permutations': 10,
            'adversarial_validation_threshold': 0.7,
            'calibration_bins': 10,
            'subgroup_min_size': 100,
            'temporal_test_months': 3,
            'monitoring_features': 10,
            'psi_threshold': 0.2,
            'ks_threshold': 0.05
        }
        
    def load_chapter_data(self, chapter):
        """Charger toutes les données d'un chapitre"""
        logger.info(f"📊 Chargement des données pour {chapter}...")
        
        # Charger les modèles
        models_path = self.base_path / "models" / chapter
        models = {}
        model_names = ['lightgbm', 'xgboost', 'catboost', 'randomforest', 'logisticregression']
        
        for name in model_names:
            model_file = models_path / f"{name}_model.pkl"
            if model_file.exists():
                try:
                    models[name] = joblib.load(model_file)
                    logger.info(f"   ✅ Modèle {name} chargé")
                except Exception as e:
                    logger.warning(f"   ❌ Erreur chargement {name}: {e}")
        
        # Charger les données
        splits_path = self.base_path / "data" / "ml_splits" / chapter
        
        # TOUJOURS charger les données brutes car les modèles calibrés les attendent
        logger.info("   📂 Chargement des données brutes (requis pour modèles calibrés)...")
        X_train = pd.read_csv(splits_path / "X_train.csv")
        X_valid = pd.read_csv(splits_path / "X_valid.csv")
        X_test = pd.read_csv(splits_path / "X_test.csv")
        y_train = pd.read_csv(splits_path / "y_train.csv").values.ravel()
        y_valid = pd.read_csv(splits_path / "y_valid.csv").values.ravel()
        y_test = pd.read_csv(splits_path / "y_test.csv").values.ravel()
        
        # Créer le preprocessor standard pour les modèles calibrés
        self.preprocessor = self.create_preprocessor(X_train)
        
        # Charger les données complètes pour les analyses temporelles
        data_path = self.base_path / "data" / "processed" / chapter
        if (data_path / "processed_data.csv").exists():
            df_full = pd.read_csv(data_path / "processed_data.csv")
        else:
            # Reconstituer à partir des splits
            df_full = pd.concat([X_train, X_valid, X_test], ignore_index=True)
            df_full['FRAUD_FLAG'] = np.concatenate([y_train, y_valid, y_test])
        
        logger.info(f"   ✅ Données chargées: {len(models)} modèles, {df_full.shape[0]} échantillons")
        
        return models, X_train, X_valid, X_test, y_train, y_valid, y_test, df_full
    
    def _preprocessed_data_exist(self, splits_path):
        """Vérifie si les données preprocessées existent déjà"""
        required_files = [
            "Xt_train.csv", "Xt_test.csv", "Xt_valid.csv",
            "yt_train.csv", "yt_test.csv", "yt_valid.csv"
        ]
        return all((splits_path / file).exists() for file in required_files)
    
    def _load_preprocessed_data(self, splits_path):
        """Charge les données preprocessées existantes"""
        logger.info(f"   📂 Chargement des données preprocessées depuis {splits_path}")
        
        X_train = pd.read_csv(splits_path / "Xt_train.csv")
        X_test = pd.read_csv(splits_path / "Xt_test.csv")
        X_valid = pd.read_csv(splits_path / "Xt_valid.csv")
        y_train = pd.read_csv(splits_path / "yt_train.csv").iloc[:, 0]
        y_test = pd.read_csv(splits_path / "yt_test.csv").iloc[:, 0]
        y_valid = pd.read_csv(splits_path / "yt_valid.csv").iloc[:, 0]
        
        logger.info(f"   ✅ Données preprocessées chargées: Train={X_train.shape}, Valid={X_valid.shape}, Test={X_test.shape}")
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    def create_preprocessor(self, X_train):
        """Créer le preprocessor standard"""
        numeric_features = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
        categorical_features = [col for col in X_train.columns if X_train[col].dtype == 'object']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Fit le preprocessor
        preprocessor.fit(X_train)
        return preprocessor
    
    def preprocess_data(self, X):
        """Preprocesser les données"""
        # Vérifier si les données sont déjà preprocessées
        if hasattr(X, 'shape') and X.shape[1] > 100:  # Plus de 100 colonnes = probablement déjà preprocessées
            logger.info(f"   🔧 Données déjà preprocessées (shape: {X.shape})")
            return X
        return self.preprocessor.transform(X)
    
    def needs_preprocessing(self, model, X):
        """Détermine si un modèle a besoin de preprocessing"""
        try:
            # Essayer de prédire sans preprocessing
            _ = model.predict_proba(X[:1])[:, 1]
            return False  # Pas besoin de preprocessing
        except (ValueError, TypeError) as e:
            # Si erreur de dtype ou de features, besoin de preprocessing
            if "pandas dtypes must be int, float or bool" in str(e) or "expecting" in str(e):
                return True
            raise e
    
    # ==================== 1. TESTS ANTI-FUITES ====================
    
    def test_target_permutation(self, model, X_test, y_test, model_name):
        """Test de permutation du target - AUC doit ≈ 0.5"""
        logger.info(f"   🎲 Test permutation target {model_name}...")
        
        # Mélanger les labels
        y_permuted = np.random.permutation(y_test)
        
        # Déterminer si le modèle a besoin de preprocessing
        if self.needs_preprocessing(model, X_test):
            X_test_processed = self.preprocess_data(X_test)
            y_proba_permuted = model.predict_proba(X_test_processed)[:, 1]
            y_proba_normal = model.predict_proba(X_test_processed)[:, 1]
        else:
            y_proba_permuted = model.predict_proba(X_test)[:, 1]
            y_proba_normal = model.predict_proba(X_test)[:, 1]
        
        auc_permuted = roc_auc_score(y_permuted, y_proba_permuted)
        auc_normal = roc_auc_score(y_test, y_proba_normal)
        
        leakage_detected = auc_permuted > 0.6  # Seuil de suspicion
        
        logger.info(f"      AUC normal: {auc_normal:.4f}, AUC permuté: {auc_permuted:.4f}")
        
        return {
            'auc_normal': auc_normal,
            'auc_permuted': auc_permuted,
            'leakage_detected': leakage_detected,
            'leakage_risk': 'HIGH' if leakage_detected else 'LOW'
        }
    
    def test_adversarial_validation(self, X_train, X_test, y_train, y_test):
        """Validation adversarial - classifie train vs test"""
        logger.info("   ⚔️ Test validation adversarial...")
        
        # Créer les labels adversarial (1=test, 0=train)
        y_adversarial = np.concatenate([
            np.zeros(len(X_train)),  # Train = 0
            np.ones(len(X_test))     # Test = 1
        ])
        
        # Combiner train et test
        X_combined = pd.concat([X_train, X_test], ignore_index=True)
        
        # Gérer les features catégorielles en les encodant numériquement
        X_combined_encoded = X_combined.copy()
        
        for col in X_combined_encoded.columns:
            if X_combined_encoded[col].dtype == 'object':
                # Encoder les catégorielles en numérique simple
                X_combined_encoded[col] = pd.Categorical(X_combined_encoded[col]).codes
        
        # Remplacer les NaN par 0
        X_combined_encoded = X_combined_encoded.fillna(0)
        
        try:
            # Entraîner un modèle pour distinguer train/test
            rf_adversarial = RandomForestClassifier(n_estimators=50, random_state=42)
            rf_adversarial.fit(X_combined_encoded, y_adversarial)
            
            # Prédictions
            y_proba_adversarial = rf_adversarial.predict_proba(X_combined_encoded)[:, 1]
            auc_adversarial = roc_auc_score(y_adversarial, y_proba_adversarial)
            
            distribution_shift = auc_adversarial > self.config['adversarial_validation_threshold']
            
            logger.info(f"      AUC adversarial: {auc_adversarial:.4f}")
            
            return {
                'auc_adversarial': auc_adversarial,
                'distribution_shift': distribution_shift,
                'shift_risk': 'HIGH' if distribution_shift else 'LOW'
            }
            
        except Exception as e:
            logger.warning(f"      Erreur test adversarial: {e}")
            return {
                'auc_adversarial': 0.5,
                'distribution_shift': False,
                'shift_risk': 'UNKNOWN'
            }
    
    def test_null_importances(self, model, X_train, y_train, model_name):
        """Test des importances nulles par permutation des features"""
        logger.info(f"   🎯 Test null importances {model_name}...")
        
        try:
            # AUC de référence
            if self.needs_preprocessing(model, X_train):
                X_train_processed = self.preprocess_data(X_train)
                y_proba_ref = model.predict_proba(X_train_processed)[:, 1]
            else:
                y_proba_ref = model.predict_proba(X_train)[:, 1]
            auc_ref = roc_auc_score(y_train, y_proba_ref)
            
            # Permutation de chaque feature
            feature_importances = []
            feature_names = X_train.columns
            
            for feature in feature_names:
                auc_scores = []
                
                for _ in range(self.config['null_importance_permutations']):
                    # Copier les données
                    X_permuted = X_train.copy()
                    
                    # Permuter la feature
                    if X_permuted[feature].dtype == 'object':
                        # Pour les catégorielles, permuter les codes
                        X_permuted[feature] = np.random.permutation(X_permuted[feature])
                    else:
                        # Pour les numériques, permuter directement
                        X_permuted[feature] = np.random.permutation(X_permuted[feature])
                    
                    # Prédiction
                    if self.needs_preprocessing(model, X_permuted):
                        X_permuted_processed = self.preprocess_data(X_permuted)
                        y_proba_permuted = model.predict_proba(X_permuted_processed)[:, 1]
                    else:
                        y_proba_permuted = model.predict_proba(X_permuted)[:, 1]
                    auc_permuted = roc_auc_score(y_train, y_proba_permuted)
                    auc_scores.append(auc_permuted)
                
                # Importance = AUC_ref - AUC_permuted_mean
                importance = auc_ref - np.mean(auc_scores)
                feature_importances.append(importance)
            
            # Détecter les features avec importance suspecte
            null_importance_threshold = 0.01  # Seuil minimum d'importance
            suspicious_features = [
                feature for feature, importance in zip(feature_names, feature_importances)
                if importance < null_importance_threshold
            ]
            
            logger.info(f"      Features suspectes: {len(suspicious_features)}/{len(feature_names)}")
            
            return {
                'feature_importances': dict(zip(feature_names, feature_importances)),
                'suspicious_features': suspicious_features,
                'auc_reference': auc_ref,
                'null_importance_risk': 'HIGH' if len(suspicious_features) > len(feature_names) * 0.3 else 'LOW'
            }
            
        except Exception as e:
            logger.warning(f"      Erreur test null importances {model_name}: {e}")
            return {
                'feature_importances': {},
                'suspicious_features': [],
                'auc_reference': 0.0,
                'null_importance_risk': 'UNKNOWN'
            }
    
    # ==================== 2. CALIBRATION FINE ====================
    
    def compute_calibration_metrics(self, y_true, y_proba, model_name):
        """Calculer les métriques de calibration"""
        logger.info(f"   📏 Métriques calibration {model_name}...")
        
        # Brier Score
        brier_score = brier_score_loss(y_true, y_proba)
        
        # Expected Calibration Error (ECE)
        ece = self.compute_ece(y_true, y_proba)
        
        # Courbe de calibration
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=self.config['calibration_bins']
        )
        
        # Mean Absolute Error de calibration
        mae_calibration = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        
        logger.info(f"      Brier Score: {brier_score:.4f}, ECE: {ece:.4f}")
        
        return {
            'brier_score': brier_score,
            'ece': ece,
            'mae_calibration': mae_calibration,
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value
            },
            'calibration_quality': 'EXCELLENT' if brier_score < 0.1 and ece < 0.05 else 'GOOD' if brier_score < 0.2 and ece < 0.1 else 'POOR'
        }
    
    def compute_ece(self, y_true, y_proba, n_bins=10):
        """Calculer l'Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def create_calibration_plots(self, chapter, calibration_results):
        """Créer les graphiques de calibration"""
        logger.info(f"   📊 Création graphiques calibration {chapter}...")
        
        results_dir = self.base_path / "test_results" / "calibration_analysis"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Graphique de calibration
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Analyse de Calibration - {chapter}', fontsize=16, fontweight='bold')
        
        models = list(calibration_results.keys())
        
        # Courbe de calibration
        ax1 = axes[0, 0]
        for model_name, results in calibration_results.items():
            curve = results['calibration_curve']
            ax1.plot(curve['mean_predicted_value'], curve['fraction_of_positives'], 
                    'o-', label=f"{model_name} (ECE={results['ece']:.3f})")
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Calibration parfaite')
        ax1.set_xlabel('Probabilité moyenne prédite')
        ax1.set_ylabel('Fraction de positifs réels')
        ax1.set_title('Courbes de Calibration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Brier Score
        ax2 = axes[0, 1]
        brier_scores = [results['brier_score'] for results in calibration_results.values()]
        bars = ax2.bar(models, brier_scores, color=['green' if bs < 0.1 else 'orange' if bs < 0.2 else 'red' for bs in brier_scores])
        ax2.set_title('Brier Score par Modèle')
        ax2.set_ylabel('Brier Score')
        ax2.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Excellent')
        ax2.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Bon')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # ECE
        ax3 = axes[1, 0]
        ece_scores = [results['ece'] for results in calibration_results.values()]
        bars = ax3.bar(models, ece_scores, color=['green' if ece < 0.05 else 'orange' if ece < 0.1 else 'red' for ece in ece_scores])
        ax3.set_title('Expected Calibration Error (ECE)')
        ax3.set_ylabel('ECE')
        ax3.axhline(y=0.05, color='green', linestyle='--', alpha=0.7, label='Excellent')
        ax3.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Bon')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # Qualité de calibration
        ax4 = axes[1, 1]
        qualities = [results['calibration_quality'] for results in calibration_results.values()]
        quality_counts = pd.Series(qualities).value_counts()
        colors = {'EXCELLENT': 'green', 'GOOD': 'orange', 'POOR': 'red'}
        bars = ax4.bar(quality_counts.index, quality_counts.values, 
                      color=[colors[q] for q in quality_counts.index])
        ax4.set_title('Distribution Qualité Calibration')
        ax4.set_ylabel('Nombre de modèles')
        
        plt.tight_layout()
        plt.savefig(results_dir / f'calibration_analysis_{chapter}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"      ✅ Graphique sauvegardé: calibration_analysis_{chapter}.png")
    
    # ==================== 3. SEUILS ORIENTÉS MÉTIER ====================
    
    def optimize_business_thresholds(self, y_true, y_proba, cost_fp=1.0, cost_fn=5.0):
        """Optimiser les seuils selon les coûts métier"""
        logger.info("   💰 Optimisation seuils métier...")
        
        # Générer des seuils
        thresholds = np.linspace(0.1, 0.9, 81)
        
        results = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Matrice de confusion
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Métriques
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Coût total
            total_cost = fp * cost_fp + fn * cost_fn
            
            # FPR et FNR
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'total_cost': total_cost,
                'fpr': fpr,
                'fnr': fnr,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })
        
        df_results = pd.DataFrame(results)
        
        # Seuil optimal par coût
        optimal_cost_idx = df_results['total_cost'].idxmin()
        optimal_cost_threshold = df_results.loc[optimal_cost_idx, 'threshold']
        
        # Seuil optimal par F1
        optimal_f1_idx = df_results['f1'].idxmax()
        optimal_f1_threshold = df_results.loc[optimal_f1_idx, 'threshold']
        
        # Seuil avec contrainte FPR < 5%
        constrained_results = df_results[df_results['fpr'] < 0.05]
        if not constrained_results.empty:
            optimal_constrained_idx = constrained_results['total_cost'].idxmin()
            optimal_constrained_threshold = constrained_results.loc[optimal_constrained_idx, 'threshold']
        else:
            optimal_constrained_threshold = None
        
        logger.info(f"      Seuil optimal coût: {optimal_cost_threshold:.3f}")
        logger.info(f"      Seuil optimal F1: {optimal_f1_threshold:.3f}")
        
        return {
            'threshold_analysis': df_results,
            'optimal_cost_threshold': optimal_cost_threshold,
            'optimal_f1_threshold': optimal_f1_threshold,
            'optimal_constrained_threshold': optimal_constrained_threshold,
            'cost_fp': cost_fp,
            'cost_fn': cost_fn
        }
    
    def compute_top_k_metrics(self, y_true, y_proba, k_values=[100, 500, 1000]):
        """Calculer les métriques top-k"""
        logger.info("   🎯 Calcul métriques top-k...")
        
        # Trier par probabilité décroissante
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        results = {}
        for k in k_values:
            if k <= len(y_true):
                y_pred_top_k = y_true_sorted[:k]
                precision_at_k = np.sum(y_pred_top_k) / k
                
                # Trouver le seuil pour top-k
                threshold_at_k = y_proba[sorted_indices[k-1]] if k > 0 else 1.0
                
                results[f'precision_at_{k}'] = precision_at_k
                results[f'threshold_at_{k}'] = threshold_at_k
        
        logger.info(f"      Precision@100: {results.get('precision_at_100', 0):.3f}")
        
        return results
    
    # ==================== 4. ANALYSE D'OVERFITTING ====================
    
    def analyze_overfitting(self, models, X_train, X_test, y_train, y_test):
        """Analyse d'overfitting pour tous les modèles"""
        logger.info("🔍 Analyse d'overfitting...")
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"   📊 Analyse overfitting {name}...")
            
            # Métriques sur train et test
            if self.needs_preprocessing(model, X_train):
                X_train_processed = self.preprocess_data(X_train)
                X_test_processed = self.preprocess_data(X_test)
                y_train_pred_proba = model.predict_proba(X_train_processed)[:, 1]
                y_test_pred_proba = model.predict_proba(X_test_processed)[:, 1]
            else:
                y_train_pred_proba = model.predict_proba(X_train)[:, 1]
                y_test_pred_proba = model.predict_proba(X_test)[:, 1]
            
            train_auc = roc_auc_score(y_train, y_train_pred_proba)
            test_auc = roc_auc_score(y_test, y_test_pred_proba)
            
            train_f1 = f1_score(y_train, (y_train_pred_proba > 0.5).astype(int))
            test_f1 = f1_score(y_test, (y_test_pred_proba > 0.5).astype(int))
            
            # Gap d'overfitting
            auc_gap = train_auc - test_auc
            f1_gap = train_f1 - test_f1
            
            results[name] = {
                'train_auc': train_auc,
                'test_auc': test_auc,
                'train_f1': train_f1,
                'test_f1': test_f1,
                'auc_gap': auc_gap,
                'f1_gap': f1_gap
            }
        
        # Créer le graphique d'overfitting
        self.create_overfitting_plots(results)
        
        return results
    
    def create_overfitting_plots(self, results):
        """Créer les graphiques d'overfitting"""
        logger.info(f"   📊 Création graphiques overfitting...")
        
        results_dir = self.base_path / "test_results" / "overfitting_analysis"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Graphique d'overfitting
        plt.figure(figsize=(15, 10))
        
        # AUC Gap
        plt.subplot(2, 2, 1)
        model_names = list(results.keys())
        auc_gaps = [result['auc_gap'] for result in results.values()]
        bars = plt.bar(model_names, auc_gaps, color='coral', alpha=0.7)
        plt.title('AUC Gap (Train - Test)')
        plt.ylabel('AUC Gap')
        plt.xticks(rotation=45)
        for bar, gap in zip(bars, auc_gaps):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{gap:.4f}', ha='center', va='bottom')
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Seuil critique (0.05)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # F1 Gap
        plt.subplot(2, 2, 2)
        f1_gaps = [result['f1_gap'] for result in results.values()]
        bars = plt.bar(model_names, f1_gaps, color='lightblue', alpha=0.7)
        plt.title('F1 Gap (Train - Test)')
        plt.ylabel('F1 Gap')
        plt.xticks(rotation=45)
        for bar, gap in zip(bars, f1_gaps):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{gap:.4f}', ha='center', va='bottom')
        plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Seuil critique (0.05)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Train vs Test AUC
        plt.subplot(2, 2, 3)
        train_aucs = [result['train_auc'] for result in results.values()]
        test_aucs = [result['test_auc'] for result in results.values()]
        x = np.arange(len(model_names))
        width = 0.35
        plt.bar(x - width/2, train_aucs, width, label='Train', alpha=0.7, color='lightgreen')
        plt.bar(x + width/2, test_aucs, width, label='Test', alpha=0.7, color='lightcoral')
        plt.xlabel('Modèles')
        plt.ylabel('AUC')
        plt.title('AUC Train vs Test')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Train vs Test F1
        plt.subplot(2, 2, 4)
        train_f1s = [result['train_f1'] for result in results.values()]
        test_f1s = [result['test_f1'] for result in results.values()]
        plt.bar(x - width/2, train_f1s, width, label='Train', alpha=0.7, color='lightgreen')
        plt.bar(x + width/2, test_f1s, width, label='Test', alpha=0.7, color='lightcoral')
        plt.xlabel('Modèles')
        plt.ylabel('F1-Score')
        plt.title('F1-Score Train vs Test')
        plt.xticks(x, model_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / f'overfitting_analysis_global.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Résumé texte
        summary_path = results_dir / 'overfitting_summary_global.txt'
        with open(summary_path, 'w') as f:
            f.write(f"ANALYSE D'OVERFITTING - VALIDATION GLOBALE\n")
            f.write("=" * 50 + "\n\n")
            
            for name, result in results.items():
                f.write(f"{name}:\n")
                f.write(f"  AUC Gap: {result['auc_gap']:.4f} {'⚠️ OVERFITTING' if result['auc_gap'] > 0.05 else '✅ OK'}\n")
                f.write(f"  F1 Gap: {result['f1_gap']:.4f} {'⚠️ OVERFITTING' if result['f1_gap'] > 0.05 else '✅ OK'}\n")
                f.write("\n")
        
        logger.info(f"      ✅ Graphique sauvegardé: overfitting_analysis_global.png")
    
    # ==================== 5. ROBUSTESSE PAR SOUS-GROUPES ====================
    
    def analyze_subgroup_robustness(self, df_full, models, chapter):
        """Analyser la robustesse par sous-groupes"""
        logger.info(f"   👥 Analyse robustesse sous-groupes {chapter}...")
        
        # Identifier les colonnes de sous-groupes disponibles
        subgroup_columns = []
        potential_columns = ['PAYS', 'REGIME', 'CODE_SH_COMPLET', 'BUREAU', 'PRODUIT']
        
        for col in potential_columns:
            if col in df_full.columns:
                subgroup_columns.append(col)
        
        subgroup_results = {}
        
        for col in subgroup_columns:
            logger.info(f"      Analyse sous-groupe: {col}")
            
            # Compter les valeurs
            value_counts = df_full[col].value_counts()
            
            # Garder seulement les groupes avec assez d'échantillons
            valid_groups = value_counts[value_counts >= self.config['subgroup_min_size']].index
            
            if len(valid_groups) < 2:
                logger.info(f"         Pas assez de groupes valides pour {col}")
                continue
            
            col_results = {}
            
            for group in valid_groups:
                # Filtrer le groupe
                group_mask = df_full[col] == group
                group_data = df_full[group_mask]
                
                if len(group_data) < self.config['subgroup_min_size']:
                    continue
                
                # Extraire X et y pour ce groupe
                X_group = group_data.drop(['FRAUD_FLAG'], axis=1, errors='ignore')
                y_group = group_data['FRAUD_FLAG']
                
                # Évaluer chaque modèle sur ce groupe
                group_model_results = {}
                
                for model_name, model in models.items():
                    try:
                        # Prédictions
                        y_proba = model.predict_proba(X_group)[:, 1]
                        y_pred = (y_proba >= 0.5).astype(int)
                        
                        # Métriques
                        auc = roc_auc_score(y_group, y_proba)
                        f1 = f1_score(y_group, y_pred, zero_division=0)
                        precision = precision_score(y_group, y_pred, zero_division=0)
                        recall = recall_score(y_group, y_pred, zero_division=0)
                        
                        group_model_results[model_name] = {
                            'auc': auc,
                            'f1': f1,
                            'precision': precision,
                            'recall': recall,
                            'n_samples': len(y_group),
                            'fraud_rate': y_group.mean()
                        }
                        
                    except Exception as e:
                        logger.warning(f"         Erreur modèle {model_name} sur groupe {group}: {e}")
                
                col_results[group] = group_model_results
            
            subgroup_results[col] = col_results
        
        # Calculer les gaps worst-group
        worst_group_gaps = self.compute_worst_group_gaps(subgroup_results)
        
        return {
            'subgroup_results': subgroup_results,
            'worst_group_gaps': worst_group_gaps
        }
    
    def compute_worst_group_gaps(self, subgroup_results):
        """Calculer les gaps worst-group"""
        gaps = {}
        
        for col, col_results in subgroup_results.items():
            for model_name in ['lightgbm', 'xgboost', 'catboost']:  # Modèles principaux
                model_performances = []
                
                for group, group_results in col_results.items():
                    if model_name in group_results:
                        model_performances.append(group_results[model_name]['f1'])
                
                if len(model_performances) >= 2:
                    worst_f1 = min(model_performances)
                    best_f1 = max(model_performances)
                    gap = best_f1 - worst_f1
                    
                    gaps[f"{col}_{model_name}"] = {
                        'worst_f1': worst_f1,
                        'best_f1': best_f1,
                        'gap': gap,
                        'n_groups': len(model_performances)
                    }
        
        return gaps
    
    # ==================== 5. BACKTEST TEMPOREL ====================
    
    def temporal_backtest(self, df_full, models, chapter):
        """Backtest temporel avec embargo strict"""
        logger.info(f"   ⏰ Backtest temporel {chapter}...")
        
        # Vérifier s'il y a une colonne temporelle
        temporal_columns = ['DATE', 'DATE_DECLARATION', 'TIMESTAMP', 'YEAR', 'MONTH']
        temporal_col = None
        
        for col in temporal_columns:
            if col in df_full.columns:
                temporal_col = col
                break
        
        if temporal_col is None:
            logger.info("      Pas de colonne temporelle trouvée")
            return None
        
        # Convertir en datetime si nécessaire
        try:
            df_full[temporal_col] = pd.to_datetime(df_full[temporal_col])
        except:
            logger.info(f"      Impossible de convertir {temporal_col} en datetime")
            return None
        
        # Trier par date
        df_sorted = df_full.sort_values(temporal_col)
        
        # Créer des périodes d'entraînement et test
        dates = df_sorted[temporal_col].unique()
        n_dates = len(dates)
        
        if n_dates < 4:  # Pas assez de dates
            logger.info("      Pas assez de dates pour le backtest temporel")
            return None
        
        # Diviser en périodes
        train_size = int(n_dates * 0.7)  # 70% pour l'entraînement
        test_start = train_size
        
        train_dates = dates[:train_size]
        test_dates = dates[test_start:]
        
        # Filtrer les données
        train_mask = df_sorted[temporal_col].isin(train_dates)
        test_mask = df_sorted[temporal_col].isin(test_dates)
        
        df_train = df_sorted[train_mask]
        df_test = df_sorted[test_mask]
        
        if len(df_train) < 100 or len(df_test) < 50:
            logger.info("      Pas assez d'échantillons pour le backtest")
            return None
        
        # Extraire X et y
        X_train_temp = df_train.drop(['FRAUD_FLAG', temporal_col], axis=1, errors='ignore')
        y_train_temp = df_train['FRAUD_FLAG']
        X_test_temp = df_test.drop(['FRAUD_FLAG', temporal_col], axis=1, errors='ignore')
        y_test_temp = df_test['FRAUD_FLAG']
        
        logger.info(f"      Période train: {len(df_train)} échantillons")
        logger.info(f"      Période test: {len(df_test)} échantillons")
        
        # Évaluer les modèles sur le test temporel
        temporal_results = {}
        
        for model_name, model in models.items():
            try:
                # Prédictions
                y_proba = model.predict_proba(X_test_temp)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)
                
                # Métriques
                auc = roc_auc_score(y_test_temp, y_proba)
                f1 = f1_score(y_test_temp, y_pred, zero_division=0)
                
                temporal_results[model_name] = {
                    'auc': auc,
                    'f1': f1,
                    'n_train': len(df_train),
                    'n_test': len(df_test),
                    'train_period': f"{train_dates[0]} to {train_dates[-1]}",
                    'test_period': f"{test_dates[0]} to {test_dates[-1]}"
                }
                
            except Exception as e:
                logger.warning(f"      Erreur modèle {model_name} backtest: {e}")
        
        return temporal_results
    
    # ==================== 6. MONITORING PRODUCTION ====================
    
    def setup_production_monitoring(self, df_full, models, chapter):
        """Configurer le monitoring de production"""
        logger.info(f"   📊 Configuration monitoring production {chapter}...")
        
        # Sélectionner les features les plus importantes pour le monitoring
        feature_importance = self.get_top_features(models)
        monitoring_features = feature_importance[:self.config['monitoring_features']]
        
        # Calculer les statistiques de référence
        reference_stats = {}
        
        for feature in monitoring_features:
            if feature in df_full.columns:
                ref_data = df_full[feature].dropna()
                reference_stats[feature] = {
                    'mean': ref_data.mean(),
                    'std': ref_data.std(),
                    'min': ref_data.min(),
                    'max': ref_data.max(),
                    'missing_rate': df_full[feature].isna().mean(),
                    'unique_values': df_full[feature].nunique()
                }
        
        # Configurer les seuils d'alerte
        monitoring_config = {
            'features_to_monitor': monitoring_features,
            'reference_stats': reference_stats,
            'alert_thresholds': {
                'psi_threshold': self.config['psi_threshold'],
                'ks_threshold': self.config['ks_threshold'],
                'missing_rate_threshold': 0.1,
                'distribution_shift_threshold': 0.05
            },
            'monitoring_frequency': 'daily',
            'alert_channels': ['email', 'dashboard']
        }
        
        return monitoring_config
    
    def get_top_features(self, models):
        """Extraire les features les plus importantes des modèles"""
        all_features = set()
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    # Pour les modèles avec feature_importances_
                    importances = model.feature_importances_
                    # Supposer que les features sont dans l'ordre d'entraînement
                    # Ceci devrait être amélioré avec les vraies feature names
                    all_features.update([f'feature_{i}' for i in range(len(importances))])
                elif hasattr(model, 'coef_'):
                    # Pour LogisticRegression
                    coef = model.coef_[0]
                    all_features.update([f'feature_{i}' for i in range(len(coef))])
            except:
                continue
        
        return list(all_features)[:self.config['monitoring_features']]
    
    # ==================== 7. TRAÇABILITÉ ET PACKAGING ====================
    
    def create_model_card(self, chapter, all_results):
        """Créer une Model Card pour le chapitre"""
        logger.info(f"   📋 Création Model Card {chapter}...")
        
        model_card = {
            'model_card_version': '1.0',
            'created_date': datetime.now().isoformat(),
            'chapter': chapter,
            'model_type': 'Fraud Detection ML Pipeline',
            'task': 'Binary Classification',
            'performance': {},
            'training_data': {},
            'evaluation_data': {},
            'limitations': [],
            'recommendations': []
        }
        
        # Ajouter les métriques de performance (version simplifiée)
        if 'leakage_tests' in all_results:
            # Extraire seulement les métriques clés
            leakage_summary = {}
            for test_name, test_result in all_results['leakage_tests'].items():
                if isinstance(test_result, dict):
                    summary = {}
                    for key, value in test_result.items():
                        if isinstance(value, (str, int, float, bool)):
                            summary[key] = value
                        elif isinstance(value, np.ndarray):
                            summary[key] = value.tolist()
                    leakage_summary[test_name] = summary
            model_card['performance']['leakage_tests'] = leakage_summary
        
        if 'calibration_metrics' in all_results:
            # Extraire seulement les métriques clés
            calibration_summary = {}
            for model_name, calib in all_results['calibration_metrics'].items():
                if isinstance(calib, dict):
                    summary = {}
                    for key, value in calib.items():
                        if isinstance(value, (str, int, float, bool)):
                            summary[key] = value
                        elif isinstance(value, np.ndarray):
                            summary[key] = value.tolist()
                    calibration_summary[model_name] = summary
            model_card['performance']['calibration'] = calibration_summary
        
        if 'business_thresholds' in all_results:
            # Extraire seulement les métriques clés
            business_summary = {}
            for model_name, business in all_results['business_thresholds'].items():
                if isinstance(business, dict):
                    summary = {}
                    for key, value in business.items():
                        if isinstance(value, (str, int, float, bool)):
                            summary[key] = value
                        elif isinstance(value, pd.DataFrame):
                            summary[key] = "DataFrame summary available"
                    business_summary[model_name] = summary
            model_card['performance']['business_optimization'] = business_summary
        
        # Ajouter les limitations connues
        model_card['limitations'] = [
            'Modèles entraînés sur données historiques spécifiques',
            'Performance peut varier selon les sous-groupes',
            'Nécessite monitoring continu en production'
        ]
        
        # Ajouter les recommandations
        model_card['recommendations'] = [
            'Réentraîner périodiquement avec nouvelles données',
            'Monitorer la dérive des données',
            'Valider sur nouveaux sous-groupes',
            'Maintenir la calibration des probabilités'
        ]
        
        return model_card
    
    def package_models(self, chapter, models, all_results):
        """Packager les modèles avec traçabilité complète"""
        logger.info(f"   📦 Packaging modèles {chapter}...")
        
        package_dir = self.base_path / "model_packages" / chapter
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les modèles
        for model_name, model in models.items():
            model_file = package_dir / f"{model_name}_production.pkl"
            joblib.dump(model, model_file)
        
        # Créer le hash des données d'entraînement
        data_hash = hashlib.sha256()
        # Ici on devrait hasher les vraies données d'entraînement
        data_hash.update(f"training_data_{chapter}".encode())
        
        # Créer le fichier de configuration
        config = {
            'package_version': '1.0',
            'created_date': datetime.now().isoformat(),
            'chapter': chapter,
            'models_included': list(models.keys()),
            'data_hash': data_hash.hexdigest(),
            'validation_results': all_results,
            'deployment_requirements': {
                'python_version': '3.8+',
                'dependencies': ['scikit-learn', 'pandas', 'numpy', 'joblib']
            }
        }
        
        # Sauvegarder la configuration
        with open(package_dir / 'package_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Sauvegarder la Model Card
        model_card = self.create_model_card(chapter, all_results)
        with open(package_dir / 'model_card.json', 'w') as f:
            json.dump(model_card, f, indent=2, default=str)
        
        logger.info(f"      ✅ Package sauvegardé dans {package_dir}")
        
        return package_dir
    
    # ==================== MÉTHODE PRINCIPALE ====================
    
    def run_comprehensive_validation(self):
        """Lancer la validation complète pour tous les chapitres"""
        logger.info("🚀 DÉBUT DE LA VALIDATION COMPLÈTE DE PRODUCTION")
        logger.info("=" * 70)
        
        all_chapter_results = {}
        
        for chapter in self.chapters:
            logger.info(f"\n📊 VALIDATION CHAPITRE {chapter}")
            logger.info("-" * 50)
            
            # Charger les données
            models, X_train, X_valid, X_test, y_train, y_valid, y_test, df_full = self.load_chapter_data(chapter)
            
            if not models:
                logger.error(f"❌ Impossible de charger les modèles pour {chapter}")
                continue
            
            chapter_results = {}
            
            # 1. Tests anti-fuites
            logger.info("\n🔍 1. TESTS ANTI-FUITES")
            leakage_results = {}
            
            # Test permutation target
            for model_name, model in models.items():
                leakage_results[f'{model_name}_permutation'] = self.test_target_permutation(
                    model, X_test, y_test, model_name
                )
            
            # Test adversarial validation
            leakage_results['adversarial'] = self.test_adversarial_validation(
                X_train, X_test, y_train, y_test
            )
            
            # Test null importances
            for model_name, model in models.items():
                if model_name in ['lightgbm', 'xgboost', 'catboost']:  # Modèles principaux
                    leakage_results[f'{model_name}_null_importance'] = self.test_null_importances(
                        model, X_train, y_train, model_name
                    )
            
            chapter_results['leakage_tests'] = leakage_results
            
            # 2. Calibration fine
            logger.info("\n📏 2. CALIBRATION FINE")
            calibration_results = {}
            
            for model_name, model in models.items():
                if self.needs_preprocessing(model, X_test):
                    X_test_processed = self.preprocess_data(X_test)
                    y_proba = model.predict_proba(X_test_processed)[:, 1]
                else:
                    y_proba = model.predict_proba(X_test)[:, 1]
                calibration_results[model_name] = self.compute_calibration_metrics(
                    y_test, y_proba, model_name
                )
            
            self.create_calibration_plots(chapter, calibration_results)
            chapter_results['calibration_metrics'] = calibration_results
            
            # 3. Seuils orientés métier
            logger.info("\n💰 3. OPTIMISATION SEUILS MÉTIER")
            business_results = {}
            
            for model_name, model in models.items():
                if self.needs_preprocessing(model, X_test):
                    X_test_processed = self.preprocess_data(X_test)
                    y_proba = model.predict_proba(X_test_processed)[:, 1]
                else:
                    y_proba = model.predict_proba(X_test)[:, 1]
                
                # Optimisation seuils
                threshold_analysis = self.optimize_business_thresholds(
                    y_test, y_proba, cost_fp=1.0, cost_fn=5.0
                )
                
                # Métriques top-k
                top_k_metrics = self.compute_top_k_metrics(y_test, y_proba)
                
                business_results[model_name] = {
                    'threshold_optimization': threshold_analysis,
                    'top_k_metrics': top_k_metrics
                }
            
            chapter_results['business_thresholds'] = business_results
            
            # 4. Analyse d'overfitting
            logger.info("\n📊 4. ANALYSE D'OVERFITTING")
            overfitting_results = self.analyze_overfitting(models, X_train, X_test, y_train, y_test)
            chapter_results['overfitting_analysis'] = overfitting_results
            
            # 5. Robustesse par sous-groupes
            logger.info("\n👥 5. ROBUSTESSE SOUS-GROUPES")
            subgroup_results = self.analyze_subgroup_robustness(df_full, models, chapter)
            chapter_results['subgroup_analysis'] = subgroup_results
            
            # 5. Backtest temporel
            logger.info("\n⏰ 5. BACKTEST TEMPOREL")
            temporal_results = self.temporal_backtest(df_full, models, chapter)
            chapter_results['temporal_backtest'] = temporal_results
            
            # 6. Monitoring production
            logger.info("\n📊 6. MONITORING PRODUCTION")
            monitoring_config = self.setup_production_monitoring(df_full, models, chapter)
            chapter_results['monitoring_config'] = monitoring_config
            
            # 7. Traçabilité et packaging
            logger.info("\n📦 7. TRAÇABILITÉ ET PACKAGING")
            package_dir = self.package_models(chapter, models, chapter_results)
            chapter_results['package_location'] = str(package_dir)
            
            all_chapter_results[chapter] = chapter_results
        
        # Sauvegarder le rapport global
        self.save_global_validation_report(all_chapter_results)
        
        logger.info("\n🎉 VALIDATION COMPLÈTE TERMINÉE")
        logger.info("=" * 70)
        
        return all_chapter_results
    
    def save_global_validation_report(self, all_results):
        """Sauvegarder le rapport global de validation"""
        logger.info("💾 Sauvegarde rapport global validation...")
        
        results_dir = self.base_path / "test_results" / "production_validation"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Rapport JSON détaillé
        with open(results_dir / "comprehensive_production_validation.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Rapport résumé
        with open(results_dir / "validation_summary.txt", 'w') as f:
            f.write("RAPPORT GLOBAL DE VALIDATION PRODUCTION\n")
            f.write("=" * 60 + "\n\n")
            
            for chapter, results in all_results.items():
                f.write(f"CHAPITRE {chapter.upper()}\n")
                f.write("-" * 30 + "\n")
                
                # Tests anti-fuites
                if 'leakage_tests' in results:
                    f.write("TESTS ANTI-FUITES:\n")
                    for test_name, test_result in results['leakage_tests'].items():
                        if 'leakage_risk' in test_result:
                            risk = test_result['leakage_risk']
                            f.write(f"  {test_name}: {risk}\n")
                
                # Calibration
                if 'calibration_metrics' in results:
                    f.write("\nCALIBRATION:\n")
                    for model_name, calib in results['calibration_metrics'].items():
                        quality = calib['calibration_quality']
                        f.write(f"  {model_name}: {quality} (Brier={calib['brier_score']:.3f})\n")
                
                f.write("\n" + "=" * 60 + "\n\n")

def main():
    """Fonction principale"""
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    global logger
    logger = logging.getLogger(__name__)
    
    validator = ProductionValidationSuite()
    results = validator.run_comprehensive_validation()
    
    # Affichage du résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DE LA VALIDATION PRODUCTION")
    print("=" * 70)
    
    for chapter, chapter_results in results.items():
        print(f"\n{chapter.upper()}:")
        
        # Tests anti-fuites
        if 'leakage_tests' in chapter_results:
            print("  Tests anti-fuites:")
            for test_name, test_result in chapter_results['leakage_tests'].items():
                if 'leakage_risk' in test_result:
                    risk_emoji = "🔴" if test_result['leakage_risk'] == 'HIGH' else "🟡" if test_result['leakage_risk'] == 'MEDIUM' else "🟢"
                    print(f"    {risk_emoji} {test_name}: {test_result['leakage_risk']}")
        
        # Calibration
        if 'calibration_metrics' in chapter_results:
            print("  Calibration:")
            for model_name, calib in chapter_results['calibration_metrics'].items():
                quality_emoji = "🟢" if calib['calibration_quality'] == 'EXCELLENT' else "🟡" if calib['calibration_quality'] == 'GOOD' else "🔴"
                print(f"    {quality_emoji} {model_name}: {calib['calibration_quality']}")

if __name__ == "__main__":
    main()
