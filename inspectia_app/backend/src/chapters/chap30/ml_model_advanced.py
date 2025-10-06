#!/usr/bin/env python3
"""
Modèle ML Supervisé pour la Détection de Fraude - Chapitre 30 AVANCÉ
Utilise les nouvelles données préprocessées avec techniques avancées de détection de fraude
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from datetime import datetime
import warnings
import yaml
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, f1_score, precision_score, recall_score,
    average_precision_score, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Gradient Boosting
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP for feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP non disponible - l'analyse SHAP sera ignorée")

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Chap30MLAdvanced:
    """Modèle ML Supervisé pour la Détection de Fraude - Chapitre 30 AVANCÉ"""
    
    def __init__(self):
        # Chemins des données
        self.backend_root = Path(__file__).resolve().parents[3]
        self.data_path = self.backend_root / "data/processed/CHAP30_PROCESSED_ADVANCED.csv"
        self.splits_path = self.backend_root / "data/ml_splits/chap30"
        self.results_path = self.backend_root / "results/chap30"
        self.models_path = self.results_path / "models"
        
        # Créer les dossiers nécessaires
        self.splits_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Configuration des modèles
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'CatBoost': cb.CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            )
        }
        
        # Colonnes à utiliser pour l'entraînement
        self.feature_columns = self._get_feature_columns()
        
        logger.info("✅ Modèle ML Chapitre 30 AVANCÉ initialisé")

    def _get_feature_columns(self):
        """Définir les colonnes à utiliser comme features"""
        # Colonnes numériques de base
        numeric_features = [
            'VALEUR_CAF', 'VALEUR_DOUANE', 'MONTANT_LIQUIDATION', 'POIDS_NET',
            'VALEUR_UNITAIRE_KG', 'TAUX_DROITS_PERCENT', 'RATIO_DOUANE_CAF',
            'NUMERO_ARTICLE', 'PRECISION_UEMOA'
        ]
        
        # Colonnes des techniques avancées de détection de fraude
        fraud_detection_features = [
            'BIENAYME_CHEBYCHEV_SCORE', 'TEI_CALCULE', 'MIRROR_TEI_SCORE', 
            'MIRROR_TEI_DEVIATION', 'SPECTRAL_CLUSTER_SCORE', 
            'HIERARCHICAL_CLUSTER_SCORE', 'ADMIN_VALUES_SCORE', 
            'ADMIN_VALUES_DEVIATION', 'COMPOSITE_FRAUD_SCORE', 'RATIO_POIDS_VALEUR'
        ]
        
        # Colonnes business (toutes les features BUSINESS_)
        business_features = [
            'BUSINESS_GLISSEMENT_COSMETIQUE', 'BUSINESS_GLISSEMENT_PAYS_COSMETIQUES',
            'BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'BUSINESS_RISK_PAYS_HIGH',
            'BUSINESS_ORIGINE_DIFF_PROVENANCE', 'BUSINESS_REGIME_PREFERENTIEL',
            'BUSINESS_REGIME_NORMAL', 'BUSINESS_VALEUR_ELEVEE',
            'BUSINESS_VALEUR_EXCEPTIONNELLE', 'BUSINESS_POIDS_ELEVE',
            'BUSINESS_DROITS_ELEVES', 'BUSINESS_RATIO_LIQUIDATION_CAF',
            'BUSINESS_RATIO_DOUANE_CAF', 'BUSINESS_IS_MEDICAMENT',
            'BUSINESS_IS_ANTIPALUDEEN', 'BUSINESS_IS_PRECISION_UEMOA',
            'BUSINESS_ARTICLES_MULTIPLES', 'BUSINESS_AVEC_DPI'
        ]
        
        # Colonnes catégorielles
        categorical_features = [
            'CODE_PRODUIT_STR', 'PAYS_ORIGINE_STR', 'PAYS_PROVENANCE_STR',
            'BUREAU', 'REGIME_FISCAL', 'NUMERO_DPI'
        ]
        
        return {
            'numeric': numeric_features,
            'fraud_detection': fraud_detection_features,
            'business': business_features,
            'categorical': categorical_features
        }

    def load_data(self):
        """Charger les données préprocessées"""
        logger.info("📊 Chargement des données préprocessées...")
        
        df = pd.read_csv(self.data_path)
        logger.info(f"✅ Données chargées: {df.shape}")
        
        # Vérifier la présence du target
        if 'FRAUD_FLAG' not in df.columns:
            raise ValueError("FRAUD_FLAG non trouvé dans les données")
        
        # Afficher la distribution du target
        fraud_distribution = df['FRAUD_FLAG'].value_counts()
        logger.info(f"📊 Distribution FRAUD_FLAG: {fraud_distribution.to_dict()}")
        logger.info(f"📊 Taux de fraude: {df['FRAUD_FLAG'].mean()*100:.2f}%")
        
        return df

    def prepare_features(self, df):
        """Préparer les features pour l'entraînement"""
        logger.info("🔧 Préparation des features...")
        
        # Toutes les colonnes de features
        all_features = []
        for feature_type, features in self.feature_columns.items():
            all_features.extend(features)
        
        # Vérifier que toutes les colonnes existent
        missing_cols = [col for col in all_features if col not in df.columns]
        if missing_cols:
            logger.warning(f"⚠️ Colonnes manquantes: {missing_cols}")
            all_features = [col for col in all_features if col in df.columns]
        
        # Sélectionner les features et le target
        X = df[all_features].copy()
        y = df['FRAUD_FLAG'].copy()
        
        logger.info(f"✅ Features préparées: {X.shape[1]} features, {X.shape[0]} échantillons")
        logger.info(f"✅ Target préparé: {y.shape[0]} échantillons")
        
        return X, y

    def create_train_test_splits(self, X, y):
        """Créer les splits train/test/validation"""
        logger.info("📊 Création des splits train/test/validation...")
        
        # Split initial: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Split du train: 80% train, 20% validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Sauvegarder les splits
        splits = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        for name, data in splits.items():
            data.to_csv(self.splits_path / f"{name}.csv", index=False)
        
        logger.info(f"✅ Splits créés et sauvegardés dans {self.splits_path}")
        logger.info(f"   Train: {X_train.shape[0]} échantillons")
        logger.info(f"   Validation: {X_val.shape[0]} échantillons")
        logger.info(f"   Test: {X_test.shape[0]} échantillons")
        
        return splits

    def create_preprocessing_pipeline(self, X):
        """Créer le pipeline de preprocessing avec TOUTES les features importantes"""
        logger.info("🔧 Création du pipeline de preprocessing...")
        
        # Toutes les features numériques importantes (inclure toutes les features disponibles)
        numeric_features = [
            # Features numériques de base
            'VALEUR_CAF', 'VALEUR_DOUANE', 'MONTANT_LIQUIDATION', 'POIDS_NET', 'POIDS_NET_KG',
            'VALEUR_UNITAIRE_KG', 'TAUX_DROITS_PERCENT', 'RATIO_DOUANE_CAF',
            'NUMERO_ARTICLE', 'PRECISION_UEMOA', 'NOMBRE_COLIS', 'QUANTITE_COMPLEMENT',
            'VALEUR_UNITAIRE_PAR_KG', 'VALEUR_FOB', 'VALEUR_PAR_COLIS', 'POIDS_BRUT',
            'ASSURANCE', 'FRET', 'TAUX', 'MONTANT', 'BASE_TAXABLE', 'NOMBRE_CONTENEUR',
            
            # Features de détection de fraude avancée
            'BIENAYME_CHEBYCHEV_SCORE', 'TEI_CALCULE', 'MIRROR_TEI_SCORE',
            'MIRROR_TEI_DEVIATION', 'SPECTRAL_CLUSTER_SCORE', 'HIERARCHICAL_CLUSTER_SCORE',
            'ADMIN_VALUES_SCORE', 'ADMIN_VALUES_DEVIATION', 'COMPOSITE_FRAUD_SCORE', 'RATIO_POIDS_VALEUR',
            
            # Features business (toutes les features BUSINESS_)
            'BUSINESS_GLISSEMENT_COSMETIQUE', 'BUSINESS_GLISSEMENT_PAYS_COSMETIQUES',
            'BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'BUSINESS_RISK_PAYS_HIGH',
            'BUSINESS_ORIGINE_DIFF_PROVENANCE', 'BUSINESS_REGIME_PREFERENTIEL',
            'BUSINESS_REGIME_NORMAL', 'BUSINESS_VALEUR_ELEVEE', 'BUSINESS_VALEUR_EXCEPTIONNELLE',
            'BUSINESS_POIDS_ELEVE', 'BUSINESS_DROITS_ELEVES', 'BUSINESS_RATIO_LIQUIDATION_CAF',
            'BUSINESS_RATIO_DOUANE_CAF', 'BUSINESS_IS_MEDICAMENT', 'BUSINESS_IS_ANTIPALUDEEN',
            'BUSINESS_IS_PRECISION_UEMOA', 'BUSINESS_ARTICLES_MULTIPLES', 'BUSINESS_AVEC_DPI'
        ]
        
        # Features catégorielles importantes
        categorical_features = [
            'CODE_PRODUIT_STR', 'PAYS_ORIGINE_STR', 'PAYS_PROVENANCE_STR', 'BUREAU',
            'REGIME_FISCAL', 'NUMERO_ARTICLE_STR', 'PRECISION_UEMOA_STR', 'DATE_DECLARATION_STR',
            'CODE_SH', 'LIBELLE_TARIF', 'DESCRIPTION_COMMERCIALE', 'CATEGORIE_PRODUIT',
            'ALERTE_MOTS_CLES', 'DESTINATION', 'BUREAU_FRONTIERE', 'TYPE_REGIME',
            'REGIME_DOUANIER', 'REGIME_FISCAL_CODE', 'STATUT_BAE', 'CODE_TAXE',
            'LIBELLE_TAXE', 'NOM_NAVIRE', 'DATE_ARRIVEE', 'DATE_EMBARQUEMENT'
        ]
        
        # Filtrer les features qui existent dans les données
        numeric_features = [col for col in numeric_features if col in X.columns]
        categorical_features = [col for col in categorical_features if col in X.columns]
        
        # Pipeline de preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        logger.info(f"✅ Pipeline créé: {len(numeric_features)} numériques, {len(categorical_features)} catégorielles")
        logger.info(f"   Features numériques: {numeric_features[:10]}..." if len(numeric_features) > 10 else f"   Features numériques: {numeric_features}")
        logger.info(f"   Features catégorielles: {categorical_features[:5]}..." if len(categorical_features) > 5 else f"   Features catégorielles: {categorical_features}")
        
        return preprocessor

    def train_models(self, splits, preprocessor):
        """Entraîner tous les modèles en respectant la convention train/val/test"""
        logger.info("🚀 Début de l'entraînement des modèles...")
        logger.info("📋 Convention respectée: fit sur train, val pour sélection, test pour évaluation finale")
        
        results = {}
        trained_models = {}
        validation_results = {}
        
        for name, model in self.models.items():
            logger.info(f"📊 Entraînement {name}...")
            
            # Créer le pipeline complet
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            
            # ÉTAPE 1: Fit uniquement sur X_train (convention respectée)
            logger.info(f"   🔧 Fit sur X_train uniquement...")
            pipeline.fit(splits['X_train'], splits['y_train'])
            
            # ÉTAPE 2: Évaluation sur X_val pour la sélection d'hyperparamètres
            logger.info(f"   📊 Évaluation sur X_val pour sélection...")
            y_val_pred = pipeline.predict(splits['X_val'])
            y_val_pred_proba = pipeline.predict_proba(splits['X_val'])[:, 1]
            
            # Métriques de validation
            val_metrics = self._calculate_metrics(splits['y_val'], y_val_pred, y_val_pred_proba)
            validation_results[name] = val_metrics
            
            logger.info(f"   📈 {name} - Validation AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # ÉTAPE 3: Test final sur X_test (une seule fois, après sélection)
            logger.info(f"   🎯 Test final sur X_test...")
            y_test_pred = pipeline.predict(splits['X_test'])
            y_test_pred_proba = pipeline.predict_proba(splits['X_test'])[:, 1]
            
            # Métriques de test
            test_metrics = self._calculate_metrics(splits['y_test'], y_test_pred, y_test_pred_proba)
            results[name] = test_metrics
            
            logger.info(f"   🏆 {name} - Test AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}")
            
            trained_models[name] = pipeline
        
        # Sauvegarder les modèles
        for name, model in trained_models.items():
            joblib.dump(model, self.models_path / f"{name.lower()}_model.pkl")
        
        logger.info(f"✅ Tous les modèles entraînés et sauvegardés dans {self.models_path}")
        logger.info(f"📊 Résultats de validation: {len(validation_results)} modèles")
        logger.info(f"📊 Résultats de test: {len(results)} modèles")
        
        return results, trained_models, validation_results

    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculer les métriques de performance"""
        return {
            'accuracy': (y_pred == y_true).mean(),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_pred_proba),
            'avg_precision': average_precision_score(y_true, y_pred_proba)
        }

    def find_best_model(self, validation_results):
        """Trouver le meilleur modèle basé sur le F1-Score de validation (convention respectée)"""
        best_model = max(validation_results.items(), key=lambda x: x[1]['f1'])
        logger.info(f"🏆 Meilleur modèle sélectionné sur validation: {best_model[0]} (Validation F1: {best_model[1]['f1']:.4f}, AUC: {best_model[1]['auc']:.4f})")
        return best_model[0]

    def generate_comprehensive_results(self, results, trained_models, splits, best_model_name):
        """Générer tous les résultats et graphiques"""
        logger.info("📊 Génération des résultats complets...")
        
        # 1. Métriques de comparaison
        self._plot_metrics_comparison(results, best_model_name)
        
        # 2. Matrices de confusion
        self._plot_confusion_matrices(trained_models, splits, best_model_name)
        
        # 3. Courbes ROC
        self._plot_roc_curves(trained_models, splits, best_model_name)
        
        # 4. Courbes Precision-Recall
        self._plot_precision_recall_curves(trained_models, splits, best_model_name)
        
        # 5. SHAP pour le meilleur modèle
        self._generate_shap_analysis(trained_models[best_model_name], splits)
        
        # 6. Rapport YAML
        self._generate_yaml_report(results, best_model_name)
        
        logger.info("✅ Tous les résultats générés avec succès")

    def _plot_metrics_comparison(self, results, best_model_name):
        """Générer les graphiques de comparaison des métriques"""
        logger.info("📊 Génération des graphiques de métriques...")
        
        # Préparer les données
        metrics_df = pd.DataFrame(results).T
        
        # Graphique de comparaison des métriques - AMÉLIORÉ
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Comparaison des Métriques - Chapitre 30 (Produits Pharmaceutiques)', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'avg_precision']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Average Precision']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
            ax = axes[i//3, i%3]
            
            # Créer les barres avec des couleurs différentes
            bars = ax.bar(range(len(metrics_df)), metrics_df[metric], 
                         color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.05)
            
            # Personnaliser les labels des axes
            ax.set_xticks(range(len(metrics_df)))
            ax.set_xticklabels(metrics_df.index, rotation=45, ha='right', fontsize=10)
            
            # Ajouter les valeurs sur les barres avec un meilleur positionnement
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{height:.3f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=9)
            
            # Améliorer la grille
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.results_path / 'metrics_comparison_all_algorithms.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Graphique du meilleur modèle - AMÉLIORÉ
        best_model = metrics_df.loc[best_model_name]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Créer un graphique en barres horizontales pour une meilleure lisibilité
        y_pos = np.arange(len(metrics))
        bars = ax.barh(y_pos, [best_model[metric] for metric in metrics], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(titles, fontsize=12, fontweight='bold')
        ax.set_xlabel('Score', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.set_title(f'Métriques du Meilleur Modèle - {best_model_name}\nChapitre 30 (Produits Pharmaceutiques)', 
                    fontweight='bold', fontsize=16, pad=20)
        
        # Ajouter les valeurs sur les barres
        for i, (bar, metric) in enumerate(zip(bars, metrics)):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                   f'{width:.3f}', ha='left', va='center', 
                   fontweight='bold', fontsize=11)
        
        # Améliorer la grille
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax.set_axisbelow(True)
        
        # Inverser l'ordre des barres pour avoir le meilleur score en haut
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'metrics_best.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("✅ Graphiques de métriques générés")

    def _plot_confusion_matrices(self, trained_models, splits, best_model_name):
        """Générer les matrices de confusion"""
        logger.info("📊 Génération des matrices de confusion...")
        
        # Toutes les matrices de confusion - AMÉLIORÉ
        n_models = len(trained_models)
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Matrices de Confusion - Chapitre 30 (Produits Pharmaceutiques)', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Couleurs pour chaque modèle
        model_colors = ['Blues', 'Greens', 'Oranges', 'Reds', 'Purples']
        
        for i, (name, model) in enumerate(trained_models.items()):
            ax = axes[i//3, i%3]
            
            y_pred = model.predict(splits['X_test'])
            cm = confusion_matrix(splits['y_test'], y_pred)
            
            # Calculer les pourcentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Créer l'annotation avec les valeurs et pourcentages
            annotations = []
            for j in range(cm.shape[0]):
                row = []
                for k in range(cm.shape[1]):
                    row.append(f'{cm[j, k]}\n({cm_percent[j, k]:.1f}%)')
                annotations.append(row)
            
            sns.heatmap(cm, annot=annotations, fmt='', cmap=model_colors[i % len(model_colors)], 
                       ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title(f'{name}', fontweight='bold', fontsize=14, pad=15)
            ax.set_xlabel('Prédiction', fontsize=12, fontweight='bold')
            ax.set_ylabel('Vraie Valeur', fontsize=12, fontweight='bold')
            
            # Personnaliser les labels
            ax.set_xticklabels(['Conforme', 'Fraude'], fontsize=10)
            ax.set_yticklabels(['Conforme', 'Fraude'], fontsize=10)
        
        # Masquer le dernier subplot s'il n'est pas utilisé
        if n_models < 6:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(self.results_path / 'confusion_matrices_all.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Matrice de confusion du meilleur modèle (basé sur F1 validation)
        best_model = trained_models[best_model_name]
        
        y_pred = best_model.predict(splits['X_test'])
        cm = confusion_matrix(splits['y_test'], y_pred)
        
        # Calculer les métriques détaillées
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        plt.figure(figsize=(10, 8))
        
        # Créer l'annotation avec les métriques
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        annotations = []
        for j in range(cm.shape[0]):
            row = []
            for k in range(cm.shape[1]):
                row.append(f'{cm[j, k]}\n({cm_percent[j, k]:.1f}%)')
            annotations.append(row)
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='RdYlBu_r', 
                   cbar_kws={'shrink': 0.8})
        
        plt.title(f'Matrice de Confusion - {best_model_name} (Meilleur Modèle)\n'
                 f'Chapitre 30 (Produits Pharmaceutiques)\n'
                 f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}', 
                 fontweight='bold', fontsize=14, pad=20)
        plt.xlabel('Prédiction', fontsize=12, fontweight='bold')
        plt.ylabel('Vraie Valeur', fontsize=12, fontweight='bold')
        
        # Personnaliser les labels
        plt.xticks([0.5, 1.5], ['Conforme', 'Fraude'], fontsize=11)
        plt.yticks([0.5, 1.5], ['Conforme', 'Fraude'], fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'confusion_matrix_best_algorithm.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("✅ Matrices de confusion générées")

    def _plot_roc_curves(self, trained_models, splits, best_model_name):
        """Générer les courbes ROC"""
        logger.info("📊 Génération des courbes ROC...")
        
        # Toutes les courbes ROC - AMÉLIORÉ
        plt.figure(figsize=(14, 10))
        
        # Couleurs et styles pour chaque modèle
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        linestyles = ['-', '--', '-.', ':', '-']
        
        for i, (name, model) in enumerate(trained_models.items()):
            y_pred_proba = model.predict_proba(splits['X_test'])[:, 1]
            fpr, tpr, _ = roc_curve(splits['y_test'], y_pred_proba)
            auc_score = roc_auc_score(splits['y_test'], y_pred_proba)
            
            plt.plot(fpr, tpr, 
                    color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=3, 
                    label=f'{name} (AUC = {auc_score:.4f})',
                    alpha=0.8)
        
        # Ligne de référence (classificateur aléatoire)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, label='Classificateur Aléatoire')
        
        plt.xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=14, fontweight='bold')
        plt.ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=14, fontweight='bold')
        plt.title('Courbes ROC - Chapitre 30 (Produits Pharmaceutiques)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Améliorer la légende
        plt.legend(loc='lower right', fontsize=12, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Personnaliser les axes
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'roc_curves_all.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Courbe ROC du meilleur modèle (basé sur F1 validation)
        best_model = trained_models[best_model_name]
        
        y_pred_proba = best_model.predict_proba(splits['X_test'])[:, 1]
        fpr, tpr, _ = roc_curve(splits['y_test'], y_pred_proba)
        auc_score = roc_auc_score(splits['y_test'], y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        
        # Courbe principale
        plt.plot(fpr, tpr, color='#FF6B6B', linewidth=4, 
                label=f'{best_model_name} (AUC = {auc_score:.4f})', alpha=0.8)
        
        # Ligne de référence
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, 
                label='Classificateur Aléatoire')
        
        # Zone sous la courbe
        plt.fill_between(fpr, tpr, alpha=0.2, color='#FF6B6B')
        
        plt.xlabel('Taux de Faux Positifs (1 - Spécificité)', fontsize=14, fontweight='bold')
        plt.ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=14, fontweight='bold')
        plt.title(f'Courbe ROC - {best_model_name} (Meilleur Modèle)\n'
                 f'Chapitre 30 (Produits Pharmaceutiques)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Améliorer la légende
        plt.legend(loc='lower right', fontsize=14, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Personnaliser les axes
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Ajouter des annotations
        plt.text(0.6, 0.2, f'AUC = {auc_score:.4f}', 
                fontsize=16, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'roc_curve_best_algorithm.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("✅ Courbes ROC générées")

    def _plot_precision_recall_curves(self, trained_models, splits, best_model_name):
        """Générer les courbes Precision-Recall"""
        logger.info("📊 Génération des courbes Precision-Recall...")
        
        # Toutes les courbes Precision-Recall - AMÉLIORÉ
        plt.figure(figsize=(14, 10))
        
        # Couleurs et styles pour chaque modèle
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        linestyles = ['-', '--', '-.', ':', '-']
        
        for i, (name, model) in enumerate(trained_models.items()):
            y_pred_proba = model.predict_proba(splits['X_test'])[:, 1]
            precision, recall, _ = precision_recall_curve(splits['y_test'], y_pred_proba)
            avg_precision = average_precision_score(splits['y_test'], y_pred_proba)
            
            plt.plot(recall, precision, 
                    color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=3, 
                    label=f'{name} (AP = {avg_precision:.4f})',
                    alpha=0.8)
        
        # Ligne de référence (baseline)
        baseline = splits['y_test'].mean()
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=2, alpha=0.6, 
                   label=f'Baseline (AP = {baseline:.4f})')
        
        plt.xlabel('Recall (Sensibilité)', fontsize=14, fontweight='bold')
        plt.ylabel('Precision (Précision)', fontsize=14, fontweight='bold')
        plt.title('Courbes Precision-Recall - Chapitre 30 (Produits Pharmaceutiques)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Améliorer la légende
        plt.legend(loc='lower left', fontsize=12, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Personnaliser les axes
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'precision_recall_curves_all.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Courbe Precision-Recall du meilleur modèle (basé sur F1 validation)
        best_model = trained_models[best_model_name]
        
        y_pred_proba = best_model.predict_proba(splits['X_test'])[:, 1]
        precision, recall, _ = precision_recall_curve(splits['y_test'], y_pred_proba)
        avg_precision = average_precision_score(splits['y_test'], y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        
        # Courbe principale
        plt.plot(recall, precision, color='#FF6B6B', linewidth=4, 
                label=f'{best_model_name} (AP = {avg_precision:.4f})', alpha=0.8)
        
        # Ligne de référence
        baseline = splits['y_test'].mean()
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=2, alpha=0.6, 
                   label=f'Baseline (AP = {baseline:.4f})')
        
        # Zone sous la courbe
        plt.fill_between(recall, precision, alpha=0.2, color='#FF6B6B')
        
        plt.xlabel('Recall (Sensibilité)', fontsize=14, fontweight='bold')
        plt.ylabel('Precision (Précision)', fontsize=14, fontweight='bold')
        plt.title(f'Courbe Precision-Recall - {best_model_name} (Meilleur Modèle)\n'
                 f'Chapitre 30 (Produits Pharmaceutiques)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Améliorer la légende
        plt.legend(loc='lower left', fontsize=14, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Personnaliser les axes
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Ajouter des annotations
        plt.text(0.6, 0.3, f'AP = {avg_precision:.4f}', 
                fontsize=16, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.results_path / 'precision_recall_curve_best_algorithm.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("✅ Courbes Precision-Recall générées")

    def _get_feature_names_after_preprocessing(self, model, X_sample):
        """Obtenir les noms des features après preprocessing"""
        preprocessor = model.named_steps['preprocessor']
        
        # Toutes les features numériques importantes (même liste que dans create_preprocessing_pipeline)
        numeric_features = [
            # Features numériques de base
            'VALEUR_CAF', 'VALEUR_DOUANE', 'MONTANT_LIQUIDATION', 'POIDS_NET', 'POIDS_NET_KG',
            'VALEUR_UNITAIRE_KG', 'TAUX_DROITS_PERCENT', 'RATIO_DOUANE_CAF',
            'NUMERO_ARTICLE', 'PRECISION_UEMOA', 'NOMBRE_COLIS', 'QUANTITE_COMPLEMENT',
            'VALEUR_UNITAIRE_PAR_KG', 'VALEUR_FOB', 'VALEUR_PAR_COLIS', 'POIDS_BRUT',
            'ASSURANCE', 'FRET', 'TAUX', 'MONTANT', 'BASE_TAXABLE', 'NOMBRE_CONTENEUR',
            
            # Features de détection de fraude avancée
            'BIENAYME_CHEBYCHEV_SCORE', 'TEI_CALCULE', 'MIRROR_TEI_SCORE',
            'MIRROR_TEI_DEVIATION', 'SPECTRAL_CLUSTER_SCORE', 'HIERARCHICAL_CLUSTER_SCORE',
            'ADMIN_VALUES_SCORE', 'ADMIN_VALUES_DEVIATION', 'COMPOSITE_FRAUD_SCORE', 'RATIO_POIDS_VALEUR',
            
            # Features business (toutes les features BUSINESS_)
            'BUSINESS_GLISSEMENT_COSMETIQUE', 'BUSINESS_GLISSEMENT_PAYS_COSMETIQUES',
            'BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'BUSINESS_RISK_PAYS_HIGH',
            'BUSINESS_ORIGINE_DIFF_PROVENANCE', 'BUSINESS_REGIME_PREFERENTIEL',
            'BUSINESS_REGIME_NORMAL', 'BUSINESS_VALEUR_ELEVEE', 'BUSINESS_VALEUR_EXCEPTIONNELLE',
            'BUSINESS_POIDS_ELEVE', 'BUSINESS_DROITS_ELEVES', 'BUSINESS_RATIO_LIQUIDATION_CAF',
            'BUSINESS_RATIO_DOUANE_CAF', 'BUSINESS_IS_MEDICAMENT', 'BUSINESS_IS_ANTIPALUDEEN',
            'BUSINESS_IS_PRECISION_UEMOA', 'BUSINESS_ARTICLES_MULTIPLES', 'BUSINESS_AVEC_DPI'
        ]
        
        # Features catégorielles importantes
        categorical_features = [
            'CODE_PRODUIT_STR', 'PAYS_ORIGINE_STR', 'PAYS_PROVENANCE_STR', 'BUREAU',
            'REGIME_FISCAL', 'NUMERO_ARTICLE_STR', 'PRECISION_UEMOA_STR', 'DATE_DECLARATION_STR',
            'CODE_SH', 'LIBELLE_TARIF', 'DESCRIPTION_COMMERCIALE', 'CATEGORIE_PRODUIT',
            'ALERTE_MOTS_CLES', 'DESTINATION', 'BUREAU_FRONTIERE', 'TYPE_REGIME',
            'REGIME_DOUANIER', 'REGIME_FISCAL_CODE', 'STATUT_BAE', 'CODE_TAXE',
            'LIBELLE_TAXE', 'NOM_NAVIRE', 'DATE_ARRIVEE', 'DATE_EMBARQUEMENT'
        ]
        
        # Filtrer les features qui existent dans les données
        numeric_features = [col for col in numeric_features if col in X_sample.columns]
        categorical_features = [col for col in categorical_features if col in X_sample.columns]
        
        # Obtenir les noms des features catégorielles après OneHotEncoder
        categorical_transformer = preprocessor.named_transformers_['cat']
        if hasattr(categorical_transformer.named_steps['onehot'], 'get_feature_names_out'):
            cat_feature_names = categorical_transformer.named_steps['onehot'].get_feature_names_out(categorical_features)
        else:
            # Fallback si get_feature_names_out n'est pas disponible
            cat_feature_names = [f"{col}_{i}" for col in categorical_features for i in range(10)]  # Approximation
        
        # Combiner tous les noms de features
        all_feature_names = numeric_features + list(cat_feature_names)
        
        return all_feature_names

    def _generate_shap_analysis(self, best_model, splits):
        """Générer l'analyse SHAP pour le meilleur modèle"""
        logger.info("📊 Génération de l'analyse SHAP...")
        
        if not SHAP_AVAILABLE:
            logger.warning("⚠️ SHAP non disponible - analyse SHAP ignorée")
            return
        
        try:
            # Préparer les données pour SHAP
            X_test_processed = best_model.named_steps['preprocessor'].transform(splits['X_test'])
            
            # Obtenir les noms des features après preprocessing
            feature_names = self._get_feature_names_after_preprocessing(best_model, splits['X_test'])
            
            # Créer un explainer SHAP
            explainer = shap.TreeExplainer(best_model.named_steps['classifier'])
            shap_values = explainer.shap_values(X_test_processed)
            
            # Si le modèle est binaire, prendre les valeurs pour la classe positive
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Créer un DataFrame avec les noms des features
            X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
            
            # Graphique d'importance des features
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_test_df, max_display=20, show=False)
            plt.title('SHAP Feature Importance - Top 20 Features (Chapitre 30)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.results_path / 'shap_summary_plot_20.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Graphique des valeurs SHAP
            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_test_df, plot_type="bar", max_display=20, show=False)
            plt.title('SHAP Feature Values - Top 20 Features (Chapitre 30)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.results_path / 'shap_feature_importance_20.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("✅ Analyse SHAP générée")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur lors de la génération SHAP: {e}")

    def _generate_yaml_report(self, results, best_model_name):
        """Générer le rapport YAML"""
        logger.info("📊 Génération du rapport YAML...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'chapter': '30',
            'data_source': str(self.data_path),
            'best_model': best_model_name,
            'models_performance': results,
            'summary': {
                'total_models': len(results),
                'best_auc': max(results[model]['auc'] for model in results),
                'best_f1': max(results[model]['f1'] for model in results),
                'best_accuracy': max(results[model]['accuracy'] for model in results)
            }
        }
        
        with open(self.results_path / 'ml_supervised_report.yaml', 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info("✅ Rapport YAML généré")

    def run_complete_ml_pipeline(self):
        """Exécuter le pipeline ML complet"""
        logger.info("🚀 DÉMARRAGE DU PIPELINE ML COMPLET - CHAPITRE 30")
        logger.info("=" * 60)
        
        try:
            # 1. Charger les données
            df = self.load_data()
            
            # 2. Préparer les features
            X, y = self.prepare_features(df)
            
            # 3. Créer les splits
            splits = self.create_train_test_splits(X, y)
            
            # 4. Créer le pipeline de preprocessing
            preprocessor = self.create_preprocessing_pipeline(X)
            
            # 5. Entraîner les modèles (convention train/val/test respectée)
            results, trained_models, validation_results = self.train_models(splits, preprocessor)
            
            # 6. Trouver le meilleur modèle basé sur la validation
            best_model_name = self.find_best_model(validation_results)
            
            # 7. Générer tous les résultats
            self.generate_comprehensive_results(results, trained_models, splits, best_model_name)
            
            logger.info("=" * 60)
            logger.info("✅ PIPELINE ML COMPLET TERMINÉ AVEC SUCCÈS")
            logger.info(f"🏆 Meilleur modèle: {best_model_name}")
            logger.info(f"📁 Résultats sauvegardés dans: {self.results_path}")
            logger.info(f"📁 Modèles sauvegardés dans: {self.models_path}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du pipeline ML: {e}")
            raise

def main():
    """Fonction principale"""
    try:
        ml_pipeline = Chap30MLAdvanced()
        ml_pipeline.run_complete_ml_pipeline()
    except Exception as e:
        logger.error(f"❌ Erreur dans main: {e}")
        raise

if __name__ == "__main__":
    main()
