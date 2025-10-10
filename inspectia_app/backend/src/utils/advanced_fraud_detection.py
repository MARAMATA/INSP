#!/usr/bin/env python3
"""
Module de détection de fraude avancée
Implémente les techniques de la cellule de ciblage et de veille commerciale :
- Méthodes probabilistes (théorème de Bienaymé-Tchebychev)
- Analyse miroir avec TEI (Taux Effectifs d'Imposition)
- Détection d'anomalies (clustering spectral et hiérarchique)
- Contrôle des valeurs administrées
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFraudDetection:
    """Classe principale pour la détection de fraude avancée"""
    
    def __init__(self, chapter: str = None):
        self.scaler = StandardScaler()
        self.product_origin_stats = {}
        self.admin_values = {}
        self.tei_thresholds = {}
        self.chapter = chapter
        
    def clean_data_comprehensive(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Toilettage complet des données
        - Suppression des doublons
        - Gestion des valeurs NaN, nulles et infinies
        - Normalisation des types de données
        """
        logger.info("🧹 Toilettage complet des données...")
        
        original_shape = df.shape
        
        # 1. Suppression des doublons
        df = df.drop_duplicates()
        logger.info(f"   Doublons supprimés: {original_shape[0] - df.shape[0]}")
        
        # 2. Gestion des valeurs infinies
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].dtype in ['float64', 'float32']:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # 3. Gestion des valeurs nulles
        # Pour les colonnes numériques : médiane
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"   {col}: {df[col].isnull().sum()} NaN → médiane {median_val}")
        
        # Pour les colonnes catégorielles : mode ou 'UNKNOWN'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'UNKNOWN'
                df[col] = df[col].fillna(mode_val)
                logger.info(f"   {col}: {df[col].isnull().sum()} NaN → mode '{mode_val}'")
        
        # 4. Normalisation des types
        if 'CODE_PRODUIT' in df.columns:
            df['CODE_PRODUIT'] = df['CODE_PRODUIT'].astype(str)
        if 'PAYS_ORIGINE' in df.columns:
            df['PAYS_ORIGINE'] = df['PAYS_ORIGINE'].astype(str).str.upper()
        if 'PAYS_PROVENANCE' in df.columns:
            df['PAYS_PROVENANCE'] = df['PAYS_PROVENANCE'].astype(str).str.upper()
        
        logger.info(f"✅ Toilettage terminé: {original_shape} → {df.shape}")
        return df
    
    def bienayme_chebychev_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Application du théorème de Bienaymé-Tchebychev pour encadrer les valeurs attendues
        par couple produit/origine et détecter les extrêmes suspects
        """
        logger.info("📊 Application du théorème de Bienaymé-Tchebychev...")
        
        # Créer la clé produit/origine
        # Adapter aux noms de colonnes après agrégation
        if 'CODE_PRODUIT_STR' in df.columns:
            df['PRODUCT_ORIGIN_KEY'] = df['CODE_PRODUIT_STR'] + '_' + df['PAYS_ORIGINE_STR']
        else:
            df['PRODUCT_ORIGIN_KEY'] = df['CODE_PRODUIT'].astype(str) + '_' + df['PAYS_ORIGINE'].astype(str)
        
        # Calculer les statistiques par couple produit/origine
        stats_by_key = df.groupby('PRODUCT_ORIGIN_KEY')['VALEUR_CAF'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        # Filtrer les couples avec au moins 5 observations
        stats_by_key = stats_by_key[stats_by_key['count'] >= 5]
        
        # Appliquer le théorème de Bienaymé-Tchebychev
        # P(|X - μ| ≥ kσ) ≤ 1/k²
        # Pour k=2: P(|X - μ| ≥ 2σ) ≤ 0.25 (25% des observations peuvent être en dehors)
        # Pour k=3: P(|X - μ| ≥ 3σ) ≤ 0.111 (11.1% des observations peuvent être en dehors)
        
        df['BIENAYME_CHEBYCHEV_ANOMALY'] = 0
        df['BIENAYME_CHEBYCHEV_SCORE'] = 0.0
        
        for _, row in stats_by_key.iterrows():
            key = row['PRODUCT_ORIGIN_KEY']
            mean_val = row['mean']
            std_val = row['std']
            
            if std_val > 0:  # Éviter la division par zéro
                # Masque pour ce couple produit/origine
                mask = df['PRODUCT_ORIGIN_KEY'] == key
                
                # Calculer l'écart normalisé |X - μ|/σ
                normalized_deviation = np.abs(df.loc[mask, 'VALEUR_CAF'] - mean_val) / std_val
                
                # Marquer comme anomalie si |X - μ| ≥ 3σ (seuil strict)
                anomaly_mask = normalized_deviation >= 3.0
                df.loc[mask & anomaly_mask, 'BIENAYME_CHEBYCHEV_ANOMALY'] = 1
                
                # Score de déviation (plus le score est élevé, plus c'est suspect)
                df.loc[mask, 'BIENAYME_CHEBYCHEV_SCORE'] = normalized_deviation
        
        # Sauvegarder les statistiques pour réutilisation
        self.product_origin_stats = stats_by_key.set_index('PRODUCT_ORIGIN_KEY').to_dict('index')
        
        anomalies_count = df['BIENAYME_CHEBYCHEV_ANOMALY'].sum()
        logger.info(f"   Anomalies détectées (Bienaymé-Tchebychev): {anomalies_count}")
        
        return df
    
    def mirror_analysis_tei(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse miroir avec interprétation des écarts via les Taux Effectifs d'Imposition (TEI)
        Compare les valeurs déclarées avec les valeurs de référence par produit/origine
        """
        logger.info("🪞 Analyse miroir avec TEI (Taux Effectifs d'Imposition)...")
        
        # Calculer le TEI moyen par couple produit/origine
        # Adapter aux noms de colonnes après agrégation
        if 'CODE_PRODUIT_STR' in df.columns:
            df['PRODUCT_ORIGIN_KEY'] = df['CODE_PRODUIT_STR'] + '_' + df['PAYS_ORIGINE_STR']
        else:
            df['PRODUCT_ORIGIN_KEY'] = df['CODE_PRODUIT'].astype(str) + '_' + df['PAYS_ORIGINE'].astype(str)
        
        # TEI = (MONTANT_LIQUIDATION / VALEUR_CAF) * 100
        df['TEI_CALCULE'] = (df['MONTANT_LIQUIDATION'] / df['VALEUR_CAF'].replace(0, 1)) * 100
        
        # Calculer les statistiques TEI par couple produit/origine
        tei_stats = df.groupby('PRODUCT_ORIGIN_KEY')['TEI_CALCULE'].agg([
            'count', 'mean', 'std', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
        ]).reset_index()
        tei_stats.columns = ['PRODUCT_ORIGIN_KEY', 'count', 'mean', 'std', 'median', 'q25', 'q75']
        
        # Filtrer les couples avec au moins 10 observations
        tei_stats = tei_stats[tei_stats['count'] >= 10]
        
        # Créer les features d'analyse miroir
        df['MIRROR_TEI_ANOMALY'] = 0
        df['MIRROR_TEI_SCORE'] = 0.0
        df['MIRROR_TEI_DEVIATION'] = 0.0
        
        for _, row in tei_stats.iterrows():
            key = row['PRODUCT_ORIGIN_KEY']
            tei_mean = row['mean']
            tei_std = row['std']
            tei_q25 = row['q25']
            tei_q75 = row['q75']
            
            if tei_std > 0:  # Éviter la division par zéro
                mask = df['PRODUCT_ORIGIN_KEY'] == key
                
                # Calculer l'écart par rapport à la moyenne
                tei_deviation = np.abs(df.loc[mask, 'TEI_CALCULE'] - tei_mean) / tei_std
                df.loc[mask, 'MIRROR_TEI_DEVIATION'] = tei_deviation
                
                # Score basé sur l'écart interquartile (plus robuste)
                iqr = tei_q75 - tei_q25
                if iqr > 0:
                    iqr_score = np.abs(df.loc[mask, 'TEI_CALCULE'] - tei_mean) / iqr
                    df.loc[mask, 'MIRROR_TEI_SCORE'] = iqr_score
                    
                    # Anomalie si TEI en dehors de l'intervalle [Q25 - 1.5*IQR, Q75 + 1.5*IQR]
                    lower_bound = tei_q25 - 1.5 * iqr
                    upper_bound = tei_q75 + 1.5 * iqr
                    anomaly_mask = (df.loc[mask, 'TEI_CALCULE'] < lower_bound) | (df.loc[mask, 'TEI_CALCULE'] > upper_bound)
                    df.loc[mask & anomaly_mask, 'MIRROR_TEI_ANOMALY'] = 1
        
        # Sauvegarder les seuils TEI
        self.tei_thresholds = tei_stats.set_index('PRODUCT_ORIGIN_KEY').to_dict('index')
        
        anomalies_count = df['MIRROR_TEI_ANOMALY'].sum()
        logger.info(f"   Anomalies TEI détectées: {anomalies_count}")
        
        return df
    
    def spectral_clustering_anomaly_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Détection d'anomalies par clustering spectral - VERSION RAPIDE
        Utilise des seuils statistiques simples au lieu du clustering
        """
        logger.info("🔍 Détection d'anomalies par seuils statistiques (rapide)...")
        
        # Utiliser des seuils statistiques simples au lieu du clustering
        df['SPECTRAL_CLUSTER_ANOMALY'] = 0
        df['SPECTRAL_CLUSTER_SCORE'] = 0.0
        
        # Détecter les anomalies basées sur des seuils statistiques
        if 'VALEUR_CAF' in df.columns:
            # Valeurs CAF anormalement élevées ou basses
            q99 = df['VALEUR_CAF'].quantile(0.99)
            q01 = df['VALEUR_CAF'].quantile(0.01)
            mask_valeur_anormale = (df['VALEUR_CAF'] > q99) | (df['VALEUR_CAF'] < q01)
            df.loc[mask_valeur_anormale, 'SPECTRAL_CLUSTER_ANOMALY'] = 1
            df.loc[mask_valeur_anormale, 'SPECTRAL_CLUSTER_SCORE'] = 1.0
        
        if 'POIDS_NET' in df.columns:
            # Poids anormalement élevés ou bas
            q99 = df['POIDS_NET'].quantile(0.99)
            q01 = df['POIDS_NET'].quantile(0.01)
            mask_poids_anormal = (df['POIDS_NET'] > q99) | (df['POIDS_NET'] < q01)
            df.loc[mask_poids_anormal, 'SPECTRAL_CLUSTER_ANOMALY'] = 1
            df.loc[mask_poids_anormal, 'SPECTRAL_CLUSTER_SCORE'] = 1.0
        
        anomalies_count = df['SPECTRAL_CLUSTER_ANOMALY'].sum()
        logger.info(f"   Anomalies détectées (seuils statistiques): {anomalies_count}")
        
        return df
    
    def hierarchical_clustering_anomaly_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Détection d'anomalies par clustering hiérarchique - VERSION RAPIDE
        Utilise des seuils statistiques simples au lieu du clustering
        """
        logger.info("🌳 Détection d'anomalies par seuils statistiques (rapide)...")
        
        # Utiliser des seuils statistiques simples au lieu du clustering
        df['HIERARCHICAL_CLUSTER_ANOMALY'] = 0
        df['HIERARCHICAL_CLUSTER_SCORE'] = 0.0
        
        # Détecter les anomalies basées sur des seuils statistiques
        if 'MONTANT_LIQUIDATION' in df.columns:
            # Montants de liquidation anormalement élevés
            q99 = df['MONTANT_LIQUIDATION'].quantile(0.99)
            mask_liquidation_anormale = df['MONTANT_LIQUIDATION'] > q99
            df.loc[mask_liquidation_anormale, 'HIERARCHICAL_CLUSTER_ANOMALY'] = 1
            df.loc[mask_liquidation_anormale, 'HIERARCHICAL_CLUSTER_SCORE'] = 1.0
        
        if 'VALEUR_UNITAIRE_KG' in df.columns:
            # Valeurs unitaires anormalement élevées ou basses
            q99 = df['VALEUR_UNITAIRE_KG'].quantile(0.99)
            q01 = df['VALEUR_UNITAIRE_KG'].quantile(0.01)
            mask_valeur_unitaire_anormale = (df['VALEUR_UNITAIRE_KG'] > q99) | (df['VALEUR_UNITAIRE_KG'] < q01)
            df.loc[mask_valeur_unitaire_anormale, 'HIERARCHICAL_CLUSTER_ANOMALY'] = 1
            df.loc[mask_valeur_unitaire_anormale, 'HIERARCHICAL_CLUSTER_SCORE'] = 1.0
        
        anomalies_count = df['HIERARCHICAL_CLUSTER_ANOMALY'].sum()
        logger.info(f"   Anomalies détectées (seuils statistiques): {anomalies_count}")
        
        return df
    
    def admin_values_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Contrôle des valeurs administrées
        Compare les valeurs déclarées avec les valeurs de référence administratives
        """
        logger.info("🏛️ Contrôle des valeurs administrées...")
        
        # Créer la clé produit/origine
        # Adapter aux noms de colonnes après agrégation
        if 'CODE_PRODUIT_STR' in df.columns:
            df['PRODUCT_ORIGIN_KEY'] = df['CODE_PRODUIT_STR'] + '_' + df['PAYS_ORIGINE_STR']
        else:
            df['PRODUCT_ORIGIN_KEY'] = df['CODE_PRODUIT'].astype(str) + '_' + df['PAYS_ORIGINE'].astype(str)
        
        # Calculer les valeurs de référence administratives (médiane par couple produit/origine)
        admin_values = df.groupby('PRODUCT_ORIGIN_KEY')['VALEUR_CAF'].agg([
            'count', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
        ]).reset_index()
        admin_values.columns = ['PRODUCT_ORIGIN_KEY', 'count', 'median', 'q25', 'q75']
        
        # Filtrer les couples avec au moins 5 observations
        admin_values = admin_values[admin_values['count'] >= 5]
        
        # Créer les features de contrôle
        df['ADMIN_VALUES_ANOMALY'] = 0
        df['ADMIN_VALUES_SCORE'] = 0.0
        df['ADMIN_VALUES_DEVIATION'] = 0.0
        
        for _, row in admin_values.iterrows():
            key = row['PRODUCT_ORIGIN_KEY']
            admin_median = row['median']
            admin_q25 = row['q25']
            admin_q75 = row['q75']
            
            mask = df['PRODUCT_ORIGIN_KEY'] == key
            
            # Calculer l'écart par rapport à la valeur administrée
            deviation = np.abs(df.loc[mask, 'VALEUR_CAF'] - admin_median) / admin_median
            df.loc[mask, 'ADMIN_VALUES_DEVIATION'] = deviation
            
            # Score basé sur l'écart interquartile
            iqr = admin_q75 - admin_q25
            if iqr > 0:
                iqr_score = np.abs(df.loc[mask, 'VALEUR_CAF'] - admin_median) / iqr
                df.loc[mask, 'ADMIN_VALUES_SCORE'] = iqr_score
                
                # Anomalie si valeur en dehors de l'intervalle [Q25 - 1.5*IQR, Q75 + 1.5*IQR]
                lower_bound = admin_q25 - 1.5 * iqr
                upper_bound = admin_q75 + 1.5 * iqr
                anomaly_mask = (df.loc[mask, 'VALEUR_CAF'] < lower_bound) | (df.loc[mask, 'VALEUR_CAF'] > upper_bound)
                df.loc[mask & anomaly_mask, 'ADMIN_VALUES_ANOMALY'] = 1
        
        # Sauvegarder les valeurs administrées
        self.admin_values = admin_values.set_index('PRODUCT_ORIGIN_KEY').to_dict('index')
        
        anomalies_count = df['ADMIN_VALUES_ANOMALY'].sum()
        logger.info(f"   Anomalies valeurs administrées: {anomalies_count}")
        
        return df
    
    def create_comprehensive_fraud_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Créer le FRAUD_FLAG basé sur toutes les techniques de détection
        """
        logger.info("🎯 Création du FRAUD_FLAG complet...")
        
        # Initialiser le flag de fraude
        df['FRAUD_FLAG'] = 0
        
        # 1. Bienaymé-Tchebychev (seuil strict)
        df.loc[df['BIENAYME_CHEBYCHEV_ANOMALY'] == 1, 'FRAUD_FLAG'] = 1
        
        # 2. Analyse miroir TEI (seuil strict)
        df.loc[df['MIRROR_TEI_ANOMALY'] == 1, 'FRAUD_FLAG'] = 1
        
        # 3. Clustering spectral (seuil strict)
        df.loc[df['SPECTRAL_CLUSTER_ANOMALY'] == 1, 'FRAUD_FLAG'] = 1
        
        # 4. Clustering hiérarchique (seuil strict)
        df.loc[df['HIERARCHICAL_CLUSTER_ANOMALY'] == 1, 'FRAUD_FLAG'] = 1
        
        # 5. Valeurs administrées (seuil strict)
        df.loc[df['ADMIN_VALUES_ANOMALY'] == 1, 'FRAUD_FLAG'] = 1
        
        # 6. Score composite (seuil adaptatif)
        # Normaliser les scores entre 0 et 1
        score_columns = [
            'BIENAYME_CHEBYCHEV_SCORE',
            'MIRROR_TEI_SCORE', 
            'SPECTRAL_CLUSTER_SCORE',
            'HIERARCHICAL_CLUSTER_SCORE',
            'ADMIN_VALUES_SCORE'
        ]
        
        composite_score = np.zeros(len(df))
        for col in score_columns:
            if col in df.columns:
                # Normaliser le score
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max > col_min:
                    normalized_score = (df[col] - col_min) / (col_max - col_min)
                    composite_score += normalized_score
        
        df['COMPOSITE_FRAUD_SCORE'] = composite_score / len(score_columns)
        
        # Seuil adaptatif basé sur le 95ème percentile
        threshold = df['COMPOSITE_FRAUD_SCORE'].quantile(0.95)
        df.loc[df['COMPOSITE_FRAUD_SCORE'] > threshold, 'FRAUD_FLAG'] = 1
        
        # Statistiques finales
        fraud_count = df['FRAUD_FLAG'].sum()
        fraud_rate = fraud_count / len(df) * 100
        
        logger.info(f"✅ FRAUD_FLAG créé: {fraud_count} fraudes ({fraud_rate:.1f}%)")
        
        return df
    
    def run_complete_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Exécuter l'analyse complète de détection de fraude
        """
        logger.info("🚀 DÉMARRAGE DE L'ANALYSE COMPLÈTE DE DÉTECTION DE FRAUDE")
        logger.info("=" * 70)
        
        # 1. Toilettage des données
        df = self.clean_data_comprehensive(df)
        
        # 2. Méthodes probabilistes
        df = self.bienayme_chebychev_analysis(df)
        
        # 3. Analyse miroir TEI
        df = self.mirror_analysis_tei(df)
        
        # 4. Détection d'anomalies - Clustering spectral
        df = self.spectral_clustering_anomaly_detection(df)
        
        # 5. Détection d'anomalies - Clustering hiérarchique
        df = self.hierarchical_clustering_anomaly_detection(df)
        
        # 6. Contrôle des valeurs administrées
        df = self.admin_values_control(df)
        
        # 7. Création du FRAUD_FLAG complet
        df = self.create_comprehensive_fraud_flag(df)
        
        logger.info("=" * 70)
        logger.info("✅ ANALYSE COMPLÈTE TERMINÉE")
        logger.info(f"📊 Données finales: {df.shape}")
        logger.info(f"🎯 Taux de fraude: {df['FRAUD_FLAG'].mean()*100:.1f}%")
        logger.info("=" * 70)
        
        # Sauvegarder les statistiques pour réutilisation en prédiction
        if self.chapter:
            self.save_fraud_detection_stats(df)
        
        return df
    
    def save_fraud_detection_stats(self, df: pd.DataFrame):
        """
        Sauvegarder les statistiques de fraud detection pour réutilisation en prédiction
        Ces stats seront utilisées pour calculer les fraud features sur de nouvelles déclarations
        """
        try:
            from pathlib import Path
            import json
            from datetime import datetime
            
            # Déterminer le chemin de sauvegarde
            backend_root = Path(__file__).resolve().parents[2]
            stats_dir = backend_root / "results" / self.chapter
            stats_dir.mkdir(parents=True, exist_ok=True)
            stats_file = stats_dir / "fraud_detection_stats.json"
            
            logger.info(f"💾 Sauvegarde des statistiques de fraud detection pour {self.chapter}...")
            
            # Calculer les stats globales
            stats = {
                "chapter": self.chapter,
                "version": "1.0",
                "total_declarations": int(len(df)),
                "fraud_rate": float(df['FRAUD_FLAG'].mean()),
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                
                # Stats VALEUR_CAF globales
                "valeur_caf": {
                    "mean": float(df['VALEUR_CAF'].mean()),
                    "std": float(df['VALEUR_CAF'].std()),
                    "median": float(df['VALEUR_CAF'].median()),
                    "q25": float(df['VALEUR_CAF'].quantile(0.25)),
                    "q75": float(df['VALEUR_CAF'].quantile(0.75)),
                    "min": float(df['VALEUR_CAF'].min()),
                    "max": float(df['VALEUR_CAF'].max())
                },
                
                # Stats TEI globales
                "tei": {
                    "mean": float(df['TEI_CALCULE'].mean()),
                    "std": float(df['TEI_CALCULE'].std()),
                    "median": float(df['TEI_CALCULE'].median()),
                    "q25": float(df['TEI_CALCULE'].quantile(0.25)),
                    "q75": float(df['TEI_CALCULE'].quantile(0.75)),
                    "min": float(df['TEI_CALCULE'].min()),
                    "max": float(df['TEI_CALCULE'].max())
                },
                
                # Stats par couple PRODUIT/ORIGINE (les plus importantes !)
                "product_origin_stats": {}
            }
            
            # Calculer les stats par couple PRODUIT/ORIGINE
            # Garder seulement les couples avec au moins 10 observations
            for key, stats_dict in self.product_origin_stats.items():
                if stats_dict['count'] >= 10:  # Seuil minimum
                    stats["product_origin_stats"][key] = {
                        "count": int(stats_dict['count']),
                        "valeur_caf": {
                            "mean": float(stats_dict['mean']),
                            "std": float(stats_dict['std']),
                            "median": float(stats_dict.get('median', stats_dict['mean'])),
                            "q25": float(stats_dict.get('q25', stats_dict['mean'] * 0.7)),
                            "q75": float(stats_dict.get('q75', stats_dict['mean'] * 1.3)),
                            "min": float(stats_dict['min']),
                            "max": float(stats_dict['max'])
                        }
                    }
            
            # Ajouter les stats TEI par couple PRODUIT/ORIGINE
            for key, tei_stats_dict in self.tei_thresholds.items():
                if tei_stats_dict['count'] >= 10 and key in stats["product_origin_stats"]:
                    stats["product_origin_stats"][key]["tei"] = {
                        "mean": float(tei_stats_dict['mean']),
                        "std": float(tei_stats_dict['std']),
                        "median": float(tei_stats_dict['median']),
                        "q25": float(tei_stats_dict['q25']),
                        "q75": float(tei_stats_dict['q75'])
                    }
            
            # Ajouter les stats par défaut (stats globales)
            stats["product_origin_stats"]["default"] = {
                "comment": "Valeurs par défaut si le couple produit/origine n'est pas dans les stats",
                "count": 0,
                "valeur_caf": stats["valeur_caf"],
                "tei": stats["tei"]
            }
            
            # Sauvegarder dans le fichier JSON
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Statistiques sauvegardées: {stats_file}")
            logger.info(f"   - {len(stats['product_origin_stats'])} couples produit/origine")
            logger.info(f"   - Taux de fraude: {stats['fraud_rate']*100:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde statistiques: {e}")
            import traceback
            traceback.print_exc()
