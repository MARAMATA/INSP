#!/usr/bin/env python3
"""
Preprocessing Chapitre 84 - Machines et équipements mécaniques
Basé sur le preprocessing du chapitre 30 avec règles de fraude spécifiques au chapitre 84
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Ajouter le chemin pour importer advanced_fraud_detection
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.advanced_fraud_detection import AdvancedFraudDetection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Chap84PreprocessorComprehensive:
    """Preprocessor COMPLET pour le Chapitre 84 avec FRAUD_FLAG basé sur TOUTES les règles métiers douanières"""
    
    def __init__(self):
        backend_root = Path(__file__).resolve().parents[3]
        self.backend_root = backend_root
        self.raw_data_path = backend_root / "data/raw/CHAPITRE_84.csv"
        self.processed_data_path = backend_root / "data/processed/CHAP84_PROCESSED_ADVANCED.csv"
        
        # Colonnes pour créer DECLARATION_ID
        self.declaration_id_cols = ['ANNEE', 'BUREAU', 'NUMERO_DECLARATION']
        
        # Initialiser le détecteur de fraude avancé
        self.fraud_detector = AdvancedFraudDetection()
        
        logger.info("✅ Preprocessor COMPLET initialisé avec toutes les règles métiers douanières")
    
    def load_data(self):
        """Charger les données brutes"""
        logger.info("📊 Chargement des données brutes...")
        df = pd.read_csv(self.raw_data_path)
        logger.info(f"✅ Données chargées: {df.shape}")
        return df
    
    def clean_data(self, df):
        """Nettoyage des données optimisé"""
        logger.info("🧹 Nettoyage des données...")
        
        # Créer DECLARATION_ID
        df['DECLARATION_ID'] = df[self.declaration_id_cols].astype(str).agg('/'.join, axis=1)
        logger.info(f"   DECLARATION_ID créé")
        
        # Gestion des valeurs manquantes
        df['VALEUR_DOUANE'] = df['VALEUR_DOUANE'].fillna(df['VALEUR_CAF'])
        df['PAYS_PROVENANCE'] = df['PAYS_PROVENANCE'].fillna(df['PAYS_ORIGINE'])
        df['NUMERO_DPI'] = df['NUMERO_DPI'].fillna('SANS_DPI')
        
        # Créer des features de base
        df['CODE_PRODUIT_STR'] = df['CODE_PRODUIT'].astype(str)
        df['PAYS_ORIGINE_STR'] = df['PAYS_ORIGINE'].astype(str)
        df['PAYS_PROVENANCE_STR'] = df['PAYS_PROVENANCE'].astype(str)
        
        # Utiliser toutes les colonnes importantes
        df['NUMERO_ARTICLE_STR'] = df['NUMERO_ARTICLE'].astype(str)
        df['PRECISION_UEMOA_STR'] = df['PRECISION_UEMOA'].astype(str)
        df['DATE_DECLARATION_STR'] = df['DATE_DECLARATION'].astype(str)
        
        # Garder aussi les colonnes originales
        df['PRECISION_UEMOA'] = df['PRECISION_UEMOA'].astype(int)
        df['NUMERO_ARTICLE'] = df['NUMERO_ARTICLE'].astype(int)
        
        # Calculer la valeur unitaire par kg
        df['VALEUR_UNITAIRE_KG'] = df['VALEUR_CAF'] / df['POIDS_NET'].replace(0, 1)
        
        # Calculer le taux de droits
        df['TAUX_DROITS_PERCENT'] = (df['MONTANT_LIQUIDATION'] / df['VALEUR_CAF'].replace(0, 1)) * 100
        
        # Calculer le ratio valeur douane/CAF
        df['RATIO_DOUANE_CAF'] = df['VALEUR_DOUANE'] / df['VALEUR_CAF'].replace(0, 1)
        
        logger.info(f"✅ Données nettoyées: {df.shape}")
        return df
    
    def aggregate_data(self, df):
        """Agrégation par DECLARATION_ID"""
        logger.info("📊 Agrégation par DECLARATION_ID...")
        
        # Colonnes numériques à agréger
        numeric_cols = [
            'VALEUR_CAF', 'VALEUR_DOUANE', 'MONTANT_LIQUIDATION', 'POIDS_NET',
            'VALEUR_UNITAIRE_KG', 'TAUX_DROITS_PERCENT', 'RATIO_DOUANE_CAF'
        ]
        
        # Colonnes catégorielles à agréger
        categorical_cols = [
            'CODE_PRODUIT_STR', 'PAYS_ORIGINE_STR', 'PAYS_PROVENANCE_STR',
            'NUMERO_ARTICLE_STR', 'PRECISION_UEMOA_STR', 'DATE_DECLARATION_STR',
            'BUREAU', 'REGIME_FISCAL', 'NUMERO_DPI', 'PRECISION_UEMOA', 'NUMERO_ARTICLE'
        ]
        
        # Dictionnaire d'agrégation
        agg_dict = {}
        
        # Agrégation numérique (somme)
        for col in numeric_cols:
            if col in df.columns:
                agg_dict[col] = 'sum'
        
        # Agrégation catégorielle (première valeur)
        for col in categorical_cols:
            if col in df.columns:
                agg_dict[col] = 'first'
        
        # Agrégation
        df_agg = df.groupby('DECLARATION_ID').agg(agg_dict).reset_index()
        
        # Recalculer VALEUR_UNITAIRE_KG après agrégation
        if 'VALEUR_CAF' in df_agg.columns and 'POIDS_NET' in df_agg.columns:
            df_agg['VALEUR_UNITAIRE_KG'] = df_agg['VALEUR_CAF'] / df_agg['POIDS_NET'].replace(0, 1)
        
        logger.info(f"✅ Données agrégées: {df.shape[0]} → {df_agg.shape[0]} déclarations")
        return df_agg
    
    def create_advanced_fraud_flag(self, df):
        """Créer un FRAUD_FLAG basé sur les techniques avancées de détection de fraude"""
        logger.info("🎯 Création du FRAUD_FLAG avec techniques avancées...")
        
        # Utiliser le détecteur de fraude avancé
        df = self.fraud_detector.run_complete_analysis(df)
        
        # Ajouter des règles spécifiques au chapitre 84 - MACHINES ET ÉQUIPEMENTS
        logger.info("📋 Application des règles spécifiques Chapitre 84 - MACHINES ET ÉQUIPEMENTS...")
        
        # 1. DÉTECTION GLISSEMENT TARIFAIRE - Machines classées comme pièces détachées
        # Règle principale : machines complètes classées comme pièces détachées
        mask_machines = df['CODE_PRODUIT_STR'].str.startswith('84')
        
        # Caractéristiques des machines : valeur unitaire élevée, poids élevé
        # Seuil élevé pour détecter les machines complètes
        seuil_machine = df['VALEUR_UNITAIRE_KG'].quantile(0.95)
        mask_valeur_machine = df['VALEUR_UNITAIRE_KG'] > seuil_machine
        
        # Poids élevé typique des machines
        seuil_poids_machine = df['POIDS_NET'].quantile(0.90)
        mask_poids_machine = df['POIDS_NET'] > seuil_poids_machine
        
        # Détecter les machines mal classées
        mask_machines_mal_classes = mask_machines & mask_valeur_machine & mask_poids_machine
        df.loc[mask_machines_mal_classes, 'FRAUD_FLAG'] = 1
        logger.info(f"   GLISSEMENT TARIFAIRE - Machines mal classées: {mask_machines_mal_classes.sum()} cas")
        
        # 2. DÉTECTION GLISSEMENT TARIFAIRE - Équipements électroniques
        # Codes d'équipements électroniques classés comme machines mécaniques
        codes_electroniques = ['8471', '8473', '8474', '8475', '8476', '8477', '8478', '8479']
        mask_codes_electroniques = df['CODE_PRODUIT_STR'].str.startswith(tuple(codes_electroniques))
        
        # Valeur unitaire très élevée (équipements électroniques)
        seuil_electronique = df['VALEUR_UNITAIRE_KG'].quantile(0.98)
        mask_electronique_luxe = df['VALEUR_UNITAIRE_KG'] > seuil_electronique
        df.loc[mask_codes_electroniques & mask_electronique_luxe, 'FRAUD_FLAG'] = 1
        logger.info(f"   GLISSEMENT TARIFAIRE - Équipements électroniques mal classés: {(mask_codes_electroniques & mask_electronique_luxe).sum()} cas")
        
        # 3. DÉTECTION GLISSEMENT TARIFAIRE - Pays d'origine machines
        # Pays spécialisés dans les machines classés comme pièces
        pays_machines = ['CN', 'DE', 'IT', 'JP', 'KR', 'US', 'FR']  # Pays producteurs de machines
        mask_pays_machines = df['PAYS_ORIGINE_STR'].isin(pays_machines)
        # Valeur unitaire élevée + code machine = glissement suspect
        seuil_glissement = df['VALEUR_UNITAIRE_KG'].quantile(0.90)
        mask_glissement_suspect = df['VALEUR_UNITAIRE_KG'] > seuil_glissement
        df.loc[mask_pays_machines & mask_glissement_suspect & mask_machines, 'FRAUD_FLAG'] = 1
        logger.info(f"   GLISSEMENT TARIFAIRE - Pays machines + code suspect: {(mask_pays_machines & mask_glissement_suspect & mask_machines).sum()} cas")
        
        # 4. DÉTECTION GLISSEMENT TARIFAIRE - Volumes suspects
        # Machines en gros volumes avec codes suspects
        seuil_volume_suspect = df['VALEUR_CAF'].quantile(0.95)
        mask_volume_suspect = df['VALEUR_CAF'] > seuil_volume_suspect
        df.loc[mask_machines & mask_volume_suspect, 'FRAUD_FLAG'] = 1
        logger.info(f"   GLISSEMENT TARIFAIRE - Volumes suspects: {(mask_machines & mask_volume_suspect).sum()} cas")
        
        # 5. DÉTECTION GLISSEMENT TARIFAIRE - Ratio poids/valeur suspect
        # Machines : poids élevé, valeur élevée
        df['RATIO_POIDS_VALEUR'] = df['POIDS_NET'] / df['VALEUR_CAF'].replace(0, 1)
        seuil_ratio_suspect = df['RATIO_POIDS_VALEUR'].quantile(0.05)  # Très faible ratio
        mask_ratio_suspect = df['RATIO_POIDS_VALEUR'] < seuil_ratio_suspect
        df.loc[mask_machines & mask_ratio_suspect, 'FRAUD_FLAG'] = 1
        logger.info(f"   GLISSEMENT TARIFAIRE - Ratio poids/valeur suspect: {(mask_machines & mask_ratio_suspect).sum()} cas")
        
        # Statistiques finales
        fraud_count = df['FRAUD_FLAG'].sum()
        fraud_rate = fraud_count / len(df) * 100
        logger.info(f"✅ FRAUD_FLAG AVANCÉ: {fraud_count} fraudes ({fraud_rate:.1f}%)")
        
        return df
    
    def create_business_features(self, df):
        """Créer des features business optimisées - FOCUS GLISSEMENT TARIFAIRE"""
        logger.info("📋 Création des features business optimisées...")
        
        # 1. FEATURES GLISSEMENT TARIFAIRE (les plus importantes)
        # Détection machines mal classées
        df['BUSINESS_GLISSEMENT_MACHINE'] = (
            (df['CODE_PRODUIT_STR'].str.startswith('84')) & 
            (df['VALEUR_UNITAIRE_KG'] > df['VALEUR_UNITAIRE_KG'].quantile(0.95)) &
            (df['POIDS_NET'] > df['POIDS_NET'].quantile(0.90))
        ).astype(int)
        
        # Pays machines + code suspect
        df['BUSINESS_GLISSEMENT_PAYS_MACHINES'] = (
            df['PAYS_ORIGINE_STR'].isin(['CN', 'DE', 'IT', 'JP', 'KR', 'US', 'FR']) &
            df['CODE_PRODUIT_STR'].str.startswith('84') &
            (df['VALEUR_UNITAIRE_KG'] > df['VALEUR_UNITAIRE_KG'].quantile(0.90))
        ).astype(int)
        
        # Ratio poids/valeur suspect (machines)
        df['BUSINESS_GLISSEMENT_RATIO_SUSPECT'] = (
            (df['RATIO_POIDS_VALEUR'] < df['RATIO_POIDS_VALEUR'].quantile(0.05)) &
            df['CODE_PRODUIT_STR'].str.startswith('84')
        ).astype(int)
        
        # 2. FEATURES RISQUE PAYS (contrefaçon)
        high_risk_countries = ['CN', 'IN', 'PK', 'BD', 'LK']
        df['BUSINESS_RISK_PAYS_HIGH'] = df['PAYS_ORIGINE_STR'].isin(high_risk_countries).astype(int)
        df['BUSINESS_ORIGINE_DIFF_PROVENANCE'] = (df['PAYS_ORIGINE_STR'] != df['PAYS_PROVENANCE_STR']).astype(int)
        
        # 3. FEATURES RÉGIME (détournement)
        df['BUSINESS_REGIME_PREFERENTIEL'] = df['REGIME_FISCAL'].isin([10, 20, 30, 40]).astype(int)
        df['BUSINESS_REGIME_NORMAL'] = (df['REGIME_FISCAL'] == 0).astype(int)
        
        # 4. FEATURES VALEUR (volumes suspects)
        df['BUSINESS_VALEUR_ELEVEE'] = (df['VALEUR_CAF'] > df['VALEUR_CAF'].quantile(0.9)).astype(int)
        df['BUSINESS_VALEUR_EXCEPTIONNELLE'] = (df['VALEUR_CAF'] > df['VALEUR_CAF'].quantile(0.95)).astype(int)
        
        # 5. FEATURES POIDS
        df['BUSINESS_POIDS_ELEVE'] = (df['POIDS_NET'] > df['POIDS_NET'].quantile(0.9)).astype(int)
        
        # 6. FEATURES TAUX DE DROITS
        df['BUSINESS_DROITS_ELEVES'] = (df['TAUX_DROITS_PERCENT'] > 20).astype(int)
        
        # 7. FEATURES RATIOS
        df['BUSINESS_RATIO_LIQUIDATION_CAF'] = df['MONTANT_LIQUIDATION'] / df['VALEUR_CAF'].replace(0, 1)
        df['BUSINESS_RATIO_DOUANE_CAF'] = df['RATIO_DOUANE_CAF']
        
        # 8. FEATURES MACHINES (légitimes)
        df['BUSINESS_IS_MACHINE'] = df['CODE_PRODUIT_STR'].str.startswith('84').astype(int)
        df['BUSINESS_IS_ELECTRONIQUE'] = df['CODE_PRODUIT_STR'].str.startswith(('8471', '8473', '8474', '8475', '8476', '8477', '8478', '8479')).astype(int)
        df['BUSINESS_IS_PRECISION_UEMOA'] = (df['PRECISION_UEMOA'] == 90).astype(int)
        
        # 9. FEATURES ARTICLES ET DPI
        df['BUSINESS_ARTICLES_MULTIPLES'] = (df['NUMERO_ARTICLE'] > 1).astype(int)
        df['BUSINESS_AVEC_DPI'] = (df['NUMERO_DPI'] != 'SANS_DPI').astype(int)
        
        logger.info("✅ Features business créées (18 features principales)")
        return df
    
    def handle_missing_values(self, df):
        """Gestion des valeurs manquantes optimisée"""
        logger.info("🔧 Gestion des valeurs manquantes...")
        
        # Remplacer les infinis par NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Colonnes catégorielles
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    df[col] = df[col].fillna('UNKNOWN')
        
        logger.info("✅ Valeurs manquantes traitées")
        return df
    
    def run_preprocessing(self):
        """Pipeline complet de preprocessing optimisé"""
        logger.info("🚀 DÉMARRAGE DU PREPROCESSING AVANCÉ CHAPITRE 84")
        logger.info("=" * 60)
        
        try:
            # 1. Charger les données
            df = self.load_data()
            
            # 2. Nettoyer les données
            df = self.clean_data(df)
            
            # 3. Agréger par DECLARATION_ID
            df = self.aggregate_data(df)
            
            # 4. Gérer les valeurs manquantes
            df = self.handle_missing_values(df)
            
            # 5. Créer le FRAUD_FLAG avec techniques avancées
            df = self.create_advanced_fraud_flag(df)
            
            # 6. Créer les features business
            df = self.create_business_features(df)
            
            # 7. Nettoyage final
            df = df.dropna(subset=['FRAUD_FLAG'])
            
            # 8. Sauvegarder
            self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.processed_data_path, index=False)
            
            # 9. Rapport final
            logger.info("=" * 60)
            logger.info("✅ PREPROCESSING CHAPITRE 84 TERMINÉ AVEC SUCCÈS")
            logger.info(f"📊 Données finales: {df.shape}")
            logger.info(f"🎯 Taux de fraude: {df['FRAUD_FLAG'].mean()*100:.1f}%")
            logger.info(f"📁 Fichier sauvegardé: {self.processed_data_path}")
            logger.info("=" * 60)
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du preprocessing: {e}")
            raise

def main():
    """Fonction principale"""
    try:
        preprocessor = Chap84PreprocessorComprehensive()
        preprocessor.run_preprocessing()
    except Exception as e:
        logger.error(f"❌ Erreur dans main: {e}")
        raise

if __name__ == "__main__":
    main()
    