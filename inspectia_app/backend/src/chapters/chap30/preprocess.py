#!/usr/bin/env python3
"""
Preprocessing Chapitre 30 - Version AVANCÉE
FRAUD_FLAG basé sur les techniques de la cellule de ciblage et de veille commerciale :
- Méthodes probabilistes (théorème de Bienaymé-Tchebychev)
- Analyse miroir avec TEI (Taux Effectifs d'Imposition)
- Détection d'anomalies (clustering spectral et hiérarchique)
- Contrôle des valeurs administrées
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml
import hashlib
from typing import Dict, List, Tuple
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.advanced_fraud_detection import AdvancedFraudDetection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_mode(x, fallback="UNKNOWN"):
    """Mode robuste avec fallback"""
    try:
        m = x.mode(dropna=True)
        if m.empty:
            return fallback
        return m.iloc[0]
    except Exception:
        return fallback

def safe_first(x, fallback=None):
    """Première valeur non-nulle avec fallback"""
    try:
        for v in x:
            if pd.notna(v):
                return v
        return fallback
    except Exception:
        return fallback

class Chap30PreprocessorComprehensive:
    """Preprocessor COMPLET avec FRAUD_FLAG basé sur TOUTES les règles métiers douanières"""
    
    def __init__(self):
        backend_root = Path(__file__).resolve().parents[3]
        self.backend_root = backend_root
        self.raw_data_path = backend_root / "data/raw/CHAPITRE_30.csv"
        self.processed_data_path = backend_root / "data/processed/CHAP30_PROCESSED_ADVANCED.csv"
        # Initialiser le détecteur de fraude avancé avec le chapitre
        self.fraud_detector = AdvancedFraudDetection(chapter='chap30')
        
        # Colonnes pour DECLARATION_ID
        self.declaration_id_cols = ['ANNEE', 'BUREAU', 'NUMERO_DECLARATION']
        
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
        
        # Calculer la valeur unitaire par kg
        df['VALEUR_UNITAIRE_KG'] = df['VALEUR_CAF'] / df['POIDS_NET'].replace(0, 1)
        
        # Calculer le taux de droits
        df['TAUX_DROITS_PERCENT'] = (df['MONTANT_LIQUIDATION'] / df['VALEUR_CAF'].replace(0, 1)) * 100
        
        # Calculer le ratio valeur douane/CAF
        df['RATIO_DOUANE_CAF'] = df['VALEUR_DOUANE'] / df['VALEUR_CAF'].replace(0, 1)
        
        logger.info(f"✅ Données nettoyées: {df.shape}")
        return df
    
    def aggregate_data(self, df):
        """Agréger par DECLARATION_ID - optimisé"""
        logger.info("📊 Agrégation par DECLARATION_ID...")
        
        # Colonnes numériques à agréger
        numeric_cols = [
            'VALEUR_CAF', 'VALEUR_DOUANE', 'MONTANT_LIQUIDATION', 'POIDS_NET',
            'VALEUR_UNITAIRE_KG', 'TAUX_DROITS_PERCENT', 'RATIO_DOUANE_CAF',
            'NUMERO_ARTICLE', 'PRECISION_UEMOA'
        ]
        
        # Colonnes catégorielles à prendre en premier
        categorical_cols = [
            'CODE_PRODUIT_STR', 'PAYS_ORIGINE_STR', 'PAYS_PROVENANCE_STR',
            'BUREAU', 'REGIME_FISCAL', 'NUMERO_DPI', 'DATE_DECLARATION_STR'
        ]
        
        # Filtrer les colonnes existantes
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        # Agrégation optimisée
        agg_dict = {}
        
        # Numériques : somme
        for col in numeric_cols:
            agg_dict[col] = 'sum'
        
        # Catégorielles : première valeur
        for col in categorical_cols:
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
        
        # Ajouter des règles spécifiques au chapitre 30 - GLISSEMENT TARIFAIRE
        logger.info("📋 Application des règles spécifiques Chapitre 30 - GLISSEMENT TARIFAIRE...")
        
        # 1. DÉTECTION GLISSEMENT TARIFAIRE - Cosmétiques (CH33) classés comme pharmaceutiques (CH30)
        # Règle principale : produits avec caractéristiques cosmétiques classés comme médicaments
        mask_medicaments = df['CODE_PRODUIT_STR'].str.startswith('300490')
        
        # Caractéristiques des cosmétiques : valeur unitaire élevée, poids faible, ratio suspect
        # Seuil élevé pour détecter les cosmétiques (valeur unitaire > Q95)
        seuil_cosmetique = df['VALEUR_UNITAIRE_KG'].quantile(0.95)
        mask_valeur_cosmetique = df['VALEUR_UNITAIRE_KG'] > seuil_cosmetique
        
        # Ratio poids/valeur typique des cosmétiques (très faible)
        df['RATIO_POIDS_VALEUR'] = df['POIDS_NET'] / df['VALEUR_CAF'].replace(0, 1)
        seuil_ratio_cosmetique = df['RATIO_POIDS_VALEUR'].quantile(0.05)
        mask_ratio_cosmetique = df['RATIO_POIDS_VALEUR'] < seuil_ratio_cosmetique
        
        # Détecter les cosmétiques mal classés
        mask_cosmetiques_mal_classes = mask_medicaments & mask_valeur_cosmetique & mask_ratio_cosmetique
        df.loc[mask_cosmetiques_mal_classes, 'FRAUD_FLAG'] = 1
        logger.info(f"   GLISSEMENT TARIFAIRE - Cosmétiques (CH33) classés pharma (CH30): {mask_cosmetiques_mal_classes.sum()} cas")
        
        # 2. DÉTECTION GLISSEMENT TARIFAIRE - Codes cosmétiques typiques mal classés
        # Codes SH cosmétiques (CH33) classés comme pharma (CH30)
        codes_cosmetiques_typiques = [
            '3301',  # Huiles essentielles
            '3302',  # Mélanges d'huiles essentielles
            '3303',  # Parfums et eaux de toilette
            '3304',  # Produits de beauté ou de maquillage
            '3305',  # Préparations pour les soins des cheveux
            '3306',  # Préparations pour l'hygiène bucco-dentaire
            '3307'   # Autres produits de parfumerie ou de toilette
        ]
        
        # Détecter les codes cosmétiques mal classés dans le chapitre 30
        mask_codes_cosmetiques = df['CODE_PRODUIT_STR'].str.startswith(tuple(codes_cosmetiques_typiques))
        # Mais ils sont classés dans le chapitre 30 (fraude évidente)
        mask_chapitre_30 = df['CODE_PRODUIT_STR'].str.startswith('30')
        mask_glissement_evident = mask_codes_cosmetiques & mask_chapitre_30
        df.loc[mask_glissement_evident, 'FRAUD_FLAG'] = 1
        logger.info(f"   GLISSEMENT TARIFAIRE - Codes cosmétiques (CH33) dans CH30: {mask_glissement_evident.sum()} cas")
        
        # 3. DÉTECTION GLISSEMENT TARIFAIRE - Produits de beauté suspects
        # Codes produits suspects pour cosmétiques mal classés
        codes_cosmetiques_suspects = ['300490', '300590', '300290']  # Codes pharma utilisés pour cosmétiques
        mask_codes_suspects = df['CODE_PRODUIT_STR'].isin(codes_cosmetiques_suspects)
        # Valeur unitaire très élevée (cosmétiques de luxe)
        seuil_cosmetique_luxe = df['VALEUR_UNITAIRE_KG'].quantile(0.98)
        mask_cosmetique_luxe = df['VALEUR_UNITAIRE_KG'] > seuil_cosmetique_luxe
        df.loc[mask_codes_suspects & mask_cosmetique_luxe, 'FRAUD_FLAG'] = 1
        logger.info(f"   GLISSEMENT TARIFAIRE - Cosmétiques de luxe mal classés: {(mask_codes_suspects & mask_cosmetique_luxe).sum()} cas")
        
        # 4. DÉTECTION GLISSEMENT TARIFAIRE - Pays d'origine cosmétiques
        # Pays spécialisés dans les cosmétiques classés comme pharma
        pays_cosmetiques = ['FR', 'IT', 'DE', 'ES', 'US', 'JP', 'KR']  # Pays producteurs de cosmétiques
        mask_pays_cosmetiques = df['PAYS_ORIGINE_STR'].isin(pays_cosmetiques)
        # Valeur unitaire élevée + code pharma = glissement suspect
        seuil_glissement = df['VALEUR_UNITAIRE_KG'].quantile(0.90)
        mask_glissement_suspect = df['VALEUR_UNITAIRE_KG'] > seuil_glissement
        df.loc[mask_pays_cosmetiques & mask_glissement_suspect & mask_medicaments, 'FRAUD_FLAG'] = 1
        logger.info(f"   GLISSEMENT TARIFAIRE - Pays cosmétiques + code pharma: {(mask_pays_cosmetiques & mask_glissement_suspect & mask_medicaments).sum()} cas")
        
        # 5. DÉTECTION GLISSEMENT TARIFAIRE - Volumes suspects
        # Cosmétiques en gros volumes avec codes pharma
        seuil_volume_suspect = df['VALEUR_CAF'].quantile(0.95)
        mask_volume_suspect = df['VALEUR_CAF'] > seuil_volume_suspect
        df.loc[mask_medicaments & mask_volume_suspect, 'FRAUD_FLAG'] = 1
        logger.info(f"   GLISSEMENT TARIFAIRE - Volumes suspects: {(mask_medicaments & mask_volume_suspect).sum()} cas")
        
        # Statistiques finales
        fraud_count = df['FRAUD_FLAG'].sum()
        fraud_rate = fraud_count / len(df) * 100
        logger.info(f"✅ FRAUD_FLAG AVANCÉ: {fraud_count} fraudes ({fraud_rate:.1f}%)")
        
        return df
    
    def create_business_features(self, df):
        """Créer des features business optimisées - FOCUS GLISSEMENT TARIFAIRE"""
        logger.info("📋 Création des features business optimisées...")
        
        # 1. FEATURES GLISSEMENT TARIFAIRE (les plus importantes)
        # FEATURE PRINCIPALE: Glissement tarifaire (CODE_SH ne commence PAS par 30)
        df['BUSINESS_GLISSEMENT_TARIFAIRE'] = (
            ~df['CODE_PRODUIT_STR'].str.startswith('30', na=False)
        ).astype(int)
        
        # SUPPRIMÉ: BUSINESS_GLISSEMENT_COSMETIQUE - Trop spécifique, redondant avec GLISSEMENT_TARIFAIRE
        # SUPPRIMÉ: BUSINESS_GLISSEMENT_PAYS_COSMETIQUES - Trop spécifique, redondant avec GLISSEMENT_TARIFAIRE
        
        # NOUVELLE FEATURE: Détecter "glissement" dans la description
        if 'DESCRIPTION_COMMERCIALE' in df.columns:
            df['BUSINESS_GLISSEMENT_DESCRIPTION'] = df['DESCRIPTION_COMMERCIALE'].str.contains(
                'glissement|cosmet|parfum|beauté|maquillage|soin|toilette', 
                case=False, 
                na=False
            ).astype(int)
        else:
            df['BUSINESS_GLISSEMENT_DESCRIPTION'] = 0
        
        # Ratio poids/valeur suspect
        df['BUSINESS_GLISSEMENT_RATIO_SUSPECT'] = (
            (df['RATIO_POIDS_VALEUR'] < df['RATIO_POIDS_VALEUR'].quantile(0.05))
        ).astype(int)
        
        # 2. FEATURES RISQUE PAYS (contrefaçon)
        high_risk_countries = ['IN', 'CN', 'PK', 'BD', 'LK']
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
        
        # 8. FEATURES MÉDICAMENTS (légitimes)
        # SUPPRIMÉ: BUSINESS_IS_MEDICAMENT - Toujours 1 pour chap30, pas discriminante
        df['BUSINESS_IS_ANTIPALUDEEN'] = df['CODE_PRODUIT_STR'].str.contains('300360|300460', na=False).astype(int)
        # SUPPRIMÉ: BUSINESS_IS_PRECISION_UEMOA - Toujours 1 pour UEMOA, pas discriminante
        
        # 9. FEATURES ARTICLES ET DPI
        df['BUSINESS_ARTICLES_MULTIPLES'] = (df['NUMERO_ARTICLE'] > 1).astype(int)
        df['BUSINESS_AVEC_DPI'] = (df['NUMERO_DPI'] != 'SANS_DPI').astype(int)
        
        logger.info("✅ Features business créées (16 features principales)")
        return df
    
    def handle_missing_values(self, df):
        """Gestion des valeurs manquantes optimisée"""
        logger.info("🔧 Gestion des valeurs manquantes...")
        
        # Remplacer les valeurs infinies et NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Colonnes numériques : médiane
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Colonnes catégorielles : mode ou 'UNKNOWN'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'UNKNOWN'
                df[col] = df[col].fillna(mode_val)
        
        logger.info("✅ Valeurs manquantes traitées")
        return df
    
    def run_preprocessing(self):
        """Pipeline complet de preprocessing optimisé"""
        logger.info("🚀 DÉMARRAGE DU PREPROCESSING AVANCÉ CHAPITRE 30")
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
            logger.info("✅ PREPROCESSING CHAPITRE 30 TERMINÉ AVEC SUCCÈS")
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
        preprocessor = Chap30PreprocessorComprehensive()
        df = preprocessor.run_preprocessing()
        
        print("\n" + "=" * 60)
        print("🎉 PREPROCESSING CHAPITRE 30 TERMINÉ AVEC SUCCÈS")
        print("=" * 60)
        print(f"📊 Données finales: {df.shape}")
        print(f"🎯 Taux de fraude: {df['FRAUD_FLAG'].mean()*100:.1f}%")
        print(f"📁 Fichier: {preprocessor.processed_data_path}")
        print("=" * 60)
        print("📋 Techniques appliquées:")
        print("   ✅ Méthodes probabilistes (Bienaymé-Tchebychev)")
        print("   ✅ Analyse miroir avec TEI")
        print("   ✅ Détection d'anomalies (clustering)")
        print("   ✅ Contrôle des valeurs administrées")
        print("   ✅ Règles spécifiques Chapitre 30")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
