#!/usr/bin/env python3
"""
Script de diagnostic pour le pipeline ML
"""

import sys
import os
sys.path.append('/Users/macbook/Desktop/INSP/inspectia_app/backend')

from src.shared.ocr_ingest import create_advanced_context_from_ocr_data
from src.shared.ocr_pipeline import load_ml_model
import pandas as pd
import numpy as np

def debug_ml_pipeline():
    """Diagnostiquer le pipeline ML complet"""
    
    print("🔍 DIAGNOSTIC DU PIPELINE ML")
    print("=" * 50)
    
    # Test avec deux déclarations différentes
    declaration1 = {
        'DECLARATION_ID': '2024/DAKAR/1',
        'CODE_SH_COMPLET': '30049000',
        'CODE_PAYS_ORIGINE': 'FR',
        'VALEUR_CAF': 50000000.0,
        'POIDS_NET_KG': 100.0,
        'NOMBRE_COLIS': 1,
        'REGIME_FISCAL': 'normal'
    }
    
    declaration2 = {
        'DECLARATION_ID': '2024/DAKAR/2', 
        'CODE_SH_COMPLET': '30049001',
        'CODE_PAYS_ORIGINE': 'CN',
        'VALEUR_CAF': 1000000.0,
        'POIDS_NET_KG': 50000.0,
        'NOMBRE_COLIS': 1,
        'REGIME_FISCAL': 'normal'
    }
    
    # Créer les contextes avancés
    print("\n🔍 CRÉATION DES CONTEXTES AVANCÉS")
    context1 = create_advanced_context_from_ocr_data(declaration1, 'chap30')
    context2 = create_advanced_context_from_ocr_data(declaration2, 'chap30')
    
    print(f"   Contexte 1: {len(context1)} features")
    print(f"   Contexte 2: {len(context2)} features")
    
    # Comparer les features importantes
    important_features = [
        'VALEUR_CAF', 'POIDS_NET_KG', 'PAYS_ORIGINE_STR', 'CODE_PRODUIT_STR',
        'BUSINESS_GLISSEMENT_COSMETIQUE', 'BUSINESS_RISK_PAYS_HIGH', 'BUSINESS_VALEUR_ELEVEE',
        'BIENAYME_CHEBYCHEV_SCORE', 'MIRROR_TEI_SCORE', 'SPECTRAL_CLUSTER_SCORE'
    ]
    
    print("\n🔍 COMPARAISON DES FEATURES IMPORTANTES")
    for feature in important_features:
        val1 = context1.get(feature, 'NON_TROUVÉ')
        val2 = context2.get(feature, 'NON_TROUVÉ')
        print(f"   {feature}:")
        print(f"     Déclaration 1: {val1}")
        print(f"     Déclaration 2: {val2}")
        print(f"     Différent: {val1 != val2}")
    
    # Charger le modèle ML
    print("\n🔍 CHARGEMENT DU MODÈLE ML")
    ml_model_data = load_ml_model('chap30')
    
    if not ml_model_data:
        print("   ❌ Impossible de charger le modèle ML")
        return
    
    print(f"   ✅ Modèle ML chargé: {type(ml_model_data.get('model'))}")
    
    # Créer les DataFrames
    df1 = pd.DataFrame([context1])
    df2 = pd.DataFrame([context2])
    
    print(f"\n🔍 STRUCTURE DES DATAFRAMES")
    print(f"   DataFrame 1: {df1.shape}")
    print(f"   DataFrame 2: {df2.shape}")
    print(f"   Colonnes identiques: {list(df1.columns) == list(df2.columns)}")
    
    # Vérifier les colonnes manquantes
    all_columns = set(df1.columns) | set(df2.columns)
    missing_cols1 = all_columns - set(df1.columns)
    missing_cols2 = all_columns - set(df2.columns)
    
    if missing_cols1:
        print(f"   Colonnes manquantes dans DF1: {missing_cols1}")
    if missing_cols2:
        print(f"   Colonnes manquantes dans DF2: {missing_cols2}")
    
    # Test de prédiction
    print("\n🔍 TEST DE PRÉDICTION")
    try:
        pipeline = ml_model_data.get('model')
        
        # Prédiction pour la déclaration 1
        prob1 = pipeline.predict_proba(df1)[0][1]
        print(f"   Probabilité déclaration 1: {prob1:.6f}")
        
        # Prédiction pour la déclaration 2
        prob2 = pipeline.predict_proba(df2)[0][1]
        print(f"   Probabilité déclaration 2: {prob2:.6f}")
        
        print(f"   Probabilités identiques: {abs(prob1 - prob2) < 1e-10}")
        print(f"   Différence: {abs(prob1 - prob2):.10f}")
        
        # Analyser les features après preprocessing
        print("\n🔍 ANALYSE POST-PREPROCESSING")
        
        # Vérifier si le pipeline a des étapes de preprocessing
        if hasattr(pipeline, 'steps'):
            print(f"   Étapes du pipeline: {[step[0] for step in pipeline.steps]}")
            
            # Appliquer seulement le preprocessing
            preprocessor = pipeline.steps[0][1]  # Premier step
            df1_processed = preprocessor.transform(df1)
            df2_processed = preprocessor.transform(df2)
            
            print(f"   DataFrame 1 après preprocessing: {df1_processed.shape}")
            print(f"   DataFrame 2 après preprocessing: {df2_processed.shape}")
            
            # Comparer les premières valeurs
            print(f"   Premières valeurs DF1: {df1_processed[0][:10]}")
            print(f"   Premières valeurs DF2: {df2_processed[0][:10]}")
            print(f"   Vecteurs identiques: {np.array_equal(df1_processed[0], df2_processed[0])}")
            
    except Exception as e:
        print(f"   ❌ Erreur prédiction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ml_pipeline()





Script de diagnostic pour le pipeline ML
"""

import sys
import os
sys.path.append('/Users/macbook/Desktop/INSP/inspectia_app/backend')

from src.shared.ocr_ingest import create_advanced_context_from_ocr_data
from src.shared.ocr_pipeline import load_ml_model
import pandas as pd
import numpy as np

def debug_ml_pipeline():
    """Diagnostiquer le pipeline ML complet"""
    
    print("🔍 DIAGNOSTIC DU PIPELINE ML")
    print("=" * 50)
    
    # Test avec deux déclarations différentes
    declaration1 = {
        'DECLARATION_ID': '2024/DAKAR/1',
        'CODE_SH_COMPLET': '30049000',
        'CODE_PAYS_ORIGINE': 'FR',
        'VALEUR_CAF': 50000000.0,
        'POIDS_NET_KG': 100.0,
        'NOMBRE_COLIS': 1,
        'REGIME_FISCAL': 'normal'
    }
    
    declaration2 = {
        'DECLARATION_ID': '2024/DAKAR/2', 
        'CODE_SH_COMPLET': '30049001',
        'CODE_PAYS_ORIGINE': 'CN',
        'VALEUR_CAF': 1000000.0,
        'POIDS_NET_KG': 50000.0,
        'NOMBRE_COLIS': 1,
        'REGIME_FISCAL': 'normal'
    }
    
    # Créer les contextes avancés
    print("\n🔍 CRÉATION DES CONTEXTES AVANCÉS")
    context1 = create_advanced_context_from_ocr_data(declaration1, 'chap30')
    context2 = create_advanced_context_from_ocr_data(declaration2, 'chap30')
    
    print(f"   Contexte 1: {len(context1)} features")
    print(f"   Contexte 2: {len(context2)} features")
    
    # Comparer les features importantes
    important_features = [
        'VALEUR_CAF', 'POIDS_NET_KG', 'PAYS_ORIGINE_STR', 'CODE_PRODUIT_STR',
        'BUSINESS_GLISSEMENT_COSMETIQUE', 'BUSINESS_RISK_PAYS_HIGH', 'BUSINESS_VALEUR_ELEVEE',
        'BIENAYME_CHEBYCHEV_SCORE', 'MIRROR_TEI_SCORE', 'SPECTRAL_CLUSTER_SCORE'
    ]
    
    print("\n🔍 COMPARAISON DES FEATURES IMPORTANTES")
    for feature in important_features:
        val1 = context1.get(feature, 'NON_TROUVÉ')
        val2 = context2.get(feature, 'NON_TROUVÉ')
        print(f"   {feature}:")
        print(f"     Déclaration 1: {val1}")
        print(f"     Déclaration 2: {val2}")
        print(f"     Différent: {val1 != val2}")
    
    # Charger le modèle ML
    print("\n🔍 CHARGEMENT DU MODÈLE ML")
    ml_model_data = load_ml_model('chap30')
    
    if not ml_model_data:
        print("   ❌ Impossible de charger le modèle ML")
        return
    
    print(f"   ✅ Modèle ML chargé: {type(ml_model_data.get('model'))}")
    
    # Créer les DataFrames
    df1 = pd.DataFrame([context1])
    df2 = pd.DataFrame([context2])
    
    print(f"\n🔍 STRUCTURE DES DATAFRAMES")
    print(f"   DataFrame 1: {df1.shape}")
    print(f"   DataFrame 2: {df2.shape}")
    print(f"   Colonnes identiques: {list(df1.columns) == list(df2.columns)}")
    
    # Vérifier les colonnes manquantes
    all_columns = set(df1.columns) | set(df2.columns)
    missing_cols1 = all_columns - set(df1.columns)
    missing_cols2 = all_columns - set(df2.columns)
    
    if missing_cols1:
        print(f"   Colonnes manquantes dans DF1: {missing_cols1}")
    if missing_cols2:
        print(f"   Colonnes manquantes dans DF2: {missing_cols2}")
    
    # Test de prédiction
    print("\n🔍 TEST DE PRÉDICTION")
    try:
        pipeline = ml_model_data.get('model')
        
        # Prédiction pour la déclaration 1
        prob1 = pipeline.predict_proba(df1)[0][1]
        print(f"   Probabilité déclaration 1: {prob1:.6f}")
        
        # Prédiction pour la déclaration 2
        prob2 = pipeline.predict_proba(df2)[0][1]
        print(f"   Probabilité déclaration 2: {prob2:.6f}")
        
        print(f"   Probabilités identiques: {abs(prob1 - prob2) < 1e-10}")
        print(f"   Différence: {abs(prob1 - prob2):.10f}")
        
        # Analyser les features après preprocessing
        print("\n🔍 ANALYSE POST-PREPROCESSING")
        
        # Vérifier si le pipeline a des étapes de preprocessing
        if hasattr(pipeline, 'steps'):
            print(f"   Étapes du pipeline: {[step[0] for step in pipeline.steps]}")
            
            # Appliquer seulement le preprocessing
            preprocessor = pipeline.steps[0][1]  # Premier step
            df1_processed = preprocessor.transform(df1)
            df2_processed = preprocessor.transform(df2)
            
            print(f"   DataFrame 1 après preprocessing: {df1_processed.shape}")
            print(f"   DataFrame 2 après preprocessing: {df2_processed.shape}")
            
            # Comparer les premières valeurs
            print(f"   Premières valeurs DF1: {df1_processed[0][:10]}")
            print(f"   Premières valeurs DF2: {df2_processed[0][:10]}")
            print(f"   Vecteurs identiques: {np.array_equal(df1_processed[0], df2_processed[0])}")
            
    except Exception as e:
        print(f"   ❌ Erreur prédiction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ml_pipeline()


