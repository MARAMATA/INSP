#!/usr/bin/env python3
"""
Script pour analyser les données de tous les chapitres et calculer les vrais seuils
basés sur les données réelles et les flags de fraude.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

def analyze_chapter_data(chapter: str) -> dict:
    """Analyser les données d'un chapitre pour calculer les seuils dynamiques"""
    
    print(f"\n🔍 ANALYSE DU CHAPITRE {chapter.upper()}")
    print("=" * 50)
    
    # Charger les données d'entraînement
    X_path = f"data/ml_splits/{chapter}/X_train.csv"
    y_path = f"data/ml_splits/{chapter}/y_train.csv"
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        print(f"❌ Fichiers non trouvés pour {chapter}")
        return {}
    
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    
    # Fusionner les données
    data = pd.concat([X, y], axis=1)
    
    print(f"📊 Données chargées: {len(data)} déclarations")
    print(f"🎯 Taux de fraude: {y['FRAUD_FLAG'].mean():.2%}")
    
    # Séparer les cas conformes et frauduleux
    conformes = data[data['FRAUD_FLAG'] == 0]
    frauduleux = data[data['FRAUD_FLAG'] == 1]
    
    print(f"✅ Cas conformes: {len(conformes)} ({len(conformes)/len(data):.1%})")
    print(f"❌ Cas frauduleux: {len(frauduleux)} ({len(frauduleux)/len(data):.1%})")
    
    # Calculer les seuils basés sur les cas conformes
    thresholds = {}
    
    # Colonnes numériques à analyser
    numeric_cols = ['POIDS_NET_KG', 'NOMBRE_COLIS', 'QUANTITE_COMPLEMENT', 'TAUX_DROITS_PERCENT']
    
    for col in numeric_cols:
        if col in conformes.columns:
            values_conformes = conformes[col].dropna()
            values_frauduleux = frauduleux[col].dropna() if len(frauduleux) > 0 else pd.Series()
            
            if len(values_conformes) > 0:
                # Seuils basés sur les percentiles des cas conformes
                thresholds[f'{col.lower()}_eleve'] = values_conformes.quantile(0.85)
                thresholds[f'{col.lower()}_tres_eleve'] = values_conformes.quantile(0.95)
                thresholds[f'{col.lower()}_exceptionnel'] = values_conformes.quantile(0.99)
                
                # Statistiques descriptives
                print(f"\n📈 {col}:")
                print(f"   Conformes - Min: {values_conformes.min():.2f}, Max: {values_conformes.max():.2f}")
                print(f"   Conformes - P85: {thresholds[f'{col.lower()}_eleve']:.2f}, P95: {thresholds[f'{col.lower()}_tres_eleve']:.2f}, P99: {thresholds[f'{col.lower()}_exceptionnel']:.2f}")
                
                if len(values_frauduleux) > 0:
                    print(f"   Frauduleux - Min: {values_frauduleux.min():.2f}, Max: {values_frauduleux.max():.2f}")
                    print(f"   Frauduleux - P85: {values_frauduleux.quantile(0.85):.2f}, P95: {values_frauduleux.quantile(0.95):.2f}")
    
    # Analyser les ratios pour détecter les anomalies
    if 'POIDS_NET_KG' in data.columns and 'NOMBRE_COLIS' in data.columns:
        data['RATIO_COLIS_POIDS'] = data['NOMBRE_COLIS'] / data['POIDS_NET_KG'].replace(0, 1)
        conformes_ratio = data[data['FRAUD_FLAG'] == 0]['RATIO_COLIS_POIDS'].dropna()
        if len(conformes_ratio) > 0:
            thresholds['ratio_colis_poids_anormal'] = conformes_ratio.quantile(0.95)
            thresholds['ratio_colis_poids_tres_anormal'] = conformes_ratio.quantile(0.99)
            print(f"\n📊 RATIO_COLIS_POIDS:")
            print(f"   Conformes - P95: {thresholds['ratio_colis_poids_anormal']:.4f}, P99: {thresholds['ratio_colis_poids_tres_anormal']:.4f}")
    
    if 'POIDS_NET_KG' in data.columns and 'QUANTITE_COMPLEMENT' in data.columns:
        data['RATIO_QUANTITE_POIDS'] = data['QUANTITE_COMPLEMENT'] / data['POIDS_NET_KG'].replace(0, 1)
        conformes_ratio = data[data['FRAUD_FLAG'] == 0]['RATIO_QUANTITE_POIDS'].dropna()
        if len(conformes_ratio) > 0:
            thresholds['ratio_quantite_poids_anormal'] = conformes_ratio.quantile(0.95)
            thresholds['ratio_quantite_poids_tres_anormal'] = conformes_ratio.quantile(0.99)
            print(f"\n📊 RATIO_QUANTITE_POIDS:")
            print(f"   Conformes - P95: {thresholds['ratio_quantite_poids_anormal']:.4f}, P99: {thresholds['ratio_quantite_poids_tres_anormal']:.4f}")
    
    # Analyser les codes SH pour détecter les patterns
    if 'CODE_SH_COMPLET' in data.columns:
        sh_conformes = conformes['CODE_SH_COMPLET'].value_counts().head(10)
        sh_frauduleux = frauduleux['CODE_SH_COMPLET'].value_counts().head(10) if len(frauduleux) > 0 else pd.Series()
        
        print(f"\n🏷️ CODES SH LES PLUS FRÉQUENTS:")
        print(f"   Conformes: {sh_conformes.index.tolist()[:5]}")
        if len(sh_frauduleux) > 0:
            print(f"   Frauduleux: {sh_frauduleux.index.tolist()[:5]}")
    
    # Analyser les pays d'origine
    if 'CODE_PAYS_ORIGINE' in data.columns:
        pays_conformes = conformes['CODE_PAYS_ORIGINE'].value_counts().head(10)
        pays_frauduleux = frauduleux['CODE_PAYS_ORIGINE'].value_counts().head(10) if len(frauduleux) > 0 else pd.Series()
        
        print(f"\n🌍 PAYS D'ORIGINE LES PLUS FRÉQUENTS:")
        print(f"   Conformes: {pays_conformes.index.tolist()[:5]}")
        if len(pays_frauduleux) > 0:
            print(f"   Frauduleux: {pays_frauduleux.index.tolist()[:5]}")
    
    return thresholds

def main():
    """Analyser tous les chapitres et générer les seuils dynamiques"""
    
    print("🚀 ANALYSE DES DONNÉES POUR CALCULER LES SEUILS DYNAMIQUES")
    print("=" * 70)
    
    chapters = ['chap30', 'chap84', 'chap85']
    all_thresholds = {}
    
    for chapter in chapters:
        thresholds = analyze_chapter_data(chapter)
        if thresholds:
            all_thresholds[chapter] = thresholds
    
    # Sauvegarder les seuils calculés
    output_file = "data/dynamic_thresholds.json"
    with open(output_file, 'w') as f:
        json.dump(all_thresholds, f, indent=2, default=str)
    
    print(f"\n✅ SEUILS DYNAMIQUES SAUVEGARDÉS DANS: {output_file}")
    print("\n📋 RÉSUMÉ DES SEUILS CALCULÉS:")
    
    for chapter, thresholds in all_thresholds.items():
        print(f"\n{chapter.upper()}:")
        for key, value in thresholds.items():
            print(f"  {key}: {value:.2f}")
    
    print(f"\n🎯 UTILISATION:")
    print(f"   Les seuils sont maintenant basés sur les données réelles")
    print(f"   et les flags de fraude de chaque chapitre.")
    print(f"   Ils remplacent les seuils fixes arbitraires.")

if __name__ == "__main__":
    main()
