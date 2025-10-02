#!/usr/bin/env python3
"""
Script pour régénérer les splits train/valid/test pour tous les chapitres
"""

import sys
import os
sys.path.append('src')

from chapters.chap30.ml_model import Chap30SupervisedML
from chapters.chap84.ml_model import Chap84SupervisedML
from chapters.chap85.ml_model import Chap85SupervisedML

def regenerate_splits_for_chapter(chapter_class, chapter_name):
    """Régénère les splits pour un chapitre donné"""
    print(f"\n🔄 Régénération des splits pour {chapter_name}...")
    
    try:
        # Initialiser le modèle
        ml_model = chapter_class()
        
        # Charger les données
        print(f"   📊 Chargement des données...")
        df = ml_model.load_data()
        print(f"   ✅ Données chargées: {df.shape}")
        
        # Vérifier si les splits existent déjà
        if ml_model._splits_exist():
            print(f"   ⚠️ Splits existants détectés, suppression...")
            import shutil
            shutil.rmtree(ml_model.splits_dir)
            ml_model.splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Régénérer les splits
        print(f"   🔄 Régénération des splits...")
        X_train, X_valid, X_test, y_train, y_valid, y_test = ml_model.split_data_robust(df)
        
        # Vérifier les tailles
        print(f"   📊 Tailles des splits:")
        print(f"      - Train: {X_train.shape[0]} échantillons")
        print(f"      - Valid: {X_valid.shape[0]} échantillons") 
        print(f"      - Test:  {X_test.shape[0]} échantillons")
        print(f"      - Total: {X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]} échantillons")
        
        # Vérifier les proportions de classes
        print(f"   📊 Proportions des classes:")
        print(f"      - Train: {y_train.value_counts(normalize=True).round(3).to_dict()}")
        print(f"      - Valid: {y_valid.value_counts(normalize=True).round(3).to_dict()}")
        print(f"      - Test:  {y_test.value_counts(normalize=True).round(3).to_dict()}")
        
        print(f"   ✅ Splits régénérés avec succès pour {chapter_name}")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur lors de la régénération des splits pour {chapter_name}: {e}")
        return False

def main():
    """Fonction principale"""
    print("🚀 RÉGÉNÉRATION DES SPLITS POUR TOUS LES CHAPITRES")
    print("=" * 60)
    
    chapters = [
        (Chap30SupervisedML, "Chapitre 30"),
        (Chap84SupervisedML, "Chapitre 84"), 
        (Chap85SupervisedML, "Chapitre 85")
    ]
    
    success_count = 0
    total_count = len(chapters)
    
    for chapter_class, chapter_name in chapters:
        if regenerate_splits_for_chapter(chapter_class, chapter_name):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 RÉSULTATS: {success_count}/{total_count} chapitres traités avec succès")
    
    if success_count == total_count:
        print("✅ Tous les splits ont été régénérés avec succès!")
    else:
        print("⚠️ Certains chapitres ont échoué")
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    exit(main())
