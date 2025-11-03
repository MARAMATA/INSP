#!/usr/bin/env python3
"""
Script de test pour vérifier que les prédictions retournent des déclarations "conformes"
pour des déclarations normales après la séparation OCR_INGEST / OCR_PIPELINE
"""
import sys
import os
from pathlib import Path

# Ajouter le chemin du backend
backend_path = Path(__file__).parent / "inspectia_app" / "backend"
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(backend_path.parent))

from src.shared.ocr_pipeline import AdvancedOCRPipeline

def create_normal_declaration(chapter: str, index: int = 1) -> dict:
    """Créer une déclaration normale (valeurs dans l'IQR)"""
    
    if chapter == "chap30":
        # Valeurs normales pour chap30 (dans l'IQR)
        return {
            "DECLARATION_ID": f"2024/12A/{1000 + index}",
            "ANNEE": 2024,
            "BUREAU": "12A",
            "NUMERO": 1000 + index,
            "VALEUR_CAF": 20000000.0,  # Entre Q25 (5M) et Q75 (127M) - NORMAL
            "POIDS_NET": 500.0,  # Poids normal
            "POIDS_NET_KG": 500.0,
            "NOMBRE_COLIS": 2,
            "CODE_PRODUIT_STR": "3004909000",  # Code médicament chap30
            "PAYS_ORIGINE_STR": "FR",  # Pays normal
            "REGIME_FISCAL": 0,  # Régime normal
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 5.0,
            "MONTANT_LIQUIDATION": 1000000.0,
        }
    
    elif chapter == "chap84":
        # Valeurs normales pour chap84 (dans l'IQR)
        return {
            "DECLARATION_ID": f"2024/12A/{2000 + index}",
            "ANNEE": 2024,
            "BUREAU": "12A",
            "NUMERO": 2000 + index,
            "VALEUR_CAF": 3000000.0,  # Entre Q25 (661K) et Q75 (8.7M) - NORMAL
            "POIDS_NET": 1000.0,
            "POIDS_NET_KG": 1000.0,
            "NOMBRE_COLIS": 1,
            "CODE_PRODUIT_STR": "8471300000",  # Code machine chap84
            "PAYS_ORIGINE_STR": "CN",  # Pays normal
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 10.0,
            "MONTANT_LIQUIDATION": 300000.0,
        }
    
    elif chapter == "chap85":
        # Valeurs normales pour chap85 (dans l'IQR)
        return {
            "DECLARATION_ID": f"2024/12A/{3000 + index}",
            "ANNEE": 2024,
            "BUREAU": "12A",
            "NUMERO": 3000 + index,
            "VALEUR_CAF": 1500000.0,  # Entre Q25 (386K) et Q75 (4.9M) - NORMAL
            "POIDS_NET": 100.0,
            "POIDS_NET_KG": 100.0,
            "NOMBRE_COLIS": 1,
            "CODE_PRODUIT_STR": "8517120000",  # Code électronique chap85
            "PAYS_ORIGINE_STR": "KR",  # Pays normal
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 15.0,
            "MONTANT_LIQUIDATION": 225000.0,
        }
    
    return {}

def main():
    """Test des prédictions pour des déclarations normales"""
    
    pipeline = AdvancedOCRPipeline()
    chapters = ["chap30", "chap84", "chap85"]
    
    print("=" * 80)
    print("TEST DES PRÉDICTIONS POUR DÉCLARATIONS NORMALES (CONFORMES)")
    print("=" * 80)
    print()
    
    total_conformes = 0
    total_fraudes = 0
    
    for chapter in chapters:
        print(f"\n📊 CHAPITRE {chapter.upper()}")
        print("-" * 80)
        
        conformes_chap = 0
        fraudes_chap = 0
        
        # Tester 10 déclarations normales par chapitre
        for i in range(1, 11):
            decl = create_normal_declaration(chapter, i)
            
            try:
                result = pipeline.predict_fraud(decl, chapter, level="basic")
                
                fraud_prob = result.get("fraud_probability", 0.0)
                decision = result.get("decision", "unknown")
                is_fraud = result.get("predicted_fraud", False)
                
                status = "✅ CONFORME" if not is_fraud else "❌ FRAUDE"
                
                if not is_fraud:
                    conformes_chap += 1
                    total_conformes += 1
                else:
                    fraudes_chap += 1
                    total_fraudes += 1
                
                print(f"  Déclaration {i:2d}: {status:12s} | Prob: {fraud_prob*100:5.2f}% | Décision: {decision}")
                
            except Exception as e:
                print(f"  Déclaration {i:2d}: ❌ ERREUR - {e}")
                fraudes_chap += 1
                total_fraudes += 1
        
        print(f"\n  📈 Résumé {chapter}: {conformes_chap}/10 conformes ({conformes_chap*10}%), {fraudes_chap}/10 fraudes ({fraudes_chap*10}%)")
    
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ GLOBAL")
    print("=" * 80)
    total = total_conformes + total_fraudes
    if total > 0:
        pct_conformes = (total_conformes / total) * 100
        pct_fraudes = (total_fraudes / total) * 100
        print(f"✅ CONFORMES: {total_conformes}/{total} ({pct_conformes:.1f}%)")
        print(f"❌ FRAUDES:   {total_fraudes}/{total} ({pct_fraudes:.1f}%)")
        
        if pct_conformes >= 80:
            print("\n✅ SUCCESS: Plus de 80% des déclarations normales sont conformes !")
        elif pct_conformes >= 50:
            print("\n⚠️  ATTENTION: Entre 50% et 80% des déclarations normales sont conformes.")
        else:
            print("\n❌ PROBLÈME: Moins de 50% des déclarations normales sont conformes !")
    else:
        print("❌ AUCUNE PRÉDICTION RÉUSSIE")

if __name__ == "__main__":
    main()

