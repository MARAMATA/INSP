#!/usr/bin/env python3
"""
Script de vérification pour s'assurer que les déclarations conformes ne sont plus toutes marquées comme frauduleuses
"""
import sys
import os
from pathlib import Path

# Ajouter le chemin du backend
backend_path = Path(__file__).parent / "inspectia_app" / "backend"
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(backend_path.parent))

from src.shared.ocr_pipeline import AdvancedOCRPipeline

def create_diverse_normal_declarations(chapter: str) -> list:
    """Créer diverses déclarations normales avec différentes valeurs"""
    
    declarations = []
    
    if chapter == "chap30":
        # Valeurs variées dans l'IQR normal (5M - 127M)
        test_values = [
            10000000.0,   # 10M - Bas IQR
            30000000.0,   # 30M - Proche médiane (22.7M)
            60000000.0,   # 60M - Milieu IQR
            100000000.0,  # 100M - Haut IQR
            120000000.0,  # 120M - Proche Q75
        ]
        
        for i, valeur_caf in enumerate(test_values, 1):
            declarations.append({
                "DECLARATION_ID": f"2024/12A/N{i}",
                "VALEUR_CAF": valeur_caf,
                "POIDS_NET": 500.0 + (i * 100),
                "POIDS_NET_KG": 500.0 + (i * 100),
                "NOMBRE_COLIS": 1 + (i % 3),
                "CODE_PRODUIT_STR": "3004909000",
                "PAYS_ORIGINE_STR": ["FR", "IT", "ES", "DE", "US"][i % 5],
                "REGIME_FISCAL": 0,
                "PRECISION_UEMOA": 90,
                "TAUX_DROITS_PERCENT": 5.0,
                "MONTANT_LIQUIDATION": valeur_caf * 0.05,
            })
    
    elif chapter == "chap84":
        # Valeurs variées dans l'IQR normal (661K - 8.7M)
        test_values = [
            1000000.0,    # 1M - Bas IQR
            3000000.0,    # 3M - Milieu IQR
            5000000.0,    # 5M - Milieu-haut IQR
            7000000.0,    # 7M - Haut IQR
            8500000.0,    # 8.5M - Proche Q75
        ]
        
        for i, valeur_caf in enumerate(test_values, 1):
            declarations.append({
                "DECLARATION_ID": f"2024/12A/N{i}",
                "VALEUR_CAF": valeur_caf,
                "POIDS_NET": 500.0 + (i * 200),
                "POIDS_NET_KG": 500.0 + (i * 200),
                "NOMBRE_COLIS": 1,
                "CODE_PRODUIT_STR": "8471300000",
                "PAYS_ORIGINE_STR": ["CN", "KR", "JP", "DE", "US"][i % 5],
                "REGIME_FISCAL": 0,
                "PRECISION_UEMOA": 90,
                "TAUX_DROITS_PERCENT": 10.0,
                "MONTANT_LIQUIDATION": valeur_caf * 0.10,
            })
    
    elif chapter == "chap85":
        # Valeurs variées dans l'IQR normal (386K - 4.9M)
        test_values = [
            500000.0,     # 500K - Bas IQR
            1500000.0,    # 1.5M - Milieu IQR
            3000000.0,    # 3M - Milieu-haut IQR
            4000000.0,    # 4M - Haut IQR
            4800000.0,    # 4.8M - Proche Q75
        ]
        
        for i, valeur_caf in enumerate(test_values, 1):
            declarations.append({
                "DECLARATION_ID": f"2024/12A/N{i}",
                "VALEUR_CAF": valeur_caf,
                "POIDS_NET": 50.0 + (i * 50),
                "POIDS_NET_KG": 50.0 + (i * 50),
                "NOMBRE_COLIS": 1,
                "CODE_PRODUIT_STR": "8517120000",
                "PAYS_ORIGINE_STR": ["KR", "CN", "JP", "US", "TW"][i % 5],
                "REGIME_FISCAL": 0,
                "PRECISION_UEMOA": 90,
                "TAUX_DROITS_PERCENT": 15.0,
                "MONTANT_LIQUIDATION": valeur_caf * 0.15,
            })
    
    return declarations

def main():
    """Vérifier que les déclarations conformes ne sont pas marquées comme frauduleuses"""
    
    pipeline = AdvancedOCRPipeline()
    chapters = ["chap30", "chap84", "chap85"]
    
    print("=" * 80)
    print("VÉRIFICATION : LES DÉCLARATIONS CONFORMES NE SONT PAS FRAUDULEUSES")
    print("=" * 80)
    print()
    
    total_tested = 0
    total_conformes = 0
    total_frauduleuses = 0
    seuil_optimal = {"chap30": 0.35, "chap84": 0.15, "chap85": 0.20}
    
    for chapter in chapters:
        print(f"\n📊 CHAPITRE {chapter.upper()}")
        print("-" * 80)
        
        declarations = create_diverse_normal_declarations(chapter)
        conformes_chap = 0
        frauduleuses_chap = 0
        
        for decl in declarations:
            total_tested += 1
            
            try:
                result = pipeline.predict_fraud(decl, chapter, level="basic")
                
                fraud_prob = result.get("fraud_probability", 0.0)
                decision = result.get("decision", "unknown")
                is_fraud = result.get("predicted_fraud", False)
                seuil = seuil_optimal.get(chapter, 0.5)
                
                decl_id = decl.get("DECLARATION_ID", "UNKNOWN")
                valeur_caf = decl.get("VALEUR_CAF", 0)
                
                if is_fraud:
                    status = "❌ FRAUDE"
                    frauduleuses_chap += 1
                    total_frauduleuses += 1
                else:
                    status = "✅ CONFORME"
                    conformes_chap += 1
                    total_conformes += 1
                
                # Afficher si proche du seuil
                proximity = ""
                if fraud_prob >= seuil * 0.8 and fraud_prob < seuil:
                    proximity = " ⚠️  (proche seuil)"
                
                print(f"  {decl_id:15s} | CAF: {valeur_caf:12,.0f} | {status:12s} | Prob: {fraud_prob*100:5.2f}% | Seuil: {seuil*100:.0f}%{proximity}")
                
            except Exception as e:
                print(f"  {decl.get('DECLARATION_ID', 'UNKNOWN'):15s} | ❌ ERREUR - {e}")
                frauduleuses_chap += 1
                total_frauduleuses += 1
        
        print(f"\n  📈 Résumé {chapter}:")
        print(f"     ✅ CONFORMES: {conformes_chap}/{len(declarations)} ({conformes_chap*100/len(declarations):.0f}%)")
        print(f"     ❌ FRAUDULEUSES: {frauduleuses_chap}/{len(declarations)} ({frauduleuses_chap*100/len(declarations):.0f}%)")
    
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ GLOBAL")
    print("=" * 80)
    print(f"Total testé: {total_tested}")
    print(f"✅ CONFORMES: {total_conformes}/{total_tested} ({total_conformes*100/total_tested:.1f}%)")
    print(f"❌ FRAUDULEUSES: {total_frauduleuses}/{total_tested} ({total_frauduleuses*100/total_tested:.1f}%)")
    
    if total_conformes >= total_tested * 0.95:
        print("\n✅ SUCCESS: Plus de 95% des déclarations normales sont conformes !")
        print("   Le système fonctionne correctement.")
    elif total_conformes >= total_tested * 0.80:
        print("\n⚠️  ATTENTION: Entre 80% et 95% des déclarations normales sont conformes.")
    else:
        print("\n❌ PROBLÈME: Moins de 80% des déclarations normales sont conformes !")
        print("   Il y a trop de faux positifs.")

if __name__ == "__main__":
    main()

