#!/usr/bin/env python3
"""
Script de test pour vérifier les prédictions avec un mélange de déclarations normales et suspectes
"""
import sys
import os
from pathlib import Path

# Ajouter le chemin du backend
backend_path = Path(__file__).parent / "inspectia_app" / "backend"
sys.path.insert(0, str(backend_path))
sys.path.insert(0, str(backend_path.parent))

from src.shared.ocr_pipeline import AdvancedOCRPipeline

def create_test_declarations(chapter: str) -> list:
    """Créer un mélange de déclarations normales et suspectes pour un chapitre"""
    
    declarations = []
    
    if chapter == "chap30":
        # === DÉCLARATIONS NORMALES (5) ===
        for i in range(1, 6):
            declarations.append({
                "type": "normale",
                "DECLARATION_ID": f"2024/12A/N{i}",
                "ANNEE": 2024,
                "BUREAU": "12A",
                "NUMERO": 100 + i,
                "VALEUR_CAF": 20000000.0,  # Dans l'IQR (5M-127M)
                "POIDS_NET": 500.0,
                "POIDS_NET_KG": 500.0,
                "NOMBRE_COLIS": 2,
                "CODE_PRODUIT_STR": "3004909000",
                "PAYS_ORIGINE_STR": "FR",
                "REGIME_FISCAL": 0,
                "PRECISION_UEMOA": 90,
                "TAUX_DROITS_PERCENT": 5.0,
                "MONTANT_LIQUIDATION": 1000000.0,
            })
        
        # === DÉCLARATIONS SUSPECTES/ FRAUDULEUSES (5) ===
        # 1. Valeur exceptionnellement élevée
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F1",
            "ANNEE": 2024,
            "BUREAU": "12A",
            "NUMERO": 200,
            "VALEUR_CAF": 200000000.0,  # Très élevée (>Q75=127M)
            "POIDS_NET": 100.0,  # Poids faible pour cette valeur
            "POIDS_NET_KG": 100.0,
            "NOMBRE_COLIS": 1,
            "CODE_PRODUIT_STR": "3004909000",
            "PAYS_ORIGINE_STR": "CN",  # Pays à risque
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 5.0,
            "MONTANT_LIQUIDATION": 10000000.0,
        })
        
        # 2. Ratio poids/valeur très suspect
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F2",
            "ANNEE": 2024,
            "BUREAU": "12A",
            "NUMERO": 201,
            "VALEUR_CAF": 50000000.0,
            "POIDS_NET": 0.001,  # Poids extrêmement faible
            "POIDS_NET_KG": 0.001,
            "NOMBRE_COLIS": 1,
            "CODE_PRODUIT_STR": "3004909000",
            "PAYS_ORIGINE_STR": "FR",
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 5.0,
            "MONTANT_LIQUIDATION": 2500000.0,
        })
        
        # 3. Code produit incorrect (glissement)
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F3",
            "ANNEE": 2024,
            "BUREAU": "12A",
            "NUMERO": 202,
            "VALEUR_CAF": 100000000.0,
            "POIDS_NET": 2000.0,
            "POIDS_NET_KG": 2000.0,
            "NOMBRE_COLIS": 5,
            "CODE_PRODUIT_STR": "3306100000",  # Code cosmétique au lieu de médicament
            "PAYS_ORIGINE_STR": "FR",
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 5.0,
            "MONTANT_LIQUIDATION": 5000000.0,
        })
        
        # 4. Pays à risque avec valeur élevée
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F4",
            "ANNEE": 2024,
            "BUREAU": "12A",
            "NUMERO": 203,
            "VALEUR_CAF": 150000000.0,
            "POIDS_NET": 500.0,
            "POIDS_NET_KG": 500.0,
            "NOMBRE_COLIS": 1,
            "CODE_PRODUIT_STR": "3004909000",
            "PAYS_ORIGINE_STR": "IN",  # Pays à risque
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 5.0,
            "MONTANT_LIQUIDATION": 7500000.0,
        })
        
        # 5. Valeur très élevée avec taux droits suspects
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F5",
            "ANNEE": 2024,
            "BUREAU": "12A",
            "NUMERO": 204,
            "VALEUR_CAF": 180000000.0,
            "POIDS_NET": 300.0,
            "POIDS_NET_KG": 300.0,
            "NOMBRE_COLIS": 1,
            "CODE_PRODUIT_STR": "3004909000",
            "PAYS_ORIGINE_STR": "CN",
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 25.0,  # Taux très élevé
            "MONTANT_LIQUIDATION": 45000000.0,
        })
    
    elif chapter == "chap84":
        # === DÉCLARATIONS NORMALES (5) ===
        for i in range(1, 6):
            declarations.append({
                "type": "normale",
                "DECLARATION_ID": f"2024/12A/N{i}",
                "ANNEE": 2024,
                "BUREAU": "12A",
                "NUMERO": 100 + i,
                "VALEUR_CAF": 3000000.0,  # Dans l'IQR (661K-8.7M)
                "POIDS_NET": 1000.0,
                "POIDS_NET_KG": 1000.0,
                "NOMBRE_COLIS": 1,
                "CODE_PRODUIT_STR": "8471300000",
                "PAYS_ORIGINE_STR": "CN",
                "REGIME_FISCAL": 0,
                "PRECISION_UEMOA": 90,
                "TAUX_DROITS_PERCENT": 10.0,
                "MONTANT_LIQUIDATION": 300000.0,
            })
        
        # === DÉCLARATIONS SUSPECTES/ FRAUDULEUSES (5) ===
        # 1. Valeur exceptionnellement élevée
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F1",
            "VALEUR_CAF": 150000000.0,  # Très élevée (>Q75=8.7M)
            "POIDS_NET": 100.0,
            "POIDS_NET_KG": 100.0,
            "CODE_PRODUIT_STR": "8471300000",
            "PAYS_ORIGINE_STR": "CN",
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 10.0,
            "MONTANT_LIQUIDATION": 15000000.0,
        })
        
        # 2. Code produit incorrect
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F2",
            "VALEUR_CAF": 80000000.0,
            "POIDS_NET": 5000.0,
            "POIDS_NET_KG": 5000.0,
            "CODE_PRODUIT_STR": "8517120000",  # Code électronique au lieu de machine
            "PAYS_ORIGINE_STR": "CN",
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 10.0,
            "MONTANT_LIQUIDATION": 8000000.0,
        })
        
        # 3. Ratio suspect
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F3",
            "VALEUR_CAF": 50000000.0,
            "POIDS_NET": 0.01,  # Poids extrêmement faible
            "POIDS_NET_KG": 0.01,
            "CODE_PRODUIT_STR": "8471300000",
            "PAYS_ORIGINE_STR": "CN",
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 10.0,
            "MONTANT_LIQUIDATION": 5000000.0,
        })
        
        # 4. Valeur élevée avec pays à risque
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F4",
            "VALEUR_CAF": 120000000.0,
            "POIDS_NET": 2000.0,
            "POIDS_NET_KG": 2000.0,
            "CODE_PRODUIT_STR": "8471300000",
            "PAYS_ORIGINE_STR": "PK",  # Pays à risque
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 10.0,
            "MONTANT_LIQUIDATION": 12000000.0,
        })
        
        # 5. Machine avec valeur unitaire suspecte
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F5",
            "VALEUR_CAF": 100000000.0,
            "POIDS_NET": 50.0,  # Poids faible
            "POIDS_NET_KG": 50.0,
            "CODE_PRODUIT_STR": "8471300000",
            "PAYS_ORIGINE_STR": "CN",
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 25.0,  # Taux élevé
            "MONTANT_LIQUIDATION": 25000000.0,
        })
    
    elif chapter == "chap85":
        # === DÉCLARATIONS NORMALES (5) ===
        for i in range(1, 6):
            declarations.append({
                "type": "normale",
                "DECLARATION_ID": f"2024/12A/N{i}",
                "ANNEE": 2024,
                "BUREAU": "12A",
                "NUMERO": 100 + i,
                "VALEUR_CAF": 1500000.0,  # Dans l'IQR (386K-4.9M)
                "POIDS_NET": 100.0,
                "POIDS_NET_KG": 100.0,
                "NOMBRE_COLIS": 1,
                "CODE_PRODUIT_STR": "8517120000",
                "PAYS_ORIGINE_STR": "KR",
                "REGIME_FISCAL": 0,
                "PRECISION_UEMOA": 90,
                "TAUX_DROITS_PERCENT": 15.0,
                "MONTANT_LIQUIDATION": 225000.0,
            })
        
        # === DÉCLARATIONS SUSPECTES/ FRAUDULEUSES (5) ===
        # 1. Valeur exceptionnellement élevée
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F1",
            "VALEUR_CAF": 150000000.0,  # Très élevée (>Q75=4.9M)
            "POIDS_NET": 10.0,  # Poids très faible
            "POIDS_NET_KG": 10.0,
            "CODE_PRODUIT_STR": "8517120000",
            "PAYS_ORIGINE_STR": "CN",
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 15.0,
            "MONTANT_LIQUIDATION": 22500000.0,
        })
        
        # 2. Code produit incorrect
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F2",
            "VALEUR_CAF": 80000000.0,
            "POIDS_NET": 50.0,
            "POIDS_NET_KG": 50.0,
            "CODE_PRODUIT_STR": "8471300000",  # Code machine au lieu d'électronique
            "PAYS_ORIGINE_STR": "CN",
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 15.0,
            "MONTANT_LIQUIDATION": 12000000.0,
        })
        
        # 3. Ratio poids/valeur extrême
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F3",
            "VALEUR_CAF": 50000000.0,
            "POIDS_NET": 0.001,  # Poids extrêmement faible
            "POIDS_NET_KG": 0.001,
            "CODE_PRODUIT_STR": "8517120000",
            "PAYS_ORIGINE_STR": "KR",
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 15.0,
            "MONTANT_LIQUIDATION": 7500000.0,
        })
        
        # 4. Électronique avec pays à risque
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F4",
            "VALEUR_CAF": 120000000.0,
            "POIDS_NET": 20.0,
            "POIDS_NET_KG": 20.0,
            "CODE_PRODUIT_STR": "8517120000",
            "PAYS_ORIGINE_STR": "PK",  # Pays à risque
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 15.0,
            "MONTANT_LIQUIDATION": 18000000.0,
        })
        
        # 5. Valeur élevée avec taux droits suspects
        declarations.append({
            "type": "fraude",
            "DECLARATION_ID": f"2024/12A/F5",
            "VALEUR_CAF": 100000000.0,
            "POIDS_NET": 5.0,  # Poids très faible
            "POIDS_NET_KG": 5.0,
            "CODE_PRODUIT_STR": "8517120000",
            "PAYS_ORIGINE_STR": "CN",
            "REGIME_FISCAL": 0,
            "PRECISION_UEMOA": 90,
            "TAUX_DROITS_PERCENT": 30.0,  # Taux très élevé
            "MONTANT_LIQUIDATION": 30000000.0,
        })
    
    return declarations

def main():
    """Test des prédictions avec mélange normal/fraude"""
    
    pipeline = AdvancedOCRPipeline()
    chapters = ["chap30", "chap84", "chap85"]
    
    print("=" * 80)
    print("TEST DES PRÉDICTIONS - MÉLANGE DÉCLARATIONS NORMALES ET FRAUDULEUSES")
    print("=" * 80)
    print()
    
    total_conformes = 0
    total_fraudes = 0
    total_normales_testees = 0
    total_fraudes_testees = 0
    conformes_corrects = 0
    fraudes_detectees = 0
    
    for chapter in chapters:
        print(f"\n📊 CHAPITRE {chapter.upper()}")
        print("-" * 80)
        
        declarations = create_test_declarations(chapter)
        conformes_chap = 0
        fraudes_chap = 0
        normales_testees_chap = 0
        fraudes_testees_chap = 0
        
        for i, decl in enumerate(declarations, 1):
            decl_type = decl.pop("type")
            decl_id = decl.get("DECLARATION_ID", f"DECL_{i}")
            
            try:
                result = pipeline.predict_fraud(decl, chapter, level="basic")
                
                fraud_prob = result.get("fraud_probability", 0.0)
                decision = result.get("decision", "unknown")
                is_fraud = result.get("predicted_fraud", False)
                
                # Vérifier si la prédiction est correcte
                if decl_type == "normale":
                    normales_testees_chap += 1
                    total_normales_testees += 1
                    if not is_fraud:
                        conformes_corrects += 1
                        status = "✅ CONFORME (CORRECT)"
                        conformes_chap += 1
                        total_conformes += 1
                    else:
                        status = "❌ FRAUDE (FAUX POSITIF)"
                        fraudes_chap += 1
                        total_fraudes += 1
                else:  # fraude
                    fraudes_testees_chap += 1
                    total_fraudes_testees += 1
                    if is_fraud:
                        fraudes_detectees += 1
                        status = "✅ FRAUDE (DÉTECTÉE)"
                        fraudes_chap += 1
                        total_fraudes += 1
                    else:
                        status = "❌ CONFORME (FAUX NÉGATIF)"
                        conformes_chap += 1
                        total_conformes += 1
                
                print(f"  {decl_id:15s} [{decl_type:7s}] → {status:25s} | Prob: {fraud_prob*100:5.2f}% | Décision: {decision}")
                
            except Exception as e:
                print(f"  {decl_id:15s} [{decl_type:7s}] → ❌ ERREUR - {e}")
        
        # Statistiques par chapitre
        print(f"\n  📈 Statistiques {chapter}:")
        print(f"     - Normales testées: {normales_testees_chap} → {conformes_chap - (normales_testees_chap - sum(1 for d in declarations if d.get('type') == 'normale'))} conformes détectées, {fraudes_chap - (sum(1 for d in declarations if d.get('type') == 'fraude'))} faux positifs")
        print(f"     - Frauduleuses testées: {fraudes_testees_chap} → {fraudes_chap - (sum(1 for d in declarations if d.get('type') == 'normale' and not result.get('predicted_fraud', False))) if fraudes_testees_chap > 0 else 0} fraudes détectées")
    
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ GLOBAL")
    print("=" * 80)
    
    if total_normales_testees > 0:
        pct_conformes_corrects = (conformes_corrects / total_normales_testees) * 100
        print(f"✅ DÉCLARATIONS NORMALES:")
        print(f"   - Testées: {total_normales_testees}")
        print(f"   - Correctement identifiées comme conformes: {conformes_corrects} ({pct_conformes_corrects:.1f}%)")
        print(f"   - Faux positifs (marquées fraudes): {total_normales_testees - conformes_corrects} ({(100-pct_conformes_corrects):.1f}%)")
    
    if total_fraudes_testees > 0:
        pct_fraudes_detectees = (fraudes_detectees / total_fraudes_testees) * 100
        print(f"\n❌ DÉCLARATIONS FRAUDULEUSES:")
        print(f"   - Testées: {total_fraudes_testees}")
        print(f"   - Correctement détectées: {fraudes_detectees} ({pct_fraudes_detectees:.1f}%)")
        print(f"   - Faux négatifs (marquées conformes): {total_fraudes_testees - fraudes_detectees} ({(100-pct_fraudes_detectees):.1f}%)")
    
    total_testees = total_normales_testees + total_fraudes_testees
    total_correctes = conformes_corrects + fraudes_detectees
    if total_testees > 0:
        pct_precision = (total_correctes / total_testees) * 100
        print(f"\n🎯 PRÉCISION GLOBALE: {total_correctes}/{total_testees} ({pct_precision:.1f}%)")

if __name__ == "__main__":
    main()

