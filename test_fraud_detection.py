#!/usr/bin/env python3
"""
Test de détection de fraude - Vérifier que les corrections fonctionnent
"""

import sys
import os
from pathlib import Path

# Ajouter le backend au path
backend_path = Path(__file__).parent / "inspectia_app" / "backend"
sys.path.insert(0, str(backend_path))

from src.shared.ocr_pipeline import AdvancedOCRPipeline
from src.shared.ocr_ingest import create_advanced_context_from_ocr_data
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_declaration(valeur_caf, poids_net, code_produit, pays_origine, pays_provenance, bureau, regime_fiscal, numero_dpi, chapitre="chap30"):
    """Tester une déclaration"""
    
    # Créer les données OCR simulées
    ocr_data = {
        "valeur_caf": valeur_caf,
        "valeur_douane": valeur_caf * 1.0,
        "montant_liquidation": valeur_caf * 0.05,
        "poids_net": poids_net,
        "code_sh_complet": code_produit,
        "code_produit": code_produit,
        "pays_origine": pays_origine,
        "pays_provenance": pays_provenance,
        "bureau": bureau,
        "regime_fiscal": regime_fiscal,
        "numero_dpi": numero_dpi,
        "numero_article": 1,
        "precision_uemoa": 0,
    }
    
    # Créer le contexte avancé
    context = create_advanced_context_from_ocr_data(ocr_data, chapitre)
    
    # Initialiser le pipeline
    pipeline = AdvancedOCRPipeline()
    
    # Faire la prédiction
    try:
        result = pipeline.predict_fraud(ocr_data, chapter=chapitre, level="basic")
        
        fraud_prob = result.get("fraud_probability", 0.0)
        decision = result.get("decision", "unknown")
        
        return {
            "fraud_probability": fraud_prob,
            "decision": decision,
            "success": True,
            "context": context
        }
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return {
            "fraud_probability": 0.0,
            "decision": "ERROR",
            "success": False,
            "error": str(e)
        }

def main():
    """Test principal"""
    
    print("=" * 80)
    print("🧪 TEST DE DÉTECTION DE FRAUDE")
    print("=" * 80)
    print("\n📋 Objectif: Vérifier que les déclarations ne sont pas toutes détectées comme fraudes")
    print("   et que les probabilités sont réalistes.\n")
    
    # Liste de déclarations de test variées
    test_declarations = [
        # Déclaration 1: Normale - petite valeur, pays d'origine normal
        {
            "nom": "Déclaration normale (petite valeur)",
            "valeur_caf": 5000000,
            "poids_net": 500,
            "code_produit": "300490",
            "pays_origine": "FR",
            "pays_provenance": "FR",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        
        # Déclaration 2: Normale - valeur moyenne
        {
            "nom": "Déclaration normale (valeur moyenne)",
            "valeur_caf": 15000000,
            "poids_net": 1000,
            "code_produit": "300490",
            "pays_origine": "DE",
            "pays_provenance": "DE",
            "bureau": "10S",
            "regime_fiscal": 2,
            "numero_dpi": "SANS_DPI"
        },
        
        # Déclaration 3: Suspecte - très haute valeur
        {
            "nom": "Déclaration suspecte (très haute valeur)",
            "valeur_caf": 150000000,
            "poids_net": 5000,
            "code_produit": "300490",
            "pays_origine": "CN",
            "pays_provenance": "CN",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        
        # Déclaration 4: Suspecte - glissement tarifaire (code ne commence pas par 30)
        {
            "nom": "Déclaration suspecte (glissement tarifaire)",
            "valeur_caf": 20000000,
            "poids_net": 800,
            "code_produit": "330410",  # Code cosmétique, pas pharmaceutique
            "pays_origine": "FR",
            "pays_provenance": "FR",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        
        # Déclaration 5: Normale - valeur élevée mais avec DPI
        {
            "nom": "Déclaration normale (valeur élevée avec DPI)",
            "valeur_caf": 80000000,
            "poids_net": 2000,
            "code_produit": "300490",
            "pays_origine": "SN",
            "pays_provenance": "SN",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "DPI123456"
        },
        
        # Déclaration 6: Suspecte - pays à risque
        {
            "nom": "Déclaration suspecte (pays à risque)",
            "valeur_caf": 25000000,
            "poids_net": 1200,
            "code_produit": "300490",
            "pays_origine": "PK",  # Pays à risque
            "pays_provenance": "PK",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        
        # Déclaration 7: Normale - très petite valeur
        {
            "nom": "Déclaration normale (très petite valeur)",
            "valeur_caf": 1000000,
            "poids_net": 100,
            "code_produit": "300490",
            "pays_origine": "SN",
            "pays_provenance": "SN",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
    ]
    
    results = []
    
    for i, decl in enumerate(test_declarations, 1):
        print(f"\n{'='*80}")
        print(f"📦 Test {i}/{len(test_declarations)}: {decl['nom']}")
        print(f"{'='*80}")
        
        result = test_declaration(
            valeur_caf=decl["valeur_caf"],
            poids_net=decl["poids_net"],
            code_produit=decl["code_produit"],
            pays_origine=decl["pays_origine"],
            pays_provenance=decl["pays_provenance"],
            bureau=decl["bureau"],
            regime_fiscal=decl["regime_fiscal"],
            numero_dpi=decl["numero_dpi"],
            chapitre="chap30"
        )
        
        if result["success"]:
            fraud_prob = result["fraud_probability"]
            decision = result["decision"]
            
            print(f"✅ Prédiction réussie")
            print(f"   Probabilité de fraude: {fraud_prob:.4f} ({fraud_prob*100:.2f}%)")
            print(f"   Décision: {decision}")
            
            # Afficher quelques features importantes du contexte
            context = result.get("context", {})
            print(f"\n   Features importantes:")
            print(f"   - VALEUR_CAF: {context.get('VALEUR_CAF', 0):,.0f}")
            print(f"   - POIDS_NET: {context.get('POIDS_NET', 0):,.2f}")
            print(f"   - CODE_PRODUIT_STR: {context.get('CODE_PRODUIT_STR', '')}")
            print(f"   - REGIME_FISCAL: {context.get('REGIME_FISCAL', 0)}")
            
            # Compter les features business actives
            business_features = [k for k in context.keys() if k.startswith('BUSINESS_')]
            active = sum(1 for k in business_features if context.get(k, 0) > 0)
            print(f"   - Features business actives: {active}/{len(business_features)}")
            
            # Afficher quelques fraud scores
            fraud_scores = {
                'BIENAYME_CHEBYCHEV_SCORE': context.get('BIENAYME_CHEBYCHEV_SCORE', 0),
                'MIRROR_TEI_SCORE': context.get('MIRROR_TEI_SCORE', 0),
                'COMPOSITE_FRAUD_SCORE': context.get('COMPOSITE_FRAUD_SCORE', 0),
            }
            print(f"   - Fraud scores:")
            for score_name, score_value in fraud_scores.items():
                if score_value != 0.0:
                    print(f"     * {score_name}: {score_value:.6f}")
            
            results.append({
                "nom": decl["nom"],
                "fraud_probability": fraud_prob,
                "decision": decision,
                "success": True
            })
        else:
            print(f"❌ Erreur: {result.get('error', 'Unknown error')}")
            results.append({
                "nom": decl["nom"],
                "fraud_probability": 0.0,
                "decision": "ERROR",
                "success": False
            })
    
    # Résumé final
    print(f"\n{'='*80}")
    print("📊 RÉSUMÉ DES TESTS")
    print(f"{'='*80}\n")
    
    successful_tests = [r for r in results if r["success"]]
    
    if successful_tests:
        fraud_probs = [r["fraud_probability"] for r in successful_tests]
        min_prob = min(fraud_probs)
        max_prob = max(fraud_probs)
        avg_prob = sum(fraud_probs) / len(fraud_probs)
        
        conformes = [r for r in successful_tests if r["decision"] == "conforme"]
        fraudes = [r for r in successful_tests if r["decision"] == "fraude"]
        
        print(f"✅ Tests réussis: {len(successful_tests)}/{len(results)}")
        print(f"\n📈 Statistiques des probabilités:")
        print(f"   - Minimum: {min_prob:.4f} ({min_prob*100:.2f}%)")
        print(f"   - Maximum: {max_prob:.4f} ({max_prob*100:.2f}%)")
        print(f"   - Moyenne: {avg_prob:.4f} ({avg_prob*100:.2f}%)")
        
        print(f"\n📋 Répartition des décisions:")
        print(f"   - Conformes: {len(conformes)} ({len(conformes)/len(successful_tests)*100:.1f}%)")
        print(f"   - Frauduleuses: {len(fraudes)} ({len(fraudes)/len(successful_tests)*100:.1f}%)")
        
        print(f"\n📝 Détails par déclaration:")
        for r in successful_tests:
            status = "✅" if r["decision"] != "fraude" or r["fraud_probability"] < 0.99 else "⚠️"
            print(f"   {status} {r['nom']}: {r['fraud_probability']*100:.2f}% → {r['decision']}")
        
        # Vérifier si toutes sont frauduleuses
        if len(fraudes) == len(successful_tests):
            print(f"\n⚠️  PROBLÈME: Toutes les déclarations sont détectées comme frauduleuses!")
            print(f"   Cela suggère que les corrections n'ont pas complètement résolu le problème.")
        elif len(conformes) == len(successful_tests):
            print(f"\n⚠️  ATTENTION: Toutes les déclarations sont détectées comme conformes!")
            print(f"   Cela suggère que le modèle pourrait être trop conservateur.")
        else:
            print(f"\n✅ SUCCÈS: Mix de déclarations conformes et frauduleuses détectées!")
            print(f"   Le système fonctionne correctement avec les corrections appliquées.")
        
        # Vérifier si les probabilités sont réalistes
        if max_prob > 0.99:
            print(f"\n⚠️  ATTENTION: Certaines probabilités sont très élevées (>99%)")
            print(f"   Cela pourrait indiquer un problème avec certaines features.")
        
        if avg_prob > 0.8:
            print(f"\n⚠️  ATTENTION: La probabilité moyenne est très élevée ({avg_prob*100:.1f}%)")
            print(f"   Le modèle pourrait être trop sensible aux fraudes.")
    else:
        print(f"❌ Aucun test n'a réussi!")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()

