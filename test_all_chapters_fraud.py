#!/usr/bin/env python3
"""
Test de détection de fraude pour tous les chapitres (30, 84, 85)
Vérifier que les déclarations normales ne sont pas toutes détectées comme fraudes
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

logging.basicConfig(level=logging.WARNING)  # Réduire les logs
logger = logging.getLogger(__name__)

def test_declaration(valeur_caf, poids_net, code_produit, pays_origine, pays_provenance, bureau, regime_fiscal, numero_dpi, chapitre):
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

def test_chapter(chapter_name, test_declarations):
    """Tester un chapitre avec plusieurs déclarations"""
    
    print(f"\n{'='*80}")
    print(f"🧪 TEST CHAPITRE {chapter_name.upper()}")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, decl in enumerate(test_declarations, 1):
        print(f"📦 Test {i}/{len(test_declarations)}: {decl['nom']}")
        
        result = test_declaration(
            valeur_caf=decl["valeur_caf"],
            poids_net=decl["poids_net"],
            code_produit=decl["code_produit"],
            pays_origine=decl["pays_origine"],
            pays_provenance=decl["pays_provenance"],
            bureau=decl["bureau"],
            regime_fiscal=decl["regime_fiscal"],
            numero_dpi=decl["numero_dpi"],
            chapitre=chapter_name
        )
        
        if result["success"]:
            fraud_prob = result["fraud_probability"]
            decision = result["decision"]
            
            status = "✅" if decision == "conforme" else "⚠️"
            print(f"   {status} Probabilité: {fraud_prob:.4f} ({fraud_prob*100:.2f}%) → {decision}")
            
            results.append({
                "nom": decl["nom"],
                "fraud_probability": fraud_prob,
                "decision": decision,
                "success": True
            })
        else:
            print(f"   ❌ Erreur: {result.get('error', 'Unknown error')}")
            results.append({
                "nom": decl["nom"],
                "fraud_probability": 0.0,
                "decision": "ERROR",
                "success": False
            })
    
    # Résumé pour ce chapitre
    successful_tests = [r for r in results if r["success"]]
    
    if successful_tests:
        fraud_probs = [r["fraud_probability"] for r in successful_tests]
        min_prob = min(fraud_probs)
        max_prob = max(fraud_probs)
        avg_prob = sum(fraud_probs) / len(fraud_probs)
        
        conformes = [r for r in successful_tests if r["decision"] == "conforme"]
        fraudes = [r for r in successful_tests if r["decision"] == "fraude"]
        
        print(f"\n📊 Résumé {chapter_name}:")
        print(f"   ✅ Tests réussis: {len(successful_tests)}/{len(results)}")
        print(f"   📈 Probabilités: Min={min_prob:.4f} ({min_prob*100:.2f}%), Max={max_prob:.4f} ({max_prob*100:.2f}%), Moyenne={avg_prob:.4f} ({avg_prob*100:.2f}%)")
        print(f"   📋 Décisions: Conformes={len(conformes)} ({len(conformes)/len(successful_tests)*100:.1f}%), Frauduleuses={len(fraudes)} ({len(fraudes)/len(successful_tests)*100:.1f}%)")
        
        if len(fraudes) == len(successful_tests):
            print(f"   ⚠️  PROBLÈME: Toutes les déclarations sont détectées comme frauduleuses!")
        elif len(conformes) == len(successful_tests):
            print(f"   ⚠️  ATTENTION: Toutes les déclarations sont détectées comme conformes!")
        else:
            print(f"   ✅ SUCCÈS: Mix de déclarations conformes et frauduleuses!")
        
        return {
            "chapter": chapter_name,
            "total": len(successful_tests),
            "conformes": len(conformes),
            "fraudes": len(fraudes),
            "min_prob": min_prob,
            "max_prob": max_prob,
            "avg_prob": avg_prob
        }
    else:
        print(f"   ❌ Aucun test n'a réussi pour {chapter_name}!")
        return None

def main():
    """Test principal pour tous les chapitres"""
    
    print("=" * 80)
    print("🧪 TEST DE DÉTECTION DE FRAUDE - TOUS LES CHAPITRES")
    print("=" * 80)
    print("\n📋 Objectif: Vérifier que les déclarations normales ne sont pas toutes détectées comme fraudes")
    print("   pour les chapitres 30, 84 et 85.\n")
    
    # Déclarations de test pour CHAPITRE 30
    test_chap30 = [
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
        {
            "nom": "Déclaration normale (valeur médiane)",
            "valeur_caf": 22781555,  # Médiane chap30
            "poids_net": 1000,
            "code_produit": "300490",
            "pays_origine": "SN",
            "pays_provenance": "SN",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        {
            "nom": "Déclaration normale (valeur moyenne)",
            "valeur_caf": 15000000,
            "poids_net": 800,
            "code_produit": "300490",
            "pays_origine": "DE",
            "pays_provenance": "DE",
            "bureau": "10S",
            "regime_fiscal": 2,
            "numero_dpi": "SANS_DPI"
        },
        {
            "nom": "Déclaration suspecte (très haute valeur)",
            "valeur_caf": 150000000,  # > Q75
            "poids_net": 5000,
            "code_produit": "300490",
            "pays_origine": "CN",
            "pays_provenance": "CN",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        {
            "nom": "Déclaration suspecte (glissement tarifaire)",
            "valeur_caf": 20000000,
            "poids_net": 800,
            "code_produit": "330410",  # Code cosmétique
            "pays_origine": "FR",
            "pays_provenance": "FR",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
    ]
    
    # Déclarations de test pour CHAPITRE 84
    test_chap84 = [
        {
            "nom": "Déclaration normale (petite valeur)",
            "valeur_caf": 1000000,
            "poids_net": 200,
            "code_produit": "847130",
            "pays_origine": "CN",
            "pays_provenance": "CN",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        {
            "nom": "Déclaration normale (valeur médiane)",
            "valeur_caf": 2323379,  # Médiane chap84
            "poids_net": 500,
            "code_produit": "847130",
            "pays_origine": "CN",
            "pays_provenance": "CN",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        {
            "nom": "Déclaration normale (valeur moyenne)",
            "valeur_caf": 5000000,
            "poids_net": 1000,
            "code_produit": "847130",
            "pays_origine": "DE",
            "pays_provenance": "DE",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        {
            "nom": "Déclaration suspecte (très haute valeur)",
            "valeur_caf": 15000000,  # > Q75
            "poids_net": 3000,
            "code_produit": "847130",
            "pays_origine": "CN",
            "pays_provenance": "CN",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        {
            "nom": "Déclaration suspecte (pays à risque)",
            "valeur_caf": 3000000,
            "poids_net": 600,
            "code_produit": "847130",
            "pays_origine": "PK",  # Pays à risque
            "pays_provenance": "PK",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
    ]
    
    # Déclarations de test pour CHAPITRE 85
    test_chap85 = [
        {
            "nom": "Déclaration normale (petite valeur)",
            "valeur_caf": 500000,
            "poids_net": 50,
            "code_produit": "851712",
            "pays_origine": "CN",
            "pays_provenance": "CN",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        {
            "nom": "Déclaration normale (valeur médiane)",
            "valeur_caf": 1124669,  # Médiane chap85
            "poids_net": 100,
            "code_produit": "851712",
            "pays_origine": "CN",
            "pays_provenance": "CN",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        {
            "nom": "Déclaration normale (valeur moyenne)",
            "valeur_caf": 3000000,
            "poids_net": 200,
            "code_produit": "851712",
            "pays_origine": "KR",
            "pays_provenance": "KR",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        {
            "nom": "Déclaration suspecte (très haute valeur)",
            "valeur_caf": 10000000,  # > Q75
            "poids_net": 500,
            "code_produit": "851712",
            "pays_origine": "CN",
            "pays_provenance": "CN",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
        {
            "nom": "Déclaration suspecte (téléphone haut de gamme suspect)",
            "valeur_caf": 5000000,
            "poids_net": 10,  # Très léger pour la valeur (suspect)
            "code_produit": "851712",  # Téléphone
            "pays_origine": "CN",
            "pays_provenance": "CN",
            "bureau": "10S",
            "regime_fiscal": 0,
            "numero_dpi": "SANS_DPI"
        },
    ]
    
    # Tests pour chaque chapitre
    summary = []
    
    summary.append(test_chapter("chap30", test_chap30))
    summary.append(test_chapter("chap84", test_chap84))
    summary.append(test_chapter("chap85", test_chap85))
    
    # Résumé global
    print(f"\n{'='*80}")
    print("📊 RÉSUMÉ GLOBAL - TOUS LES CHAPITRES")
    print(f"{'='*80}\n")
    
    for s in summary:
        if s:
            print(f"📦 {s['chapter'].upper()}:")
            print(f"   Total: {s['total']} tests")
            print(f"   Conformes: {s['conformes']} ({s['conformes']/s['total']*100:.1f}%)")
            print(f"   Frauduleuses: {s['fraudes']} ({s['fraudes']/s['total']*100:.1f}%)")
            print(f"   Probabilité moyenne: {s['avg_prob']:.4f} ({s['avg_prob']*100:.2f}%)")
            print(f"   Plage: [{s['min_prob']:.4f}, {s['max_prob']:.4f}]")
            print()
    
    # Conclusion globale
    all_conformes = sum(s['conformes'] for s in summary if s)
    all_total = sum(s['total'] for s in summary if s)
    all_fraudes = sum(s['fraudes'] for s in summary if s)
    
    print(f"🎯 RÉSULTAT GLOBAL:")
    print(f"   Total déclarations testées: {all_total}")
    print(f"   Conformes: {all_conformes} ({all_conformes/all_total*100:.1f}%)")
    print(f"   Frauduleuses: {all_fraudes} ({all_fraudes/all_total*100:.1f}%)")
    
    if all_fraudes == all_total:
        print(f"\n❌ PROBLÈME CRITIQUE: TOUTES les déclarations sont détectées comme frauduleuses!")
        print(f"   Les corrections n'ont pas résolu le problème pour tous les chapitres.")
    elif all_conformes == all_total:
        print(f"\n⚠️  ATTENTION: TOUTES les déclarations sont détectées comme conformes!")
        print(f"   Le modèle pourrait être trop conservateur.")
    else:
        print(f"\n✅ SUCCÈS: Mix de déclarations conformes et frauduleuses détectées!")
        print(f"   Le système fonctionne correctement avec les corrections appliquées.")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()

