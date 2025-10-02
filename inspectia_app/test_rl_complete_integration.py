#!/usr/bin/env python3
"""
Script de test complet pour vérifier l'intégration RL avec tous les changements
"""

import sys
from pathlib import Path

# Ajouter le chemin du backend au PYTHONPATH
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_complete_rl_integration():
    """Test complet de l'intégration RL avec tous les changements"""
    print("🧪 TEST COMPLET D'INTÉGRATION RL AVEC NOUVEAUX MODÈLES")
    print("=" * 70)
    
    # Test pour chaque chapitre
    chapters = ["chap30", "chap84", "chap85"]
    
    for chapter in chapters:
        print(f"\n📊 Test complet du chapitre {chapter}")
        print("-" * 50)
        
        try:
            # Importer le module RL du chapitre
            if chapter == "chap30":
                from src.chapters.chap30.rl_integration import rl_predict, rl_performance, rl_analytics
            elif chapter == "chap84":
                from src.chapters.chap84.rl_integration import rl_predict, rl_performance, rl_analytics
            elif chapter == "chap85":
                from src.chapters.chap85.rl_integration import rl_predict, rl_performance, rl_analytics
            
            # Créer un contexte de test complet avec TOUTES les nouvelles features
            test_context = create_complete_test_context(chapter)
            
            # Test 1: Prédiction RL avec toutes les nouvelles features
            print(f"🔍 Test 1: Prédiction RL avec nouvelles features...")
            result = rl_predict(test_context, level="basic")
            
            print(f"✅ Prédiction réussie:")
            print(f"   - Action: {result.get('action', 'N/A')}")
            print(f"   - Probabilité: {result.get('fraud_probability', 0):.3f}")
            print(f"   - Décision: {result.get('predicted_fraud', False)}")
            print(f"   - Confiance: {result.get('confidence_score', 0):.3f}")
            print(f"   - ML utilisé: {result.get('ml_integration_used', False)}")
            print(f"   - Seuil optimal utilisé: {result.get('optimal_threshold_used', 0):.3f}")
            
            # Vérifier les performances du modèle
            model_perf = result.get('model_performance', {})
            if model_perf:
                print(f"   - AUC du modèle: {model_perf.get('auc', 0):.4f}")
                print(f"   - F1 du modèle: {model_perf.get('f1', 0):.4f}")
                print(f"   - Precision: {model_perf.get('precision', 0):.4f}")
                print(f"   - Recall: {model_perf.get('recall', 0):.4f}")
            
            # Test 2: Prédiction avec probabilité ML fournie
            print(f"\n🔍 Test 2: Prédiction avec probabilité ML fournie...")
            result_ml = rl_predict(test_context, level="advanced", ml_probability=0.15)
            
            print(f"✅ Prédiction ML réussie:")
            print(f"   - Probabilité ML: {result_ml.get('ml_probability', 0):.3f}")
            print(f"   - Probabilité finale: {result_ml.get('fraud_probability', 0):.3f}")
            print(f"   - Décision: {result_ml.get('predicted_fraud', False)}")
            print(f"   - ML utilisé: {result_ml.get('ml_integration_used', False)}")
            
            # Test 3: Prédiction expert
            print(f"\n🔍 Test 3: Prédiction niveau expert...")
            result_expert = rl_predict(test_context, level="expert")
            
            print(f"✅ Prédiction expert réussie:")
            print(f"   - Probabilité: {result_expert.get('fraud_probability', 0):.3f}")
            print(f"   - Décision: {result_expert.get('predicted_fraud', False)}")
            print(f"   - Confiance: {result_expert.get('confidence_score', 0):.3f}")
            
            # Test 4: Performances RL
            print(f"\n📊 Test 4: Performances RL...")
            performance = rl_performance(level="basic")
            
            print(f"✅ Performances récupérées:")
            print(f"   - Prédictions: {performance.get('total_predictions', 0)}")
            print(f"   - Feedbacks: {performance.get('total_feedbacks', 0)}")
            print(f"   - Précision: {performance.get('precision', 0):.3f}")
            print(f"   - Rappel: {performance.get('recall', 0):.3f}")
            
            # Test 5: Analytics RL
            print(f"\n📈 Test 5: Analytics RL...")
            analytics = rl_analytics(level="basic")
            
            print(f"✅ Analytics récupérées:")
            print(f"   - Chapitre: {analytics.get('chapter', 'N/A')}")
            print(f"   - Taille cache inspecteurs: {analytics.get('inspector_cache_size', 0)}")
            print(f"   - Métriques session: {analytics.get('session_metrics', {})}")
            
            print(f"\n✅ Tous les tests réussis pour {chapter}")
            
        except Exception as e:
            print(f"❌ Erreur pour {chapter}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLET D'INTÉGRATION RL TERMINÉ")

def create_complete_test_context(chapter):
    """Crée un contexte de test complet avec toutes les nouvelles features"""
    
    # Contexte de base commun
    context = {
        "DECLARATION_ID": f"TEST_{chapter}_001",
        "VALEUR_CAF": 50000000,
        "VALEUR_DOUANE": 48000000,
        "MONTANT_LIQUIDATION": 2000000,
        "POIDS_NET": 1000,
        "VALEUR_UNITAIRE_KG": 50000,
        "TAUX_DROITS_PERCENT": 4.0,
        "RATIO_DOUANE_CAF": 0.96,
        "NUMERO_ARTICLE": 1,
        "PRECISION_UEMOA": 0,
        "CODE_PRODUIT_STR": "300490" if chapter == "chap30" else "847130" if chapter == "chap84" else "851712",
        "PAYS_ORIGINE_STR": "CN",
        "PAYS_PROVENANCE_STR": "CN",
        "BUREAU": "19C",
        "REGIME_FISCAL": 0,
        "NUMERO_DPI": "",
        
        # Nouvelles features de détection de fraude avancée
        "BIENAYME_CHEBYCHEV_SCORE": 5.2,
        "TEI_CALCULE": 4.0,
        "MIRROR_TEI_SCORE": 800.5,
        "MIRROR_TEI_DEVIATION": 0.15,
        "SPECTRAL_CLUSTER_SCORE": 0.3,
        "HIERARCHICAL_CLUSTER_SCORE": 0.4,
        "ADMIN_VALUES_SCORE": 75.2,
        "ADMIN_VALUES_DEVIATION": 0.1,
        "COMPOSITE_FRAUD_SCORE": 0.12,
        "RATIO_POIDS_VALEUR": 0.02,
        
        # Features communes
        "BUSINESS_RISK_PAYS_HIGH": 0.8,
        "BUSINESS_ORIGINE_DIFF_PROVENANCE": 0.0,
        "BUSINESS_REGIME_PREFERENTIEL": 0.0,
        "BUSINESS_REGIME_NORMAL": 1.0,
        "BUSINESS_VALEUR_ELEVEE": 0.5,
        "BUSINESS_VALEUR_EXCEPTIONNELLE": 0.0,
        "BUSINESS_RATIO_SUSPECT": 0.1,
        "BUSINESS_TAUX_DROITS_FAIBLE": 0.0,
        "BUSINESS_TAUX_DROITS_ELEVE": 0.0,
        "BUSINESS_VALEUR_UNITAIRE_FAIBLE": 0.0,
        "BUSINESS_VALEUR_UNITAIRE_ELEVEE": 0.5,
        "BUSINESS_ANNEE_RECENTE": 1.0,
        "BUSINESS_ARTICLES_MULTIPLES": 0.0,
        "BUSINESS_AVEC_DPI": 0.0,
        "BUSINESS_IS_PRECISION_UEMOA": 0.0,
        "BUSINESS_DROITS_ELEVES": 0.0,
        "BUSINESS_RATIO_LIQUIDATION_CAF": 0.04,
        "BUSINESS_RATIO_DOUANE_CAF": 0.96,
    }
    
    # Features spécifiques par chapitre
    if chapter == "chap30":
        context.update({
            "BUSINESS_GLISSEMENT_PHARMA": 0.3,
            "BUSINESS_GLISSEMENT_PAYS_PHARMA": 0.2,
            "BUSINESS_GLISSEMENT_RATIO_SUSPECT": 0.1,
            "BUSINESS_POIDS_FAIBLE": 0.0,
            "BUSINESS_POIDS_EXCEPTIONNEL": 0.0,
            "BUSINESS_BUREAU_RISQUE": 0.8,
            "BUSINESS_IS_MEDICAMENT": 0.5,
            "BUSINESS_POIDS_ELEVE": 0.0,
            "BUSINESS_IS_ANTIPALUDEEN": 0.2,
            "BUSINESS_GLISSEMENT_COSMETIQUE": 0.1,
            "BUSINESS_GLISSEMENT_PAYS_COSMETIQUES": 0.1
        })
    elif chapter == "chap84":
        context.update({
            "BUSINESS_GLISSEMENT_MACHINE": 0.4,
            "BUSINESS_GLISSEMENT_PAYS_MACHINES": 0.3,
            "BUSINESS_GLISSEMENT_RATIO_SUSPECT": 0.2,
            "BUSINESS_POIDS_ELEVE": 0.0,
            "BUSINESS_IS_MACHINE": 1.0,
            "BUSINESS_IS_ELECTRONIQUE": 0.0
        })
    elif chapter == "chap85":
        context.update({
            "BUSINESS_GLISSEMENT_ELECTRONIQUE": 0.3,
            "BUSINESS_GLISSEMENT_PAYS_ELECTRONIQUES": 0.2,
            "BUSINESS_GLISSEMENT_RATIO_SUSPECT": 0.1,
            "BUSINESS_POIDS_FAIBLE": 0.0,
            "BUSINESS_IS_ELECTRONIQUE": 1.0,
            "BUSINESS_IS_TELEPHONE": 0.0,
            "BUSINESS_BUREAU_RISQUE": 0.8
        })
    
    return context

def test_threshold_consistency():
    """Test la cohérence des seuils optimaux"""
    print(f"\n🎯 TEST DE COHÉRENCE DES SEUILS OPTIMAUX")
    print("-" * 50)
    
    import json
    from pathlib import Path
    
    chapters = ['chap30', 'chap84', 'chap85']
    expected_thresholds = {
        'chap30': 0.230,
        'chap84': 0.220,
        'chap85': 0.220
    }
    
    for chapter in chapters:
        thresholds_file = Path(f'backend/results/{chapter}/optimal_thresholds.json')
        if thresholds_file.exists():
            with open(thresholds_file, 'r') as f:
                thresholds = json.load(f)
            
            optimal_threshold = thresholds.get('optimal_threshold', 0)
            expected = expected_thresholds[chapter]
            
            if abs(optimal_threshold - expected) < 0.001:
                print(f"✅ {chapter}: Seuil optimal {optimal_threshold:.3f} cohérent")
            else:
                print(f"⚠️  {chapter}: Seuil optimal {optimal_threshold:.3f} vs attendu {expected:.3f}")
        else:
            print(f"❌ {chapter}: Fichier de seuils non trouvé")

if __name__ == "__main__":
    test_complete_rl_integration()
    test_threshold_consistency()
