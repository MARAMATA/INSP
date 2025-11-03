#!/usr/bin/env python3
"""
Script de test pour vérifier que les prédictions ML fonctionnent correctement
"""
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_prediction():
    """Test une prédiction pour chaque chapitre"""
    
    print("=" * 80)
    print("🧪 TEST DES PRÉDICTIONS ML")
    print("=" * 80)
    print()
    
    # Test data pour chaque chapitre
    test_cases = {
        "chap30": {
            "DECLARATION_ID": f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "ANNEE": "2025",
            "BUREAU": "10S",
            "NUMERO": "12345",
            "VALEUR_CAF": 50000.0,
            "POIDS_NET_KG": 100.0,
            "CODE_SH_COMPLET": "3004900000",
            "PAYS_ORIGINE_STR": "CN",
            "PAYS_PROVENANCE_STR": "CN",
            "TAUX_DROITS_PERCENT": 0.0,
            "QUANTITE_COMPLEMENT": 100.0,
            "NOMBRE_COLIS": 10
        },
        "chap84": {
            "DECLARATION_ID": f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}_84",
            "ANNEE": "2025",
            "BUREAU": "20D",
            "NUMERO": "67890",
            "VALEUR_CAF": 100000.0,
            "POIDS_NET_KG": 200.0,
            "CODE_SH_COMPLET": "8471300000",
            "PAYS_ORIGINE_STR": "CN",
            "PAYS_PROVENANCE_STR": "CN",
            "TAUX_DROITS_PERCENT": 5.0,
            "QUANTITE_COMPLEMENT": 1.0,
            "NOMBRE_COLIS": 5
        },
        "chap85": {
            "DECLARATION_ID": f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}_85",
            "ANNEE": "2025",
            "BUREAU": "19C",
            "NUMERO": "11111",
            "VALEUR_CAF": 80000.0,
            "POIDS_NET_KG": 150.0,
            "CODE_SH_COMPLET": "8517120000",
            "PAYS_ORIGINE_STR": "CN",
            "PAYS_PROVENANCE_STR": "CN",
            "TAUX_DROITS_PERCENT": 5.0,
            "QUANTITE_COMPLEMENT": 10.0,
            "NOMBRE_COLIS": 8
        }
    }
    
    results = {}
    
    for chapter, test_data in test_cases.items():
        print(f"\n{'='*80}")
        print(f"📊 TEST CHAPITRE: {chapter}")
        print(f"{'='*80}")
        
        try:
            # Appeler l'endpoint de prédiction (POST /predict/{chapter}/declarations)
            url = f"{BASE_URL}/predict/{chapter}/declarations"
            print(f"🔗 URL: {url}")
            print(f"📤 Données: {json.dumps(test_data, indent=2)}")
            print()
            
            response = requests.post(
                url,
                json=[test_data],
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            print(f"📥 Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                if "predictions" in result and len(result["predictions"]) > 0:
                    pred = result["predictions"][0]["prediction"]
                    
                    fraud_prob = pred.get("fraud_probability", 0.0)
                    decision = pred.get("decision_source", "unknown")
                    predicted_fraud = pred.get("predicted_fraud", False)
                    
                    print(f"✅ PRÉDICTION RÉUSSIE:")
                    print(f"   - Probabilité de fraude: {fraud_prob:.4f}")
                    print(f"   - Prédiction: {'FRAUDE' if predicted_fraud else 'CONFORME'}")
                    print(f"   - Source: {decision}")
                    
                    results[chapter] = {
                        "success": True,
                        "fraud_probability": fraud_prob,
                        "predicted_fraud": predicted_fraud,
                        "decision_source": decision
                    }
                    
                    # Vérifier si la probabilité est valide
                    if fraud_prob == 0.0:
                        print(f"   ⚠️  ATTENTION: Probabilité à 0.0 - problème possible!")
                    elif fraud_prob > 0.5:
                        print(f"   🚨 FRAUDE DÉTECTÉE: Probabilité élevée ({fraud_prob:.2%})")
                    else:
                        print(f"   ✅ CONFORME: Probabilité faible ({fraud_prob:.2%})")
                        
                else:
                    print(f"❌ ERREUR: Pas de prédiction dans la réponse")
                    results[chapter] = {"success": False, "error": "Pas de prédiction"}
                    
            else:
                print(f"❌ ERREUR HTTP {response.status_code}")
                print(f"   Réponse: {response.text[:500]}")
                results[chapter] = {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}")
            results[chapter] = {"success": False, "error": str(e)}
        
        time.sleep(1)  # Pause entre les tests
    
    # Résumé
    print(f"\n{'='*80}")
    print("📊 RÉSUMÉ DES TESTS")
    print(f"{'='*80}")
    
    for chapter, result in results.items():
        if result.get("success"):
            prob = result.get("fraud_probability", 0.0)
            fraud = result.get("predicted_fraud", False)
            print(f"{chapter:10} | Prob: {prob:6.4f} | {'FRAUDE' if fraud else 'CONFORME':8} | ✅ OK")
        else:
            error = result.get("error", "Unknown")
            print(f"{chapter:10} | {'ERROR':20} | ❌ {error}")
    
    print()
    
    # Vérifier les prédictions en base de données
    print(f"{'='*80}")
    print("🔍 VÉRIFICATION BASE DE DONNÉES")
    print(f"{'='*80}")
    
    try:
        # Vérifier les dernières prédictions
        db_url = f"{BASE_URL}/api/v2/postgresql/declarations"
        db_response = requests.get(db_url, params={"limit": 10}, timeout=10)
        
        if db_response.status_code == 200:
            db_data = db_response.json()
            if db_data.get("success") and "declarations" in db_data:
                print(f"✅ {len(db_data['declarations'])} dernières déclarations:")
                for decl in db_data["declarations"][:5]:
                    prob = decl.get("fraud_probability", 0.0)
                    decision = decl.get("decision", "N/A")
                    print(f"   - {decl.get('declaration_id', 'N/A'):20} | Prob: {prob:6.4f} | {decision:10}")
        else:
            print(f"⚠️  Impossible de vérifier la base de données (HTTP {db_response.status_code})")
            
    except Exception as e:
        print(f"⚠️  Erreur vérification DB: {e}")
    
    print()
    return results

if __name__ == "__main__":
    try:
        # Vérifier que le backend est accessible
        health_url = f"{BASE_URL}/api/v2/health"
        print("🔍 Vérification du backend...")
        health_response = requests.get(health_url, timeout=5)
        
        if health_response.status_code == 200:
            print("✅ Backend accessible\n")
            test_prediction()
        else:
            print(f"❌ Backend non accessible (HTTP {health_response.status_code})")
            print("   Assurez-vous que le backend est démarré sur http://localhost:8000")
            
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter au backend")
        print("   Assurez-vous que le backend est démarré sur http://localhost:8000")
    except Exception as e:
        print(f"❌ Erreur: {e}")

