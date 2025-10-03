#!/usr/bin/env python3
"""
Test complet de communication Frontend-Backend pour InspectIA
Système ML-RL hybride version 2.0.0
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_RESULTS = []

def log_test(test_name, success, details=""):
    """Logger les résultats de test"""
    status = "✅ PASSÉ" if success else "❌ ÉCHOUÉ"
    timestamp = datetime.now().strftime("%H:%M:%S")
    result = f"[{timestamp}] {status} {test_name}"
    if details:
        result += f" - {details}"
    print(result)
    TEST_RESULTS.append({
        "test": test_name,
        "success": success,
        "details": details,
        "timestamp": timestamp
    })

def test_backend_health():
    """Test 1: Santé du backend"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "healthy":
                log_test("Santé Backend", True, f"Version {data.get('version')}")
                return True
        log_test("Santé Backend", False, f"Code: {response.status_code}")
        return False
    except Exception as e:
        log_test("Santé Backend", False, str(e))
        return False

def test_chapters_endpoint():
    """Test 2: Endpoint des chapitres"""
    try:
        response = requests.get(f"{BASE_URL}/chapters", timeout=10)
        if response.status_code == 200:
            data = response.json()
            chapters = data.get("chapters", [])
            if len(chapters) >= 3:
                chapter_names = [c["name"] for c in chapters]
                log_test("Chapitres Disponibles", True, f"{len(chapters)} chapitres: {', '.join(chapter_names[:3])}")
                return True
        log_test("Chapitres Disponibles", False, f"Code: {response.status_code}")
        return False
    except Exception as e:
        log_test("Chapitres Disponibles", False, str(e))
        return False

def test_chapter_config():
    """Test 3: Configuration d'un chapitre"""
    try:
        response = requests.get(f"{BASE_URL}/predict/chap30/config", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("chapter") == "chap30":
                log_test("Configuration Chapitre", True, "Chap30 configuré")
                return True
        log_test("Configuration Chapitre", False, f"Code: {response.status_code}")
        return False
    except Exception as e:
        log_test("Configuration Chapitre", False, str(e))
        return False

def test_file_upload_prediction():
    """Test 4: Upload et prédiction de fichier"""
    try:
        # Créer un fichier CSV de test
        test_csv = "DECLARATION_ID,VALEUR_CAF,POIDS_NET_KG,NOMBRE_COLIS,QUANTITE_COMPLEMENT\nTEST_001,1000.0,10.5,1,0"
        
        files = {'file': ('test_declaration.csv', test_csv, 'text/csv')}
        data = {
            'declaration_id': 'TEST_001',
            'inspector_id': 'INSP_001'
        }
        
        response = requests.post(
            f"{BASE_URL}/predict/chap30", 
            files=files, 
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if "prediction" in result:
                pred = result["prediction"]
                log_test("Upload et Prédiction", True, f"Fraude: {pred.get('predicted_fraud', 'N/A')}")
                return True
        log_test("Upload et Prédiction", False, f"Code: {response.status_code}")
        return False
    except Exception as e:
        log_test("Upload et Prédiction", False, str(e))
        return False

def test_rl_system():
    """Test 5: Système RL"""
    try:
        response = requests.get(f"{BASE_URL}/predict/chap30/rl/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "performance_summary" in data:
                stats = data["performance_summary"]
                total_decisions = stats.get("general_statistics", {}).get("total_decisions", 0)
                log_test("Système RL", True, f"{total_decisions} décisions enregistrées")
                return True
        log_test("Système RL", False, f"Code: {response.status_code}")
        return False
    except Exception as e:
        log_test("Système RL", False, str(e))
        return False

def test_postgresql_connection():
    """Test 6: Connexion PostgreSQL"""
    try:
        response = requests.get(f"{BASE_URL}/api/v2/health/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("database") == "postgresql" and data.get("connection") == "active":
                log_test("Base de Données PostgreSQL", True, "Connexion active")
                return True
        log_test("Base de Données PostgreSQL", False, f"Code: {response.status_code}")
        return False
    except Exception as e:
        log_test("Base de Données PostgreSQL", False, str(e))
        return False

def test_ml_dashboard():
    """Test 7: Dashboard ML"""
    try:
        response = requests.get(f"{BASE_URL}/ml/ml-dashboard", timeout=15)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "performance" in data["data"]:
                performance = data["data"]["performance"]
                chapters = list(performance.keys())
                log_test("Dashboard ML", True, f"Données pour {len(chapters)} chapitres")
                return True
        log_test("Dashboard ML", False, f"Code: {response.status_code}")
        return False
    except Exception as e:
        log_test("Dashboard ML", False, str(e))
        return False

def test_advanced_endpoints():
    """Test 8: Endpoints avancés"""
    try:
        # Test endpoint de détection de fraude
        response = requests.get(f"{BASE_URL}/predict/fraud-detection-methods", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "advanced_techniques" in data["data"]:
                techniques = data["data"]["advanced_techniques"]
                log_test("Endpoints Avancés", True, f"{len(techniques)} techniques de détection")
                return True
        log_test("Endpoints Avancés", False, f"Code: {response.status_code}")
        return False
    except Exception as e:
        log_test("Endpoints Avancés", False, str(e))
        return False

def test_system_status():
    """Test 9: Statut système"""
    try:
        response = requests.get(f"{BASE_URL}/predict/system-status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "overall_health" in data:
                health = data["overall_health"]
                chapters = len(data.get("chapters", {}))
                log_test("Statut Système", True, f"Santé: {health}, {chapters} chapitres")
                return True
        log_test("Statut Système", False, f"Code: {response.status_code}")
        return False
    except Exception as e:
        log_test("Statut Système", False, str(e))
        return False

def test_dependencies():
    """Test 10: Dépendances"""
    try:
        response = requests.get(f"{BASE_URL}/predict/dependencies", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "dependencies" in data:
                deps = data["dependencies"]
                available = sum(1 for v in deps.values() if v)
                total = len(deps)
                log_test("Dépendances", True, f"{available}/{total} disponibles")
                return True
        log_test("Dépendances", False, f"Code: {response.status_code}")
        return False
    except Exception as e:
        log_test("Dépendances", False, str(e))
        return False

def main():
    """Fonction principale de test"""
    print("🧪 TEST COMPLET DE COMMUNICATION FRONTEND-BACKEND")
    print("=" * 60)
    print(f"🌐 URL Backend: {BASE_URL}")
    print(f"⏰ Début des tests: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Liste des tests à exécuter
    tests = [
        ("Santé Backend", test_backend_health),
        ("Chapitres Disponibles", test_chapters_endpoint),
        ("Configuration Chapitre", test_chapter_config),
        ("Upload et Prédiction", test_file_upload_prediction),
        ("Système RL", test_rl_system),
        ("Base de Données PostgreSQL", test_postgresql_connection),
        ("Dashboard ML", test_ml_dashboard),
        ("Endpoints Avancés", test_advanced_endpoints),
        ("Statut Système", test_system_status),
        ("Dépendances", test_dependencies),
    ]
    
    # Exécution des tests
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            log_test(test_name, False, f"Erreur: {str(e)}")
        time.sleep(1)  # Pause entre les tests
    
    # Résumé
    print()
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Passés: {passed}")
    print(f"Échoués: {total - passed}")
    print(f"Taux de réussite: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS!")
        print("✅ La communication Frontend-Backend fonctionne parfaitement")
    elif passed >= total * 0.8:
        print("\n⚠️ La plupart des tests sont passés")
        print("✅ La communication Frontend-Backend fonctionne globalement")
    else:
        print("\n❌ Plusieurs tests ont échoué")
        print("⚠️ Vérifiez la configuration du backend")
    
    # Sauvegarde des résultats
    results_file = Path("test_communication_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": (passed/total)*100
            },
            "tests": TEST_RESULTS
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Résultats sauvegardés dans: {results_file}")

if __name__ == "__main__":
    main()