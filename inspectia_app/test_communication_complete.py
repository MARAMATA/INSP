#!/usr/bin/env python3
"""
🔧 TEST COMPLET DE COMMUNICATION BACKEND-FRONTEND
Teste tous les endpoints de l'API FastAPI avec les données réelles
"""

import requests
import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BackendFrontendCommunicationTester:
    """Testeur complet de communication backend-frontend"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": [],
            "start_time": time.time(),
            "end_time": None
        }
    
    def log_test(self, test_name: str, success: bool, details: str = "", error: str = ""):
        """Enregistre le résultat d'un test"""
        self.results["total_tests"] += 1
        if success:
            self.results["passed_tests"] += 1
            logger.info(f"✅ {test_name}: {details}")
        else:
            self.results["failed_tests"] += 1
            logger.error(f"❌ {test_name}: {error}")
        
        self.results["test_details"].append({
            "test_name": test_name,
            "success": success,
            "details": details,
            "error": error,
            "timestamp": time.time()
        })
    
    def test_backend_health(self) -> bool:
        """Test de santé du backend"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("Backend Health", True, f"Status: {data.get('status')}")
                return True
            else:
                self.log_test("Backend Health", False, "", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Backend Health", False, "", str(e))
            return False
    
    def test_root_endpoint(self) -> bool:
        """Test du endpoint racine"""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                features = data.get('features', [])
                self.log_test("Root Endpoint", True, f"Version: {data.get('version')}, Features: {len(features)}")
                return True
            else:
                self.log_test("Root Endpoint", False, "", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Root Endpoint", False, "", str(e))
            return False
    
    def test_chapters_endpoint(self) -> bool:
        """Test du endpoint des chapitres"""
        try:
            response = requests.get(f"{self.base_url}/chapters", timeout=10)
            if response.status_code == 200:
                data = response.json()
                chapters = data.get('chapters', [])
                self.log_test("Chapters Endpoint", True, f"Chapitres disponibles: {len(chapters)}")
                return True
            else:
                self.log_test("Chapters Endpoint", False, "", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Chapters Endpoint", False, "", str(e))
            return False
    
    def test_prediction_endpoints(self) -> bool:
        """Test des endpoints de prédiction"""
        chapters = ["chap30", "chap84", "chap85"]
        success_count = 0
        
        for chapter in chapters:
            try:
                # Test endpoint de prédiction avec fichier de test
                test_file_content = "DECLARATION_ID,VALEUR_CAF,POIDS_NET_KG,NOMBRE_COLIS,QUANTITE_COMPLEMENT\nTEST001,1000.0,10.5,1,0"
                
                files = {
                    'file': ('test_declaration.csv', test_file_content, 'text/csv')
                }
                
                response = requests.post(
                    f"{self.base_url}/predict/{chapter}",
                    files=files,
                    timeout=30
                )
                
                if response.status_code in [200, 201]:
                    data = response.json()
                    if "prediction" in data:
                        success_count += 1
                        pred_data = data.get('prediction', {})
                        decision = pred_data.get('predicted_fraud', 'N/A')
                        self.log_test(f"Prediction {chapter}", True, f"Prédiction réussie: fraude={decision}")
                    else:
                        self.log_test(f"Prediction {chapter}", False, "", f"Pas de prédiction dans la réponse")
                else:
                    self.log_test(f"Prediction {chapter}", False, "", f"Status code: {response.status_code}")
                    
            except Exception as e:
                self.log_test(f"Prediction {chapter}", False, "", str(e))
        
        success_rate = success_count / len(chapters)
        if success_rate >= 0.8:  # 80% de réussite minimum
            return True
        else:
            return False
    
    def test_ml_dashboard_endpoints(self) -> bool:
        """Test des endpoints du ML Dashboard"""
        try:
            # Test endpoint ML Dashboard
            response = requests.get(f"{self.base_url}/ml/ml-dashboard", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.log_test("ML Dashboard", True, f"Dashboard ML accessible")
                return True
            else:
                self.log_test("ML Dashboard", False, "", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("ML Dashboard", False, "", str(e))
            return False
    
    def test_postgresql_endpoints(self) -> bool:
        """Test des endpoints PostgreSQL"""
        try:
            # Test endpoint PostgreSQL
            response = requests.get(f"{self.base_url}/api/v2/health", timeout=10)
            if response.status_code in [200, 307]:
                data = response.json()
                self.log_test("PostgreSQL Health", True, f"Base de données accessible")
                return True
            else:
                self.log_test("PostgreSQL Health", False, "", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test("PostgreSQL Health", False, "", str(e))
            return False
    
    def test_file_upload(self) -> bool:
        """Test d'upload de fichier"""
        try:
            # Créer un fichier de test
            test_file_content = "DECLARATION_ID,VALEUR_CAF,POIDS_NET_KG,NOMBRE_COLIS,QUANTITE_COMPLEMENT\nTEST001,1000.0,10.5,1,0"
            test_file_path = Path("test_upload.csv")
            test_file_path.write_text(test_file_content)
            
            # Test upload
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test_upload.csv', f, 'text/csv')}
                response = requests.post(
                    f"{self.base_url}/predict/chap30",
                    files=files,
                    timeout=30
                )
            
            # Nettoyer le fichier de test
            test_file_path.unlink(missing_ok=True)
            
            if response.status_code in [200, 201]:
                data = response.json()
                if "prediction" in data:
                    self.log_test("File Upload", True, f"Upload réussi pour chap30")
                    return True
                else:
                    self.log_test("File Upload", False, "", f"Upload échoué: {data.get('detail', 'Erreur inconnue')}")
                    return False
            else:
                self.log_test("File Upload", False, "", f"Status code: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("File Upload", False, "", str(e))
            return False
    
    def test_rl_feedback_loop(self) -> bool:
        """Test du système RL feedback"""
        try:
            # Test endpoint RL feedback
            feedback_data = {
                "declaration_id": "TEST001",
                "decision": "conforme",
                "confidence": 0.85,
                "feedback_type": "inspector_feedback"
            }
            
            response = requests.post(
                f"{self.base_url}/predict/chap30/feedback",
                json=feedback_data,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                data = response.json()
                if data.get("status") == "saved":
                    self.log_test("RL Feedback", True, f"Feedback RL accepté")
                    return True
                else:
                    self.log_test("RL Feedback", False, "", f"Feedback refusé: {data.get('message', 'Erreur inconnue')}")
                    return False
            else:
                self.log_test("RL Feedback", False, "", f"Status code: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("RL Feedback", False, "", str(e))
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test des performances"""
        try:
            # Test de latence
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health", timeout=10)
            end_time = time.time()
            
            latency = end_time - start_time
            
            # Test de débit (multiple requêtes)
            start_time = time.time()
            for _ in range(10):
                requests.get(f"{self.base_url}/health", timeout=5)
            end_time = time.time()
            
            throughput_time = end_time - start_time
            throughput = 10 / throughput_time  # requêtes par seconde
            
            # Critères de performance
            latency_ok = latency < 2.0  # moins de 2 secondes
            throughput_ok = throughput > 5.0  # plus de 5 req/s
            
            if latency_ok and throughput_ok:
                self.log_test("Performance", True, f"Latence: {latency:.2f}s, Débit: {throughput:.1f} req/s")
                return True
            else:
                self.log_test("Performance", False, "", f"Latence: {latency:.2f}s, Débit: {throughput:.1f} req/s")
                return False
                
        except Exception as e:
            self.log_test("Performance", False, "", str(e))
            return False
    
    def test_error_handling(self) -> bool:
        """Test de gestion des erreurs"""
        try:
            error_handling_ok = True
            
            # Test endpoint inexistant
            response = requests.get(f"{self.base_url}/nonexistent", timeout=10)
            if response.status_code == 404:
                error_handling_ok = error_handling_ok and True
            else:
                error_handling_ok = False
            
            # Test données invalides
            invalid_data = {"invalid": "data"}
            response = requests.post(f"{self.base_url}/predict/chap30", 
                                   json=invalid_data, timeout=10)
            if response.status_code in [400, 422, 500]:
                error_handling_ok = error_handling_ok and True
            else:
                error_handling_ok = False
            
            if error_handling_ok:
                self.log_test("Error Handling", True, f"Gestion d'erreurs correcte")
                return True
            else:
                self.log_test("Error Handling", False, "", f"Gestion d'erreurs incorrecte")
                return False
                
        except Exception as e:
            self.log_test("Error Handling", False, "", str(e))
            return False
    
    def test_data_consistency(self) -> bool:
        """Test de cohérence des données"""
        try:
            # Test cohérence entre différents endpoints
            chapters_response = requests.get(f"{self.base_url}/chapters", timeout=10)
            health_response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if chapters_response.status_code == 200 and health_response.status_code == 200:
                chapters_data = chapters_response.json()
                health_data = health_response.json()
                
                # Vérifier la cohérence
                if (chapters_data.get("chapters") and 
                    health_data.get("status") == "healthy"):
                    self.log_test("Data Consistency", True, f"Cohérence des données vérifiée")
                    return True
                else:
                    self.log_test("Data Consistency", False, "", f"Cohérence non vérifiée")
                    return False
            else:
                self.log_test("Data Consistency", False, "", f"Endpoints non accessibles")
                return False
                
        except Exception as e:
            self.log_test("Data Consistency", False, "", str(e))
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Exécute tous les tests"""
        logger.info("🚀 DÉMARRAGE DES TESTS DE COMMUNICATION BACKEND-FRONTEND")
        logger.info("=" * 60)
        
        # Liste des tests à exécuter
        tests = [
            ("Backend Health", self.test_backend_health),
            ("Root Endpoint", self.test_root_endpoint),
            ("Chapters Endpoint", self.test_chapters_endpoint),
            ("Prediction Endpoints", self.test_prediction_endpoints),
            ("ML Dashboard", self.test_ml_dashboard_endpoints),
            ("PostgreSQL Endpoints", self.test_postgresql_endpoints),
            ("File Upload", self.test_file_upload),
            ("RL Feedback Loop", self.test_rl_feedback_loop),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Error Handling", self.test_error_handling),
            ("Data Consistency", self.test_data_consistency)
        ]
        
        # Exécution des tests
        for test_name, test_func in tests:
            try:
                test_func()
                time.sleep(1)  # Pause entre les tests
            except Exception as e:
                self.log_test(test_name, False, "", f"Erreur d'exécution: {str(e)}")
        
        self.results["end_time"] = time.time()
        return self.results
    
    def print_summary(self):
        """Affiche le résumé des tests"""
        total = self.results["total_tests"]
        passed = self.results["passed_tests"]
        failed = self.results["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        logger.info("\n📋 RÉSUMÉ DES TESTS")
        logger.info("=" * 60)
        logger.info(f"Total: {total}")
        logger.info(f"Passés: {passed}")
        logger.info(f"Échoués: {failed}")
        logger.info(f"Taux de réussite: {success_rate:.1f}%")
        
        if success_rate >= 90:
            logger.info("🎉 EXCELLENT! Système entièrement fonctionnel!")
        elif success_rate >= 80:
            logger.info("✅ BON! Système largement fonctionnel!")
        elif success_rate >= 70:
            logger.info("⚠️ MOYEN! Quelques problèmes détectés!")
        else:
            logger.info("❌ PROBLÈMES! Système nécessite des corrections!")
        
        # Détails des échecs
        if failed > 0:
            logger.info("\n❌ TESTS ÉCHOUÉS:")
            for test in self.results["test_details"]:
                if not test["success"]:
                    logger.info(f"  - {test['test_name']}: {test['error']}")

def main():
    """Fonction principale"""
    # Vérifier si le backend est démarré
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code != 200:
            logger.error("❌ Backend non accessible sur http://127.0.0.1:8000")
            logger.info("💡 Démarrez le backend avec: cd backend && python -m uvicorn api.main:app --reload")
            return 1
    except Exception as e:
        logger.error("❌ Backend non démarré ou non accessible")
        logger.info("💡 Démarrez le backend avec: cd backend && python -m uvicorn api.main:app --reload")
        return 1
    
    # Exécuter tous les tests
    tester = BackendFrontendCommunicationTester()
    results = tester.run_all_tests()
    
    # Afficher le résumé
    tester.print_summary()
    
    # Sauvegarder les résultats
    results_file = Path("test_communication_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n📄 Résultats sauvegardés dans: {results_file}")
    
    # Retourner le code de sortie approprié
    success_rate = (results["passed_tests"] / results["total_tests"] * 100) if results["total_tests"] > 0 else 0
    return 0 if success_rate >= 80 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
