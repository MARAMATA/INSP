#!/usr/bin/env python3
"""
🧪 TESTS COMPLETS DE COMMUNICATION FRONTEND-BACKEND
Teste l'intégration complète entre le frontend Flutter et le backend Python
"""

import asyncio
import json
import time
import requests
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import os

class FrontendBackendCommunicationTester:
    """Testeur complet de communication frontend-backend"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.frontend_path = Path("inspectia_app_frontend")
        self.backend_path = Path("backend")
        self.results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "test_details": []
        }
        
    def log_test(self, test_name: str, success: bool, details: str = "", error: str = ""):
        """Enregistre le résultat d'un test"""
        self.results["total_tests"] += 1
        if success:
            self.results["passed_tests"] += 1
            print(f"✅ {test_name}")
        else:
            self.results["failed_tests"] += 1
            print(f"❌ {test_name}: {error}")
            
        self.results["test_details"].append({
            "test_name": test_name,
            "success": success,
            "details": details,
            "error": error,
            "timestamp": time.time()
        })
    
    def test_backend_health(self) -> bool:
        """Test 1: Santé du backend"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("data", {}).get("status") == "healthy":
                    self.log_test("Backend Health Check", True, f"Version: {data.get('data', {}).get('version')}")
                    return True
            self.log_test("Backend Health Check", False, "", f"Status: {response.status_code}")
            return False
        except Exception as e:
            self.log_test("Backend Health Check", False, "", str(e))
            return False
    
    def test_backend_endpoints(self) -> bool:
        """Test 2: Tous les endpoints backend"""
        endpoints_to_test = [
            ("/api/v1/chapters", "GET"),
            ("/api/v1/system/status", "GET"),
            ("/api/v1/config/chap30", "GET"),
            ("/api/v1/model/chap30/info", "GET"),
            ("/api/v1/features/chap30", "GET"),
            ("/api/v1/status/chap30", "GET"),
            ("/api/v1/ml/dashboard", "GET"),
            ("/api/v1/rl/performance/chap30/basic", "GET"),
            ("/api/v1/rl/analytics/chap30/basic", "GET"),
            ("/api/v1/retraining/chap30/status", "GET"),
            ("/api/v1/inspectors/chap30/profiles", "GET"),
            ("/api/v1/chef/dashboard", "GET"),
            ("/api/v1/declarations/", "GET"),
            ("/api/v1/postgresql/health", "GET"),
            ("/api/v1/postgresql/stats", "GET"),
        ]
        
        success_count = 0
        for endpoint, method in endpoints_to_test:
            try:
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                elif method == "POST":
                    response = requests.post(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code in [200, 201]:
                    success_count += 1
                else:
                    self.log_test(f"Endpoint {endpoint}", False, "", f"Status: {response.status_code}")
            except Exception as e:
                self.log_test(f"Endpoint {endpoint}", False, "", str(e))
        
        success_rate = success_count / len(endpoints_to_test)
        if success_rate >= 0.8:  # 80% de réussite minimum
            self.log_test("Backend Endpoints", True, f"{success_count}/{len(endpoints_to_test)} endpoints OK")
            return True
        else:
            self.log_test("Backend Endpoints", False, "", f"Only {success_count}/{len(endpoints_to_test)} endpoints OK")
            return False
    
    def test_prediction_endpoints(self) -> bool:
        """Test 3: Endpoints de prédiction"""
        test_data = {
            "declaration_id": "TEST_COMM_001",
            "valeur_caf": 15000.0,
            "poids_net_kg": 750.0,
            "nombre_colis": 15,
            "code_sh_complet": "30049000",
            "pays_origine": "FR",
            "bureau_douane": "01A"
        }
        
        chapters = ["chap30", "chap84", "chap85"]
        success_count = 0
        
        for chapter in chapters:
            try:
                # Test prédiction avec données
                response = requests.post(
                    f"{self.base_url}/api/v1/predict/{chapter}",
                    json=test_data,
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") and "prediction" in data:
                        success_count += 1
                        self.log_test(f"Prediction {chapter}", True, f"Score: {data.get('prediction', {}).get('score', 'N/A')}")
                    else:
                        self.log_test(f"Prediction {chapter}", False, "", "Invalid response structure")
                else:
                    self.log_test(f"Prediction {chapter}", False, "", f"Status: {response.status_code}")
                    
            except Exception as e:
                self.log_test(f"Prediction {chapter}", False, "", str(e))
        
        success_rate = success_count / len(chapters)
        if success_rate >= 0.8:
            self.log_test("Prediction Endpoints", True, f"{success_count}/{len(chapters)} chapters OK")
            return True
        else:
            self.log_test("Prediction Endpoints", False, "", f"Only {success_count}/{len(chapters)} chapters OK")
            return False
    
    def test_database_connection(self) -> bool:
        """Test 4: Connexion à la base de données"""
        try:
            response = requests.get(f"{self.base_url}/api/v1/postgresql/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and data.get("data", {}).get("status") == "connected":
                    self.log_test("Database Connection", True, "PostgreSQL connected")
                    return True
            
            self.log_test("Database Connection", False, "", f"Status: {response.status_code}")
            return False
        except Exception as e:
            self.log_test("Database Connection", False, "", str(e))
            return False
    
    def test_file_upload(self) -> bool:
        """Test 5: Upload de fichiers"""
        try:
            # Créer un fichier de test
            test_file_content = """DECLARATION_ID,VALEUR_CAF,POIDS_NET_KG,NOMBRE_COLIS,CODE_SH_COMPLET,PAYS_ORIGINE,BUREAU_DOUANE
TEST_UPLOAD_001,10000.0,500.0,10,30049000,FR,01A
TEST_UPLOAD_002,8000.0,400.0,8,30049000,DE,01A"""
            
            test_file_path = Path("test_upload.csv")
            test_file_path.write_text(test_file_content)
            
            # Test upload
            with open(test_file_path, 'rb') as f:
                files = {'file': ('test_upload.csv', f, 'text/csv')}
                response = requests.post(
                    f"{self.base_url}/api/v1/upload/chap30",
                    files=files,
                    timeout=30
                )
            
            # Nettoyer le fichier de test
            test_file_path.unlink(missing_ok=True)
            
            if response.status_code in [200, 201]:
                data = response.json()
                if data.get("success"):
                    self.log_test("File Upload", True, f"Processed: {data.get('processed_count', 0)} files")
                    return True
            
            self.log_test("File Upload", False, "", f"Status: {response.status_code}")
            return False
            
        except Exception as e:
            self.log_test("File Upload", False, "", str(e))
            return False
    
    def test_rl_feedback_loop(self) -> bool:
        """Test 6: Boucle de feedback RL"""
        try:
            feedback_data = {
                "declaration_id": "TEST_RL_001",
                "predicted_fraud": False,
                "predicted_probability": 0.3,
                "inspector_decision": True,
                "inspector_confidence": 0.8,
                "inspector_id": "INSP001",
                "additional_notes": "Test feedback automatique",
                "exploration_used": True
            }
            
            chapters = ["chap30", "chap84", "chap85"]
            success_count = 0
            
            for chapter in chapters:
                response = requests.post(
                    f"{self.base_url}/api/v1/rl/feedback/{chapter}",
                    json=feedback_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        success_count += 1
                        self.log_test(f"RL Feedback {chapter}", True, "Feedback submitted")
                    else:
                        self.log_test(f"RL Feedback {chapter}", False, "", "Invalid response")
                else:
                    self.log_test(f"RL Feedback {chapter}", False, "", f"Status: {response.status_code}")
            
            success_rate = success_count / len(chapters)
            if success_rate >= 0.8:
                self.log_test("RL Feedback Loop", True, f"{success_count}/{len(chapters)} chapters OK")
                return True
            else:
                self.log_test("RL Feedback Loop", False, "", f"Only {success_count}/{len(chapters)} chapters OK")
                return False
                
        except Exception as e:
            self.log_test("RL Feedback Loop", False, "", str(e))
            return False
    
    def test_frontend_flutter_tests(self) -> bool:
        """Test 7: Tests Flutter frontend"""
        try:
            # Vérifier si Flutter est installé
            flutter_check = subprocess.run(["flutter", "--version"], capture_output=True, text=True)
            if flutter_check.returncode != 0:
                self.log_test("Flutter Tests", False, "", "Flutter not installed")
                return False
            
            # Changer vers le répertoire frontend
            frontend_dir = Path("inspectia_app/inspectia_app_frontend")
            if not frontend_dir.exists():
                self.log_test("Flutter Tests", False, "", "Frontend directory not found")
                return False
            
            # Exécuter les tests Flutter
            os.chdir(frontend_dir)
            test_result = subprocess.run(
                ["flutter", "test", "test_frontend_backend.dart"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Retourner au répertoire principal
            os.chdir("../..")
            
            if test_result.returncode == 0:
                self.log_test("Flutter Tests", True, "All Flutter tests passed")
                return True
            else:
                self.log_test("Flutter Tests", False, "", f"Flutter test errors: {test_result.stderr}")
                return False
                
        except Exception as e:
            self.log_test("Flutter Tests", False, "", str(e))
            return False
    
    def test_performance_benchmarks(self) -> bool:
        """Test 8: Benchmarks de performance"""
        try:
            # Test de latence
            start_time = time.time()
            response = requests.get(f"{self.base_url}/api/v1/health", timeout=10)
            end_time = time.time()
            
            latency = end_time - start_time
            
            # Test de débit (multiple requêtes)
            start_time = time.time()
            for _ in range(10):
                requests.get(f"{self.base_url}/api/v1/chapters", timeout=5)
            end_time = time.time()
            
            throughput_time = end_time - start_time
            throughput = 10 / throughput_time  # requêtes par seconde
            
            # Critères de performance
            latency_ok = latency < 2.0  # moins de 2 secondes
            throughput_ok = throughput > 5.0  # plus de 5 req/s
            
            if latency_ok and throughput_ok:
                self.log_test("Performance Benchmarks", True, 
                            f"Latency: {latency:.2f}s, Throughput: {throughput:.1f} req/s")
                return True
            else:
                self.log_test("Performance Benchmarks", False, "", 
                            f"Latency: {latency:.2f}s (>2s), Throughput: {throughput:.1f} req/s (<5 req/s)")
                return False
                
        except Exception as e:
            self.log_test("Performance Benchmarks", False, "", str(e))
            return False
    
    def test_error_handling(self) -> bool:
        """Test 9: Gestion d'erreurs"""
        try:
            # Test endpoint inexistant
            response = requests.get(f"{self.base_url}/api/v1/nonexistent", timeout=10)
            if response.status_code == 404:
                error_handling_ok = True
            else:
                error_handling_ok = False
            
            # Test données invalides
            invalid_data = {"invalid": "data"}
            response = requests.post(f"{self.base_url}/api/v1/predict/chap30", 
                                   json=invalid_data, timeout=10)
            if response.status_code in [400, 422]:
                error_handling_ok = error_handling_ok and True
            else:
                error_handling_ok = False
            
            if error_handling_ok:
                self.log_test("Error Handling", True, "Proper error responses")
                return True
            else:
                self.log_test("Error Handling", False, "", "Incorrect error responses")
                return False
                
        except Exception as e:
            self.log_test("Error Handling", False, "", str(e))
            return False
    
    def test_data_consistency(self) -> bool:
        """Test 10: Cohérence des données"""
        try:
            # Test cohérence entre différents endpoints
            chapters_response = requests.get(f"{self.base_url}/api/v1/chapters", timeout=10)
            health_response = requests.get(f"{self.base_url}/api/v1/health", timeout=10)
            
            if chapters_response.status_code == 200 and health_response.status_code == 200:
                chapters_data = chapters_response.json()
                health_data = health_response.json()
                
                # Vérifier la cohérence
                if (chapters_data.get("success") and 
                    health_data.get("success") and 
                    len(chapters_data.get("chapters", [])) > 0):
                    
                    self.log_test("Data Consistency", True, 
                                f"Chapters: {len(chapters_data.get('chapters', []))}, Health: OK")
                    return True
            
            self.log_test("Data Consistency", False, "", "Inconsistent data between endpoints")
            return False
            
        except Exception as e:
            self.log_test("Data Consistency", False, "", str(e))
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Exécute tous les tests"""
        print("🚀 DÉMARRAGE DES TESTS COMPLETS DE COMMUNICATION FRONTEND-BACKEND")
        print("=" * 80)
        
        # Liste des tests à exécuter
        tests = [
            ("Backend Health", self.test_backend_health),
            ("Backend Endpoints", self.test_backend_endpoints),
            ("Prediction Endpoints", self.test_prediction_endpoints),
            ("Database Connection", self.test_database_connection),
            ("File Upload", self.test_file_upload),
            ("RL Feedback Loop", self.test_rl_feedback_loop),
            ("Flutter Tests", self.test_frontend_flutter_tests),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Error Handling", self.test_error_handling),
            ("Data Consistency", self.test_data_consistency),
        ]
        
        # Exécution des tests
        for test_name, test_func in tests:
            print(f"\n🧪 Test: {test_name}")
            try:
                test_func()
            except Exception as e:
                self.log_test(test_name, False, "", f"Test crashed: {str(e)}")
            time.sleep(1)  # Pause entre les tests
        
        # Résumé final
        self.print_summary()
        return self.results
    
    def print_summary(self):
        """Affiche le résumé des tests"""
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ DES TESTS DE COMMUNICATION FRONTEND-BACKEND")
        print("=" * 80)
        
        total = self.results["total_tests"]
        passed = self.results["passed_tests"]
        failed = self.results["failed_tests"]
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"📈 Total des tests: {total}")
        print(f"✅ Tests réussis: {passed}")
        print(f"❌ Tests échoués: {failed}")
        print(f"📊 Taux de réussite: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 EXCELLENT! Communication frontend-backend très stable")
        elif success_rate >= 80:
            print("👍 BIEN! Communication frontend-backend stable")
        elif success_rate >= 70:
            print("⚠️ ACCEPTABLE! Quelques problèmes mineurs détectés")
        else:
            print("🚨 PROBLÈME! Communication frontend-backend défaillante")
        
        # Détails des échecs
        if failed > 0:
            print("\n❌ TESTS ÉCHOUÉS:")
            for test in self.results["test_details"]:
                if not test["success"]:
                    print(f"  - {test['test_name']}: {test['error']}")
        
        print("\n" + "=" * 80)

def main():
    """Fonction principale"""
    tester = FrontendBackendCommunicationTester()
    
    print("🔧 Vérification de l'environnement...")
    
    # Vérifier si le backend est démarré
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code != 200:
            print("⚠️ Backend non démarré ou non accessible sur http://localhost:8000")
            print("💡 Démarrez le backend avec: cd backend && python -m uvicorn src.main:app --reload")
            return
    except:
        print("⚠️ Backend non démarré ou non accessible sur http://localhost:8000")
        print("💡 Démarrez le backend avec: cd backend && python -m uvicorn src.main:app --reload")
        return
    
    print("✅ Backend accessible, démarrage des tests...")
    
    # Exécuter tous les tests
    results = tester.run_all_tests()
    
    # Sauvegarder les résultats
    results_file = Path("test_communication_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Résultats sauvegardés dans: {results_file}")

if __name__ == "__main__":
    main()
