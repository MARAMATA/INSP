#!/usr/bin/env python3
"""
🧪 SCRIPT DE TEST COMPLET DES COMMUNICATIONS FRONTEND-BACKEND
Teste tous les endpoints et fonctionnalités d'InspectIA
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
PREDICT_BASE = f"{BASE_URL}/predict"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title):
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title.center(60)}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.END}")

def print_test(test_name, status, details=""):
    status_color = Colors.GREEN if status == "✅" else Colors.RED
    print(f"{status_color}{status}{Colors.END} {test_name}")
    if details:
        print(f"    {Colors.WHITE}{details}{Colors.END}")

def test_endpoint(method, url, data=None, files=None, expected_status=200):
    """Test un endpoint et retourne le résultat"""
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        elif method.upper() == "POST":
            if files:
                response = requests.post(url, files=files, timeout=10)
            else:
                response = requests.post(url, json=data, timeout=10)
        
        return {
            "success": response.status_code == expected_status,
            "status_code": response.status_code,
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
            "error": None
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "status_code": None,
            "response": None,
            "error": str(e)
        }

def test_backend_health():
    """Test de santé du backend"""
    print_header("🏥 TEST DE SANTÉ DU BACKEND")
    
    result = test_endpoint("GET", f"{PREDICT_BASE}/health")
    
    if result["success"]:
        print_test("Health Check", "✅", f"Status: {result['status_code']}")
        if isinstance(result["response"], dict):
            print(f"    {Colors.WHITE}Version: {result['response'].get('version', 'N/A')}{Colors.END}")
            print(f"    {Colors.WHITE}Système: {result['response'].get('system', 'N/A')}{Colors.END}")
            chapters = result['response'].get('chapters_status', {})
            for chapter, info in chapters.items():
                print(f"    {Colors.WHITE}{chapter}: {info.get('best_model', 'N/A')} - {info.get('status', 'N/A')}{Colors.END}")
    else:
        print_test("Health Check", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_chapters_endpoint():
    """Test de l'endpoint des chapitres"""
    print_header("📋 TEST DES CHAPITRES DISPONIBLES")
    
    result = test_endpoint("GET", f"{PREDICT_BASE}/chapters")
    
    if result["success"]:
        print_test("Chapitres disponibles", "✅", f"Status: {result['status_code']}")
        if isinstance(result["response"], dict) and "chapters" in result["response"]:
            chapters = result["response"]["chapters"]
            for chapter in chapters:
                if isinstance(chapter, dict):
                    print(f"    {Colors.WHITE}- {chapter.get('id', 'N/A')}: {chapter.get('title', 'N/A')}{Colors.END}")
                else:
                    print(f"    {Colors.WHITE}- {chapter}{Colors.END}")
    else:
        print_test("Chapitres disponibles", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_dependencies():
    """Test des dépendances système"""
    print_header("🔧 TEST DES DÉPENDANCES SYSTÈME")
    
    result = test_endpoint("GET", f"{PREDICT_BASE}/dependencies")
    
    if result["success"]:
        print_test("Dépendances système", "✅", f"Status: {result['status_code']}")
        if isinstance(result["response"], dict):
            deps = result["response"]
            for dep, status in deps.items():
                status_icon = "✅" if status else "❌"
                print(f"    {Colors.WHITE}{status_icon} {dep}{Colors.END}")
    else:
        print_test("Dépendances système", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_chapter_endpoints():
    """Test des endpoints spécifiques aux chapitres"""
    print_header("🎯 TEST DES ENDPOINTS PAR CHAPITRE")
    
    chapters = ["chap30", "chap84", "chap85"]
    endpoints = [
        ("config", "Configuration"),
        ("model-info", "Informations modèle"),
        ("features", "Features disponibles"),
        ("status", "Statut chapitre"),
        ("performance", "Performances modèle"),
    ]
    
    for chapter in chapters:
        print(f"\n{Colors.PURPLE}{Colors.BOLD}Chapitre {chapter}:{Colors.END}")
        
        for endpoint, description in endpoints:
            url = f"{PREDICT_BASE}/{chapter}/{endpoint}"
            result = test_endpoint("GET", url)
            
            if result["success"]:
                print_test(f"  {description}", "✅", f"Status: {result['status_code']}")
            else:
                print_test(f"  {description}", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_rl_endpoints():
    """Test des endpoints RL"""
    print_header("🧠 TEST DES ENDPOINTS RL")
    
    chapters = ["chap30", "chap84", "chap85"]
    
    for chapter in chapters:
        print(f"\n{Colors.PURPLE}{Colors.BOLD}Chapitre {chapter} - RL:{Colors.END}")
        
        # Test RL Status
        url = f"{PREDICT_BASE}/{chapter}/rl/status"
        result = test_endpoint("GET", url)
        
        if result["success"]:
            print_test("  RL Status", "✅", f"Status: {result['status_code']}")
        else:
            print_test("  RL Status", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_prediction_endpoints():
    """Test des endpoints de prédiction"""
    print_header("🎯 TEST DES ENDPOINTS DE PRÉDICTION")
    
    chapters = ["chap30", "chap84", "chap85"]
    
    # Données de test
    test_data = {
        "declaration_id": "TEST_2023/01A/12345",
        "valeur_caf": 10000.0,
        "poids_net_kg": 500.0,
        "nombre_colis": 10,
        "code_sh_complet": "30049000",
        "pays_origine": "FR",
        "bureau_douane": "01A"
    }
    
    for chapter in chapters:
        print(f"\n{Colors.PURPLE}{Colors.BOLD}Chapitre {chapter} - Prédiction:{Colors.END}")
        
        # Test Auto Predict
        url = f"{PREDICT_BASE}/{chapter}/auto-predict"
        result = test_endpoint("POST", url, test_data)
        
        if result["success"]:
            print_test("  Auto Predict", "✅", f"Status: {result['status_code']}")
            if isinstance(result["response"], dict):
                pred = result["response"]
                print(f"    {Colors.WHITE}Prédiction: {pred.get('predicted_fraud', 'N/A')}{Colors.END}")
                print(f"    {Colors.WHITE}Probabilité: {pred.get('fraud_probability', 'N/A')}{Colors.END}")
        else:
            print_test("  Auto Predict", "❌", f"Erreur: {result['error'] or result['status_code']}")
        
        # Test Validation
        url = f"{PREDICT_BASE}/{chapter}/validate"
        result = test_endpoint("POST", url, test_data)
        
        if result["success"]:
            print_test("  Validation", "✅", f"Status: {result['status_code']}")
        else:
            print_test("  Validation", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_feedback_endpoints():
    """Test des endpoints de feedback"""
    print_header("💬 TEST DES ENDPOINTS DE FEEDBACK")
    
    chapters = ["chap30", "chap84", "chap85"]
    
    # Données de feedback
    feedback_data = {
        "declaration_id": "TEST_2023/01A/12345",
        "predicted_fraud": False,
        "predicted_probability": 0.3,
        "inspector_decision": True,
        "inspector_confidence": 0.8,
        "inspector_id": "INSP001",
        "additional_notes": "Test feedback automatique",
        "exploration_used": True
    }
    
    for chapter in chapters:
        print(f"\n{Colors.PURPLE}{Colors.BOLD}Chapitre {chapter} - Feedback:{Colors.END}")
        
        # Test RL Feedback
        url = f"{PREDICT_BASE}/{chapter}/rl/feedback"
        result = test_endpoint("POST", url, feedback_data)
        
        if result["success"]:
            print_test("  RL Feedback", "✅", f"Status: {result['status_code']}")
        else:
            print_test("  RL Feedback", "❌", f"Erreur: {result['error'] or result['status_code']}")
        
        # Test General Feedback
        url = f"{PREDICT_BASE}/{chapter}/feedback"
        result = test_endpoint("POST", url, feedback_data)
        
        if result["success"]:
            print_test("  General Feedback", "✅", f"Status: {result['status_code']}")
        else:
            print_test("  General Feedback", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_aggregation_features():
    """Test des fonctionnalités d'agrégation"""
    print_header("🔄 TEST DES FONCTIONNALITÉS D'AGRÉGATION")
    
    # Test avec données agrégées
    aggregated_data = {
        "declarations": [
            {
                "DECLARATION_ID": "2023/01A/12345",
                "VALEUR_CAF": 15000.0,
                "POIDS_NET_KG": 750.0,
                "NOMBRE_COLIS": 15,
                "CODE_SH_COMPLET": "30049000",
                "PAYS_ORIGINE": "FR",
                "BUREAU_DOUANE": "01A"
            },
            {
                "DECLARATION_ID": "2023/01A/12346",
                "VALEUR_CAF": 8000.0,
                "POIDS_NET_KG": 400.0,
                "NOMBRE_COLIS": 8,
                "CODE_SH_COMPLET": "30049000",
                "PAYS_ORIGINE": "DE",
                "BUREAU_DOUANE": "01A"
            }
        ]
    }
    
    chapters = ["chap30", "chap84", "chap85"]
    
    for chapter in chapters:
        print(f"\n{Colors.PURPLE}{Colors.BOLD}Chapitre {chapter} - Agrégation:{Colors.END}")
        
        # Test Batch Predict (agrégation)
        url = f"{PREDICT_BASE}/{chapter}/batch"
        result = test_endpoint("POST", url, aggregated_data)
        
        if result["success"]:
            print_test("  Batch Predict", "✅", f"Status: {result['status_code']}")
            if isinstance(result["response"], dict) and "results" in result["response"]:
                results = result["response"]["results"]
                print(f"    {Colors.WHITE}Résultats: {len(results)} déclarations traitées{Colors.END}")
        else:
            print_test("  Batch Predict", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_pipeline_endpoints():
    """Test des endpoints de pipeline"""
    print_header("🔧 TEST DES ENDPOINTS DE PIPELINE")
    
    chapters = ["chap30", "chap84", "chap85"]
    
    # Données de test pour le pipeline
    pipeline_data = {
        "declaration_id": "PIPELINE_TEST_2023/01A/12345",
        "valeur_caf": 12000.0,
        "poids_net_kg": 600.0,
        "nombre_colis": 12,
        "code_sh_complet": "30049000",
        "pays_origine": "FR",
        "bureau_douane": "01A"
    }
    
    for chapter in chapters:
        print(f"\n{Colors.PURPLE}{Colors.BOLD}Chapitre {chapter} - Pipeline:{Colors.END}")
        
        # Test Pipeline
        url = f"{PREDICT_BASE}/{chapter}/test-pipeline"
        result = test_endpoint("POST", url, pipeline_data)
        
        if result["success"]:
            print_test("  Test Pipeline", "✅", f"Status: {result['status_code']}")
        else:
            print_test("  Test Pipeline", "❌", f"Erreur: {result['error'] or result['status_code']}")

def main():
    """Fonction principale de test"""
    print_header("🚀 TEST COMPLET DES COMMUNICATIONS FRONTEND-BACKEND")
    print(f"{Colors.WHITE}URL de base: {BASE_URL}{Colors.END}")
    print(f"{Colors.WHITE}Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    
    try:
        # Tests de base
        test_backend_health()
        test_chapters_endpoint()
        test_dependencies()
        
        # Tests des chapitres
        test_chapter_endpoints()
        test_rl_endpoints()
        
        # Tests de prédiction
        test_prediction_endpoints()
        test_feedback_endpoints()
        
        # Tests d'agrégation
        test_aggregation_features()
        
        # Tests de pipeline
        test_pipeline_endpoints()
        
        print_header("✅ TESTS TERMINÉS")
        print(f"{Colors.GREEN}{Colors.BOLD}Tous les tests ont été exécutés avec succès !{Colors.END}")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrompus par l'utilisateur{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Erreur lors des tests: {e}{Colors.END}")

if __name__ == "__main__":
    main()
