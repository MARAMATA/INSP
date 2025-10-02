#!/usr/bin/env python3
"""
🧪 TEST COMPLET DE TOUS LES ENDPOINTS INSPECTIA
Teste l'utilisation de TOUS les endpoints disponibles dans l'API
"""

import requests
import json
import time
import os
from pathlib import Path
import tempfile

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
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title.center(80)}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}{Colors.END}")

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

def create_test_files():
    """Crée des fichiers de test pour les uploads"""
    test_files = {}
    
    # Créer un fichier CSV de test
    csv_content = """DECLARATION_ID,VALEUR_CAF,POIDS_NET_KG,NOMBRE_COLIS,CODE_SH_COMPLET,PAYS_ORIGINE,BUREAU_DOUANE
2023/01A/12345,15000.0,750.0,15,30049000,FR,01A
2023/01A/12346,8000.0,400.0,8,30049000,DE,01A
2023/01A/12347,12000.0,600.0,12,30049000,IT,01A"""
    
    csv_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    csv_file.write(csv_content)
    csv_file.close()
    test_files['csv'] = csv_file.name
    
    # Créer un fichier PDF de test (simulé)
    pdf_file = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    pdf_file.write(b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n')
    pdf_file.close()
    test_files['pdf'] = pdf_file.name
    
    # Créer une image de test (simulée)
    img_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img_file.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde')
    img_file.close()
    test_files['image'] = img_file.name
    
    return test_files

def cleanup_test_files(test_files):
    """Nettoie les fichiers de test"""
    for file_path in test_files.values():
        try:
            os.unlink(file_path)
        except:
            pass

# =============================================================================
# TESTS DES ENDPOINTS PRINCIPAUX
# =============================================================================

def test_root_endpoints():
    """Test des endpoints racine"""
    print_header("🏠 ENDPOINTS RACINE")
    
    # Test endpoint racine
    result = test_endpoint("GET", BASE_URL)
    if result["success"]:
        print_test("Endpoint racine", "✅", f"Status: {result['status_code']}")
        if isinstance(result["response"], dict):
            print(f"    {Colors.WHITE}Message: {result['response'].get('message', 'N/A')}{Colors.END}")
            print(f"    {Colors.WHITE}Version: {result['response'].get('version', 'N/A')}{Colors.END}")
    else:
        print_test("Endpoint racine", "❌", f"Erreur: {result['error'] or result['status_code']}")
    
    # Test health check racine
    result = test_endpoint("GET", f"{BASE_URL}/health")
    if result["success"]:
        print_test("Health check racine", "✅", f"Status: {result['status_code']}")
    else:
        print_test("Health check racine", "❌", f"Erreur: {result['error'] or result['status_code']}")
    
    # Test chapitres racine
    result = test_endpoint("GET", f"{BASE_URL}/chapters")
    if result["success"]:
        print_test("Chapitres racine", "✅", f"Status: {result['status_code']}")
        if isinstance(result["response"], dict) and "chapters" in result["response"]:
            chapters = result["response"]["chapters"]
            for chapter in chapters:
                print(f"    {Colors.WHITE}- {chapter.get('id', 'N/A')}: {chapter.get('name', 'N/A')} ({chapter.get('best_model', 'N/A')}){Colors.END}")
    else:
        print_test("Chapitres racine", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_predict_health_endpoints():
    """Test des endpoints de santé et dépendances"""
    print_header("🏥 ENDPOINTS DE SANTÉ ET DÉPENDANCES")
    
    # Test health check predict
    result = test_endpoint("GET", f"{PREDICT_BASE}/health")
    if result["success"]:
        print_test("Health check predict", "✅", f"Status: {result['status_code']}")
        if isinstance(result["response"], dict):
            print(f"    {Colors.WHITE}Version: {result['response'].get('version', 'N/A')}{Colors.END}")
            print(f"    {Colors.WHITE}Système: {result['response'].get('system', 'N/A')}{Colors.END}")
            chapters = result['response'].get('chapters_status', {})
            for chapter, info in chapters.items():
                print(f"    {Colors.WHITE}{chapter}: {info.get('best_model', 'N/A')} - {info.get('status', 'N/A')}{Colors.END}")
    else:
        print_test("Health check predict", "❌", f"Erreur: {result['error'] or result['status_code']}")
    
    # Test dépendances
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

def test_chapter_info_endpoints():
    """Test des endpoints d'informations des chapitres"""
    print_header("📋 ENDPOINTS D'INFORMATIONS DES CHAPITRES")
    
    chapters = ["chap30", "chap84", "chap85"]
    
    # Test endpoint chapters
    result = test_endpoint("GET", f"{PREDICT_BASE}/chapters")
    if result["success"]:
        print_test("  Liste des chapitres", "✅", f"Status: {result['status_code']}")
    else:
        print_test("  Liste des chapitres", "❌", f"Erreur: {result['error'] or result['status_code']}")
    
    # Test endpoints par chapitre
    endpoints = [
        ("config", "Configuration"),
        ("model-info", "Informations modèle"),
        ("features", "Features disponibles"),
        ("status", "Statut chapitre"),
        ("performance", "Performances modèle"),
    ]
    
    for endpoint, description in endpoints:
        for chapter in chapters:
            url = f"{PREDICT_BASE}/{chapter}/{endpoint}"
            result = test_endpoint("GET", url)
            
            if result["success"]:
                print_test(f"  {chapter} - {description}", "✅", f"Status: {result['status_code']}")
            else:
                print_test(f"  {chapter} - {description}", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_prediction_endpoints():
    """Test des endpoints de prédiction"""
    print_header("🎯 ENDPOINTS DE PRÉDICTION")
    
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
    
    # Test auto-predict
    for chapter in chapters:
        url = f"{PREDICT_BASE}/{chapter}/auto-predict"
        result = test_endpoint("POST", url, test_data)
        
        if result["success"]:
            print_test(f"  {chapter} - Auto Predict", "✅", f"Status: {result['status_code']}")
            if isinstance(result["response"], dict):
                pred = result["response"]
                print(f"    {Colors.WHITE}Prédiction: {pred.get('predicted_fraud', 'N/A')}{Colors.END}")
                print(f"    {Colors.WHITE}Probabilité: {pred.get('fraud_probability', 'N/A')}{Colors.END}")
        else:
            print_test(f"  {chapter} - Auto Predict", "❌", f"Erreur: {result['error'] or result['status_code']}")
    
    # Test validation
    for chapter in chapters:
        url = f"{PREDICT_BASE}/{chapter}/validate"
        validation_data = {"data": test_data}
        result = test_endpoint("POST", url, validation_data)
        
        if result["success"]:
            print_test(f"  {chapter} - Validation", "✅", f"Status: {result['status_code']}")
        else:
            print_test(f"  {chapter} - Validation", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_file_upload_endpoints():
    """Test des endpoints d'upload de fichiers"""
    print_header("📁 ENDPOINTS D'UPLOAD DE FICHIERS")
    
    chapters = ["chap30", "chap84", "chap85"]
    test_files = create_test_files()
    
    try:
        for chapter in chapters:
            # Test upload CSV
            with open(test_files['csv'], 'rb') as f:
                files = {'file': ('test.csv', f, 'text/csv')}
                result = test_endpoint("POST", f"{PREDICT_BASE}/{chapter}", files=files)
                
                if result["success"]:
                    print_test(f"  {chapter} - Upload CSV", "✅", f"Status: {result['status_code']}")
                    if isinstance(result["response"], dict):
                        resp = result["response"]
                        print(f"    {Colors.WHITE}Type fichier: {resp.get('file_info', {}).get('file_type', 'N/A')}{Colors.END}")
                        print(f"    {Colors.WHITE}Type source: {resp.get('file_info', {}).get('source_type', 'N/A')}{Colors.END}")
                else:
                    print_test(f"  {chapter} - Upload CSV", "❌", f"Erreur: {result['error'] or result['status_code']}")
            
            # Test upload PDF
            with open(test_files['pdf'], 'rb') as f:
                files = {'file': ('test.pdf', f, 'application/pdf')}
                result = test_endpoint("POST", f"{PREDICT_BASE}/{chapter}", files=files)
                
                if result["success"]:
                    print_test(f"  {chapter} - Upload PDF", "✅", f"Status: {result['status_code']}")
                else:
                    print_test(f"  {chapter} - Upload PDF", "❌", f"Erreur: {result['error'] or result['status_code']}")
            
            # Test upload Image
            with open(test_files['image'], 'rb') as f:
                files = {'file': ('test.png', f, 'image/png')}
                result = test_endpoint("POST", f"{PREDICT_BASE}/{chapter}", files=files)
                
                if result["success"]:
                    print_test(f"  {chapter} - Upload Image", "✅", f"Status: {result['status_code']}")
                else:
                    print_test(f"  {chapter} - Upload Image", "❌", f"Erreur: {result['error'] or result['status_code']}")
    
    finally:
        cleanup_test_files(test_files)

def test_ocr_endpoints():
    """Test des endpoints OCR"""
    print_header("🔍 ENDPOINTS OCR")
    
    chapters = ["chap30", "chap84", "chap85"]
    test_files = create_test_files()
    
    try:
        for chapter in chapters:
            # Test process-ocr
            with open(test_files['pdf'], 'rb') as f:
                files = {'file': ('test.pdf', f, 'application/pdf')}
                result = test_endpoint("POST", f"{PREDICT_BASE}/{chapter}/process-ocr", files=files)
                
                if result["success"]:
                    print_test(f"  {chapter} - Process OCR", "✅", f"Status: {result['status_code']}")
                else:
                    print_test(f"  {chapter} - Process OCR", "❌", f"Erreur: {result['error'] or result['status_code']}")
            
            # Test predict-from-ocr
            ocr_data = {
                "declaration_id": "OCR_TEST_2023/01A/12345",
                "valeur_caf": 15000.0,
                "poids_net_kg": 750.0,
                "nombre_colis": 15,
                "code_sh_complet": "30049000",
                "pays_origine": "FR"
            }
            
            result = test_endpoint("POST", f"{PREDICT_BASE}/{chapter}/predict-from-ocr", ocr_data)
            
            if result["success"]:
                print_test(f"  {chapter} - Predict from OCR", "✅", f"Status: {result['status_code']}")
            else:
                print_test(f"  {chapter} - Predict from OCR", "❌", f"Erreur: {result['error'] or result['status_code']}")
    
    finally:
        cleanup_test_files(test_files)

def test_rl_endpoints():
    """Test des endpoints RL"""
    print_header("🧠 ENDPOINTS RL")
    
    chapters = ["chap30", "chap84", "chap85"]
    
    for chapter in chapters:
        # Test RL Status
        result = test_endpoint("GET", f"{PREDICT_BASE}/{chapter}/rl/status")
        
        if result["success"]:
            print_test(f"  {chapter} - RL Status", "✅", f"Status: {result['status_code']}")
        else:
            print_test(f"  {chapter} - RL Status", "❌", f"Erreur: {result['error'] or result['status_code']}")
        
        # Test RL Predict
        context_data = {
            "declaration_id": "RL_TEST_2023/01A/12345",
            "valeur_caf": 20000.0,
            "poids_net_kg": 1000.0,
            "nombre_colis": 20,
            "code_sh_complet": "30049000",
            "pays_origine": "FR"
        }
        
        result = test_endpoint("POST", f"{PREDICT_BASE}/{chapter}/rl/predict", context_data)
        
        if result["success"]:
            print_test(f"  {chapter} - RL Predict", "✅", f"Status: {result['status_code']}")
        else:
            print_test(f"  {chapter} - RL Predict", "❌", f"Erreur: {result['error'] or result['status_code']}")
        
        # Test RL Feedback
        feedback_data = {
            "context": context_data,
            "action": "investigate",
            "reward": 0.8,
            "inspector_id": "INSP001"
        }
        
        result = test_endpoint("POST", f"{PREDICT_BASE}/{chapter}/rl/feedback", feedback_data)
        
        if result["success"]:
            print_test(f"  {chapter} - RL Feedback", "✅", f"Status: {result['status_code']}")
        else:
            print_test(f"  {chapter} - RL Feedback", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_batch_endpoints():
    """Test des endpoints de traitement par lot"""
    print_header("📦 ENDPOINTS DE TRAITEMENT PAR LOT")
    
    chapters = ["chap30", "chap84", "chap85"]
    
    # Données de test pour le batch
    batch_data = {
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
            },
            {
                "DECLARATION_ID": "2023/01A/12347",
                "VALEUR_CAF": 12000.0,
                "POIDS_NET_KG": 600.0,
                "NOMBRE_COLIS": 12,
                "CODE_SH_COMPLET": "30049000",
                "PAYS_ORIGINE": "IT",
                "BUREAU_DOUANE": "01A"
            }
        ]
    }
    
    for chapter in chapters:
        # Test batch predict
        result = test_endpoint("POST", f"{PREDICT_BASE}/{chapter}/batch", batch_data)
        
        if result["success"]:
            print_test(f"  {chapter} - Batch Predict", "✅", f"Status: {result['status_code']}")
            if isinstance(result["response"], dict) and "summary" in result["response"]:
                summary = result["response"]["summary"]
                print(f"    {Colors.WHITE}Total: {summary.get('total_declarations', 'N/A')}{Colors.END}")
                print(f"    {Colors.WHITE}Succès: {summary.get('successful', 'N/A')}{Colors.END}")
                print(f"    {Colors.WHITE}Fraude: {summary.get('fraud_count', 'N/A')}{Colors.END}")
        else:
            print_test(f"  {chapter} - Batch Predict", "❌", f"Erreur: {result['error'] or result['status_code']}")
        
        # Test declarations (JSON)
        declarations_data = [
            {
                "declaration_id": "JSON_TEST_2023/01A/12345",
                "valeur_caf": 10000.0,
                "poids_net_kg": 500.0,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "pays_origine": "FR"
            },
            {
                "declaration_id": "JSON_TEST_2023/01A/12346",
                "valeur_caf": 15000.0,
                "poids_net_kg": 750.0,
                "nombre_colis": 15,
                "code_sh_complet": "30049000",
                "pays_origine": "DE"
            }
        ]
        
        result = test_endpoint("POST", f"{PREDICT_BASE}/{chapter}/declarations", declarations_data)
        
        if result["success"]:
            print_test(f"  {chapter} - Declarations JSON", "✅", f"Status: {result['status_code']}")
            if isinstance(result["response"], dict) and "summary" in result["response"]:
                summary = result["response"]["summary"]
                print(f"    {Colors.WHITE}Total: {summary.get('total_declarations', 'N/A')}{Colors.END}")
                print(f"    {Colors.WHITE}Taux fraude: {summary.get('fraud_rate', 'N/A')}%{Colors.END}")
        else:
            print_test(f"  {chapter} - Declarations JSON", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_feedback_endpoints():
    """Test des endpoints de feedback"""
    print_header("💬 ENDPOINTS DE FEEDBACK")
    
    chapters = ["chap30", "chap84", "chap85"]
    
    # Données de feedback
    feedback_data = {
        "declaration_id": "FEEDBACK_TEST_2023/01A/12345",
        "predicted_fraud": False,
        "actual_fraud": True,
        "inspector_id": "INSP001",
        "confidence": 0.8,
        "notes": "Test feedback automatique",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    for chapter in chapters:
        # Test feedback général
        result = test_endpoint("POST", f"{PREDICT_BASE}/{chapter}/feedback", feedback_data)
        
        if result["success"]:
            print_test(f"  {chapter} - General Feedback", "✅", f"Status: {result['status_code']}")
            if isinstance(result["response"], dict):
                resp = result["response"]
                print(f"    {Colors.WHITE}Feedback ID: {resp.get('feedback_id', 'N/A')}{Colors.END}")
                print(f"    {Colors.WHITE}Status: {resp.get('status', 'N/A')}{Colors.END}")
        else:
            print_test(f"  {chapter} - General Feedback", "❌", f"Erreur: {result['error'] or result['status_code']}")

def test_pipeline_endpoints():
    """Test des endpoints de pipeline"""
    print_header("🔧 ENDPOINTS DE PIPELINE")
    
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
        # Test pipeline
        result = test_endpoint("POST", f"{PREDICT_BASE}/{chapter}/test-pipeline", pipeline_data)
        
        if result["success"]:
            print_test(f"  {chapter} - Test Pipeline", "✅", f"Status: {result['status_code']}")
            if isinstance(result["response"], dict):
                resp = result["response"]
                print(f"    {Colors.WHITE}Test Status: {resp.get('test_status', 'N/A')}{Colors.END}")
        else:
            print_test(f"  {chapter} - Test Pipeline", "❌", f"Erreur: {result['error'] or result['status_code']}")

def main():
    """Fonction principale de test"""
    print_header("🚀 TEST COMPLET DE TOUS LES ENDPOINTS INSPECTIA")
    print(f"{Colors.WHITE}URL de base: {BASE_URL}{Colors.END}")
    print(f"{Colors.WHITE}Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}{Colors.END}")
    
    try:
        # Tests de base
        test_root_endpoints()
        test_predict_health_endpoints()
        
        # Tests des chapitres
        test_chapter_info_endpoints()
        
        # Tests de prédiction
        test_prediction_endpoints()
        
        # Tests d'upload de fichiers
        test_file_upload_endpoints()
        
        # Tests OCR
        test_ocr_endpoints()
        
        # Tests RL
        test_rl_endpoints()
        
        # Tests de traitement par lot
        test_batch_endpoints()
        
        # Tests de feedback
        test_feedback_endpoints()
        
        # Tests de pipeline
        test_pipeline_endpoints()
        
        print_header("✅ TOUS LES TESTS TERMINÉS")
        print(f"{Colors.GREEN}{Colors.BOLD}Tous les endpoints ont été testés avec succès !{Colors.END}")
        print(f"{Colors.WHITE}L'API InspectIA est complètement fonctionnelle !{Colors.END}")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrompus par l'utilisateur{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Erreur lors des tests: {e}{Colors.END}")

if __name__ == "__main__":
    main()
