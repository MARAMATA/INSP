#!/usr/bin/env python3
"""
Tests unitaires pour les API Routes
Tests avec mocks pour isoler les composants
"""
import unittest
import sys
import os
import json
import tempfile
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Ajouter le chemin du projet
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.routes_predict import router
from fastapi import FastAPI

# Créer une application de test
app = FastAPI()
app.include_router(router)
client = TestClient(app)

class TestAPIRoutes(unittest.TestCase):
    """Tests unitaires pour les API Routes"""
    
    def setUp(self):
        """Configuration initiale pour chaque test"""
        # Données de test
        self.test_declaration = {
            "numero_declaration": "TEST001",
            "date_declaration": "2024-01-15",
            "valeur_fob": 1000.0,
            "pays_origine": "FR",
            "bureau_douane": "TEST",
            "chapitre": "30"
        }
        
        self.test_hybrid_data = {
            "score_ml": 0.75,
            "score_metier": 0.60,
            "inconsistencies": ["valeur_fob_suspecte"],
            "business_rules": {"pays_risque": True}
        }
    
    @patch('api.routes_predict._validate_chapter')
    @patch('api.routes_predict._load_yaml_config')
    def test_get_hybrid_config_success(self, mock_load_config, mock_validate):
        """Test de récupération réussie de la configuration hybride"""
        # Configurer les mocks
        mock_validate.return_value = None  # Pas d'exception
        mock_load_config.return_value = {
            "hybrid": {
                "method_default": "HYBRID_SCORE_BASED",
                "seuils": {"ml_conforme": 0.30, "ml_fraude": 0.65}
            }
        }
        
        # Tester l'endpoint
        response = client.get("/predict/chap30/hybrid-config")
        
        # Vérifications
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("hybrid", data)
        self.assertEqual(data["hybrid"]["method_default"], "HYBRID_SCORE_BASED")
        
        # Vérifier que les mocks ont été appelés
        mock_validate.assert_called_once_with("chap30")
        mock_load_config.assert_called_once_with("chap30")
    
    @patch('api.routes_predict._validate_chapter')
    def test_get_hybrid_config_invalid_chapter(self, mock_validate):
        """Test de récupération avec chapitre invalide"""
        # Configurer le mock pour lever une exception
        mock_validate.side_effect = HTTPException(status_code=400, detail="Chapitre invalide")
        
        # Tester l'endpoint
        response = client.get("/predict/invalid_chapter/hybrid-config")
        
        # Vérifications
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
    
    @patch('api.routes_predict._validate_chapter')
    @patch('api.routes_predict._load_yaml_config')
    def test_get_hybrid_methods_success(self, mock_load_config, mock_validate):
        """Test de récupération réussie des méthodes hybrides"""
        # Configurer les mocks
        mock_validate.return_value = None
        mock_load_config.return_value = {
            "hybrid": {
                "method_default": "HYBRID_SCORE_BASED",
                "available_methods": ["HYBRID_SCORE_BASED", "DOUBLE_SEUIL"]
            }
        }
        
        # Tester l'endpoint
        response = client.get("/predict/chap30/hybrid-methods")
        
        # Vérifications
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("methods", data)
        self.assertIn("available_methods", data["methods"])
        
        # Vérifier que les mocks ont été appelés
        mock_validate.assert_called_once_with("chap30")
        mock_load_config.assert_called_once_with("chap30")
    
    @patch('api.routes_predict._validate_chapter')
    @patch('api.routes_predict._load_yaml_config')
    @patch('api.routes_predict.predict_from_csv')
    def test_predict_endpoint_success(self, mock_predict_csv, mock_load_config, mock_validate):
        """Test de prédiction réussie"""
        # Configurer les mocks
        mock_validate.return_value = None
        mock_load_config.return_value = {"test": "config"}
        mock_predict_csv.return_value = [
            {
                "fraud_probability": 0.8,
                "hybrid_decision": "FRAUDE",
                "hybrid_score_ml": 0.8,
                "hybrid_score_metier": 0.7,
                "hybrid_score_final": 0.75,
                "hybrid_confidence": 0.9
            }
        ]
        
        # Créer un fichier de test temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("numero_declaration,valeur_fob,pays_origine,bureau_douane\n")
            f.write("TEST001,1000.0,FR,TEST\n")
            temp_file_path = f.name
        
        try:
            # Tester l'endpoint avec un fichier
            with open(temp_file_path, 'rb') as file:
                response = client.post(
                    "/predict/chap30",
                    files={"file": ("test.csv", file, "text/csv")}
                )
            
            # Vérifications
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("predictions", data)
            self.assertIn("summary", data)
            
            # Vérifier que les mocks ont été appelés
            mock_validate.assert_called_once_with("chap30")
            mock_predict_csv.assert_called_once()
        finally:
            # Nettoyer le fichier temporaire
            os.unlink(temp_file_path)
    
    @patch('api.routes_predict._validate_chapter')
    def test_predict_endpoint_invalid_data(self, mock_validate):
        """Test de prédiction avec données invalides"""
        # Configurer le mock
        mock_validate.return_value = None
        
        # Tester avec des données invalides
        invalid_data = {"declarations": []}  # Liste vide
        
        response = client.post("/predict/chap30", json=invalid_data)
        
        # Vérifications
        self.assertEqual(response.status_code, 422)  # Validation error
    
    @patch('api.routes_predict._validate_chapter')
    @patch('api.routes_predict._load_yaml_config')
    def test_test_hybrid_endpoint_success(self, mock_load_config, mock_validate):
        """Test de l'endpoint de test hybride"""
        # Configurer les mocks
        mock_validate.return_value = None
        mock_load_config.return_value = {
            "hybrid": {
                "classification": {
                    "seuils_hybrides": {
                        "conforme_max": 0.30,
                        "fraude_min": 0.65
                    }
                }
            }
        }
        
        # Tester l'endpoint
        response = client.post("/predict/chap30/test-hybrid", json=self.test_hybrid_data)
        
        # Vérifications
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("hybrid_result", data)
        
        # Vérifier que les mocks ont été appelés
        mock_validate.assert_called_once_with("chap30")
        mock_load_config.assert_called_once_with("chap30")
    
    @patch('api.routes_predict._validate_chapter')
    @patch('api.routes_predict._load_yaml_config')
    def test_test_hybrid_endpoint_missing_config(self, mock_load_config, mock_validate):
        """Test de l'endpoint de test hybride avec configuration manquante"""
        # Configurer les mocks
        mock_validate.return_value = None
        mock_load_config.return_value = {
            "hybrid": {
                # Configuration incomplète
            }
        }
        
        # Tester l'endpoint
        response = client.post("/predict/chap30/test-hybrid", json=self.test_hybrid_data)
        
        # Vérifications
        self.assertEqual(response.status_code, 500)  # Erreur interne
        data = response.json()
        self.assertIn("detail", data)
    
    @patch('api.routes_predict._validate_chapter')
    @patch('api.routes_predict._load_yaml_config')
    @patch('api.routes_predict.PVGeneratorComplet')
    def test_generate_pv_endpoint_success(self, mock_pv_generator_class, mock_load_config, mock_validate):
        """Test de génération de PV réussie"""
        # Configurer les mocks
        mock_validate.return_value = None
        mock_load_config.return_value = {"test": "config"}
        
        # Mock du générateur de PV avec un objet qui peut être sérialisé
        from dataclasses import dataclass
        
        @dataclass
        class MockPVReport:
            pv_id: str
            pv_content: str
            inspecteur_nom: str
            bureau_douane: str
            date_creation: str
        
        mock_pv_generator = Mock()
        mock_pv_generator.generer_pv.return_value = MockPVReport(
            pv_id="PV_TEST",
            pv_content="Contenu du PV",
            inspecteur_nom="Test Analyst",
            bureau_douane="Bureau Principal",
            date_creation="2024-01-15"
        )
        mock_pv_generator.exporter_pv_json_complet.return_value = "test_path.json"
        mock_pv_generator_class.return_value = mock_pv_generator
        
        # Données de test pour la génération de PV
        pv_data = {
            "predictions": [{"hybrid_decision": "FRAUDE"}],
            "chapter": "chap30",
            "analyst_name": "Test Analyst"
        }
        
        # Tester l'endpoint
        response = client.post("/predict/chap30/generate-pv", json=pv_data)
        
        # Vérifications
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("pv_id", data)
        
        # Vérifier que les mocks ont été appelés
        mock_validate.assert_called_once_with("chap30")
        mock_pv_generator.generer_pv.assert_called_once()
    
    def test_health_endpoint(self):
        """Test de l'endpoint de santé"""
        response = client.get("/predict/health")
        
        # Vérifications
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
    
    @patch('api.routes_predict._validate_chapter')
    @patch('api.routes_predict._load_yaml_config')
    def test_config_endpoint_success(self, mock_load_config, mock_validate):
        """Test de l'endpoint de configuration"""
        # Configurer les mocks
        mock_validate.return_value = None
        mock_load_config.return_value = {"test": "config"}
        
        # Tester l'endpoint
        response = client.get("/predict/chap30/config")
        
        # Vérifications
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # La réponse réelle contient chapter, config, sections_count et timestamp
        self.assertIn("chapter", data)
        self.assertIn("config", data)
        self.assertIn("sections_count", data)
        self.assertIn("timestamp", data)
        self.assertEqual(data["chapter"], "chap30")
        self.assertEqual(data["config"], {"test": "config"})
        
        # Vérifier que les mocks ont été appelés
        mock_validate.assert_called_once_with("chap30")
        mock_load_config.assert_called_once_with("chap30")
    
    def test_error_handling(self):
        """Test de la gestion des erreurs"""
        # Test avec un endpoint inexistant
        response = client.get("/predict/chap30/nonexistent")
        
        # Vérifications
        self.assertEqual(response.status_code, 404)
    
    def test_request_validation(self):
        """Test de la validation des requêtes"""
        # Test avec des données JSON invalides
        response = client.post("/predict/chap30", data="invalid json")
        
        # Vérifications
        self.assertEqual(response.status_code, 422)
    
    @patch('api.routes_predict._validate_chapter')
    @patch('api.routes_predict._load_yaml_config')
    def test_cache_endpoints(self, mock_load_config, mock_validate):
        """Test des endpoints de cache"""
        # Configurer les mocks
        mock_validate.return_value = None
        mock_load_config.return_value = {"test": "config"}
        
        # Test du statut du cache
        response = client.get("/predict/cache/status")
        self.assertEqual(response.status_code, 200)
        
        # Test du nettoyage du cache
        response = client.post("/predict/cache/clear")
        self.assertEqual(response.status_code, 200)
        
        # Test du préchargement du cache
        response = client.post("/predict/cache/preload", json=["chap30"])
        self.assertEqual(response.status_code, 200)

class TestAPIRoutesIntegration(unittest.TestCase):
    """Tests d'intégration pour les API Routes"""
    
    def setUp(self):
        """Configuration pour les tests d'intégration"""
        self.test_data = {
            "declarations": [
                {
                    "numero_declaration": "TEST001",
                    "valeur_fob": 1000.0,
                    "pays_origine": "FR",
                    "bureau_douane": "TEST1"
                },
                {
                    "numero_declaration": "TEST002",
                    "valeur_fob": 5000.0,
                    "pays_origine": "DE",
                    "bureau_douane": "TEST2"
                }
            ]
        }
    
    @patch('api.routes_predict._validate_chapter')
    @patch('api.routes_predict._load_yaml_config')
    @patch('api.routes_predict.predict_from_csv')
    def test_complete_prediction_workflow(self, mock_predict_csv, mock_load_config, mock_validate):
        """Test du workflow de prédiction complet"""
        # Configurer les mocks
        mock_validate.return_value = None
        mock_load_config.return_value = {"test": "config"}
        mock_predict_csv.return_value = [
            {
                "fraud_probability": 0.8,
                "hybrid_decision": "FRAUDE",
                "hybrid_score_ml": 0.8,
                "hybrid_score_metier": 0.7,
                "hybrid_score_final": 0.75,
                "hybrid_confidence": 0.9
            },
            {
                "fraud_probability": 0.3,
                "hybrid_decision": "CONFORME",
                "hybrid_score_ml": 0.3,
                "hybrid_score_metier": 0.2,
                "hybrid_score_final": 0.25,
                "hybrid_confidence": 0.9
            }
        ]
        
        # Créer un fichier de test temporaire
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("numero_declaration,valeur_fob,pays_origine,bureau_douane\n")
            f.write("TEST001,1000.0,FR,TEST1\n")
            f.write("TEST002,5000.0,DE,TEST2\n")
            temp_file_path = f.name
        
        try:
            # 1. Tester la prédiction
            with open(temp_file_path, 'rb') as file:
                response = client.post(
                    "/predict/chap30",
                    files={"file": ("test.csv", file, "text/csv")}
                )
            self.assertEqual(response.status_code, 200)
            
            # 2. Tester la récupération de la configuration
            response = client.get("/predict/chap30/config")
            self.assertEqual(response.status_code, 200)
            
            # 3. Tester la récupération des méthodes hybrides
            response = client.get("/predict/chap30/hybrid-methods")
            self.assertEqual(response.status_code, 200)
            
            # Vérifier que tous les mocks ont été appelés
            mock_validate.assert_called()
            mock_load_config.assert_called()
            mock_predict_csv.assert_called_once()
        finally:
            # Nettoyer le fichier temporaire
            os.unlink(temp_file_path)
    
    def test_chapter_validation_workflow(self):
        """Test du workflow de validation des chapitres"""
        # Test avec des chapitres valides
        valid_chapters = ["chap30", "chap84", "chap85"]
        
        for chapter in valid_chapters:
            with self.subTest(chapter=chapter):
                # Tester l'accès à la configuration
                response = client.get(f"/predict/{chapter}/config")
                # Peut retourner 404 si le chapitre n'existe pas, mais pas 400 (invalide)
                self.assertNotEqual(response.status_code, 400)
        
        # Test avec des chapitres invalides
        invalid_chapters = ["invalid", "chap99", "test"]
        
        for chapter in invalid_chapters:
            with self.subTest(chapter=chapter):
                # Tester l'accès à la configuration
                response = client.get(f"/predict/{chapter}/config")
                # Doit retourner 400 pour un chapitre invalide
                self.assertEqual(response.status_code, 400)
    
    def test_data_flow_consistency(self):
        """Test de la cohérence du flux de données"""
        # Vérifier que les endpoints retournent des structures cohérentes
        
        # Test de l'endpoint de santé
        health_response = client.get("/predict/health")
        self.assertEqual(health_response.status_code, 200)
        health_data = health_response.json()
        self.assertIn("status", health_data)
        
        # Test de l'endpoint de configuration (avec mock)
        with patch('api.routes_predict._validate_chapter') as mock_validate, \
             patch('api.routes_predict._load_yaml_config') as mock_load_config:
            
            mock_validate.return_value = None
            mock_load_config.return_value = {"test": "config"}
            
            config_response = client.get("/predict/chap30/config")
            self.assertEqual(config_response.status_code, 200)
            config_data = config_response.json()
            self.assertIsInstance(config_data, dict)

if __name__ == '__main__':
    # Configuration des tests
    unittest.main(verbosity=2, failfast=True)
