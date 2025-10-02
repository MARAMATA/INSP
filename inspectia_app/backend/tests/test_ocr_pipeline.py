#!/usr/bin/env python3
"""
Tests unitaires pour l'OCR Pipeline
Tests avec mocks pour isoler les composants
"""
import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Ajouter le chemin du projet
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.shared.ocr_pipeline import (
    predict_from_declarations,
    predict_from_csv,
    _get_hybrid_config,
    _get_anomaly_thresholds,
    _get_pays_risque,
    _get_bureaux_risque
)

class TestOCRPipeline(unittest.TestCase):
    """Tests unitaires pour l'OCR Pipeline"""
    
    def setUp(self):
        """Configuration initiale pour chaque test"""
        # Données de test
        self.test_declarations = [
            {
                "numero_declaration": "TEST001",
                "date_declaration": "2024-01-15",
                "valeur_fob": 1000.0,
                "pays_origine": "FR",
                "bureau_douane": "TEST",
                "chapitre": "30"
            },
            {
                "numero_declaration": "TEST002",
                "date_declaration": "2024-01-16",
                "valeur_fob": 5000.0,
                "pays_origine": "DE",
                "bureau_douane": "TEST2",
                "chapitre": "30"
            }
        ]
        
        # Configuration de test
        self.test_config = {
            "hybrid": {
                "method_default": "HYBRID_SCORE_BASED",
                "seuils": {
                    "ml_conforme": 0.30,
                    "ml_fraude": 0.65
                }
            },
            "anomaly_thresholds": {
                "valeur_fob_min": 100,
                "valeur_fob_max": 10000
            },
            "pays_risque": ["XX", "YY"],
            "bureaux_risque": ["RISK1", "RISK2"]
        }
    
    @patch('src.shared.ocr_pipeline._load_chapter_config')
    def test_get_hybrid_config(self, mock_load_chapter):
        """Test de la récupération de la configuration hybride"""
        # Configurer le mock
        mock_load_yaml.return_value = self.test_config
        
        # Tester la récupération
        result = _get_hybrid_config("chap30")
        
        # Vérifications
        self.assertEqual(result["method_default"], "HYBRID_SCORE_BASED")
        self.assertEqual(result["seuils"]["ml_conforme"], 0.30)
        self.assertEqual(result["seuils"]["ml_fraude"], 0.65)
        
        # Vérifier que le mock a été appelé
        mock_load_yaml.assert_called_once_with("chap30")
    
    @patch('src.shared.ocr_pipeline._load_yaml_config')
    def test_get_anomaly_thresholds(self, mock_load_yaml):
        """Test de la récupération des seuils d'anomalie"""
        # Configurer le mock
        mock_load_yaml.return_value = self.test_config
        
        # Tester la récupération
        result = _get_anomaly_thresholds("chap30")
        
        # Vérifications
        self.assertEqual(result["valeur_fob_min"], 100)
        self.assertEqual(result["valeur_fob_max"], 10000)
        
        # Vérifier que le mock a été appelé
        mock_load_yaml.assert_called_once_with("chap30")
    
    @patch('src.shared.ocr_pipeline._load_yaml_config')
    def test_get_pays_risque(self, mock_load_yaml):
        """Test de la récupération des pays à risque"""
        # Configurer le mock
        mock_load_yaml.return_value = self.test_config
        
        # Tester la récupération
        result = _get_pays_risque("chap30")
        
        # Vérifications
        self.assertIn("XX", result)
        self.assertIn("YY", result)
        self.assertEqual(len(result), 2)
        
        # Vérifier que le mock a été appelé
        mock_load_yaml.assert_called_once_with("chap30")
    
    @patch('src.shared.ocr_pipeline._load_yaml_config')
    def test_get_bureaux_risque(self, mock_load_yaml):
        """Test de la récupération des bureaux à risque"""
        # Configurer le mock
        mock_load_yaml.return_value = self.test_config
        
        # Tester la récupération
        result = _get_bureaux_risque("chap30")
        
        # Vérifications
        self.assertIn("RISK1", result)
        self.assertIn("RISK2", result)
        self.assertEqual(len(result), 2)
        
        # Vérifier que le mock a été appelé
        mock_load_yaml.assert_called_once_with("chap30")
    
    @patch('src.shared.ocr_pipeline._load_yaml_config')
    @patch('src.shared.ocr_pipeline._load_ml_models')
    def test_predict_from_declarations_success(self, mock_load_models, mock_load_config):
        """Test de prédiction réussie depuis des déclarations"""
        # Configurer les mocks
        mock_load_config.return_value = self.test_config
        
        # Mock du modèle ML
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])  # 80% fraude
        mock_load_models.return_value = {
            "best_model": mock_model,
            "preprocessor": Mock(),
            "feature_selector": Mock(),
            "feature_weighter": Mock()
        }
        
        # Tester la prédiction
        result = predict_from_declarations("chap30", self.test_declarations)
        
        # Vérifications de base
        self.assertIsNotNone(result)
        self.assertIn("predictions", result)
        self.assertIn("summary", result)
        
        # Vérifier que les mocks ont été appelés
        mock_load_config.assert_called_once_with("chap30")
        mock_load_models.assert_called_once_with("chap30")
    
    @patch('src.shared.ocr_pipeline._load_yaml_config')
    def test_predict_from_declarations_invalid_chapter(self, mock_load_config):
        """Test de prédiction avec un chapitre invalide"""
        # Configurer le mock pour lever une exception
        mock_load_config.side_effect = FileNotFoundError("Chapitre non trouvé")
        
        # Tester que l'exception est levée
        with self.assertRaises(FileNotFoundError):
            predict_from_declarations("invalid_chapter", self.test_declarations)
    
    @patch('src.shared.ocr_pipeline._load_yaml_config')
    def test_predict_from_declarations_empty_data(self, mock_load_config):
        """Test de prédiction avec des données vides"""
        # Configurer le mock
        mock_load_config.return_value = self.test_config
        
        # Tester avec des données vides
        result = predict_from_declarations("chap30", [])
        
        # Vérifications
        self.assertIsNotNone(result)
        self.assertEqual(len(result["predictions"]), 0)
        self.assertEqual(result["summary"]["total_declarations"], 0)
    
    @patch('src.shared.ocr_pipeline._load_yaml_config')
    @patch('src.shared.ocr_pipeline._load_ml_models')
    def test_predict_from_declarations_model_error(self, mock_load_models, mock_load_config):
        """Test de prédiction avec erreur du modèle ML"""
        # Configurer les mocks
        mock_load_config.return_value = self.test_config
        
        # Mock du modèle ML qui lève une exception
        mock_model = Mock()
        mock_model.predict_proba.side_effect = Exception("Erreur modèle")
        mock_load_models.return_value = {
            "best_model": mock_model,
            "preprocessor": Mock(),
            "feature_selector": Mock(),
            "feature_weighter": Mock()
        }
        
        # Tester que l'exception est propagée
        with self.assertRaises(Exception):
            predict_from_declarations("chap30", self.test_declarations)
    
    @patch('src.shared.ocr_pipeline._load_yaml_config')
    @patch('src.shared.ocr_pipeline._load_ml_models')
    def test_predict_from_csv_success(self, mock_load_models, mock_load_config):
        """Test de prédiction réussie depuis un CSV"""
        # Configurer les mocks
        mock_load_config.return_value = self.test_config
        
        # Mock du modèle ML
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        mock_load_models.return_value = {
            "best_model": mock_model,
            "preprocessor": Mock(),
            "feature_selector": Mock(),
            "feature_weighter": Mock()
        }
        
        # Créer un DataFrame de test
        test_df = pd.DataFrame(self.test_declarations)
        
        # Tester la prédiction
        result = predict_from_csv("chap30", test_df)
        
        # Vérifications
        self.assertIsNotNone(result)
        self.assertIn("predictions", result)
        self.assertIn("summary", result)
        self.assertEqual(len(result["predictions"]), 2)
    
    def test_data_validation(self):
        """Test de la validation des données d'entrée"""
        # Test avec des données valides
        valid_data = [
            {
                "numero_declaration": "TEST001",
                "valeur_fob": 1000.0,
                "pays_origine": "FR"
            }
        ]
        
        # Les données doivent être valides
        self.assertIsInstance(valid_data, list)
        self.assertGreater(len(valid_data), 0)
        self.assertIn("numero_declaration", valid_data[0])
        self.assertIn("valeur_fob", valid_data[0])
        
        # Test avec des données invalides
        invalid_data = [
            {
                "numero_declaration": "",  # Numéro vide
                "valeur_fob": -100,        # Valeur négative
                "pays_origine": None       # Pays manquant
            }
        ]
        
        # Les données invalides doivent être détectées
        self.assertEqual(invalid_data[0]["numero_declaration"], "")
        self.assertLess(invalid_data[0]["valeur_fob"], 0)
        self.assertIsNone(invalid_data[0]["pays_origine"])
    
    @patch('src.shared.ocr_pipeline._load_yaml_config')
    def test_config_loading_error_handling(self, mock_load_config):
        """Test de la gestion des erreurs de chargement de configuration"""
        # Configurer le mock pour lever différentes exceptions
        test_cases = [
            (FileNotFoundError("Fichier non trouvé"), "Chapitre inexistant"),
            (PermissionError("Accès refusé"), "Problème de permissions"),
            (ValueError("YAML invalide"), "Configuration corrompue")
        ]
        
        for exception, description in test_cases:
            with self.subTest(description=description):
                mock_load_config.side_effect = exception
                
                # Tester que l'exception est propagée
                with self.assertRaises(type(exception)):
                    _get_hybrid_config("chap30")
    
    def test_data_structure_consistency(self):
        """Test de la cohérence des structures de données"""
        # Vérifier que toutes les déclarations ont la même structure
        required_fields = ["numero_declaration", "valeur_fob", "pays_origine"]
        
        for declaration in self.test_declarations:
            for field in required_fields:
                self.assertIn(field, declaration, f"Champ manquant: {field}")
        
        # Vérifier les types de données
        for declaration in self.test_declarations:
            self.assertIsInstance(declaration["numero_declaration"], str)
            self.assertIsInstance(declaration["valeur_fob"], (int, float))
            self.assertIsInstance(declaration["pays_origine"], str)
    
    @patch('src.shared.ocr_pipeline._load_yaml_config')
    @patch('src.shared.ocr_pipeline._load_ml_models')
    def test_prediction_result_structure(self, mock_load_models, mock_load_config):
        """Test de la structure des résultats de prédiction"""
        # Configurer les mocks
        mock_load_config.return_value = self.test_config
        
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        mock_load_models.return_value = {
            "best_model": mock_model,
            "preprocessor": Mock(),
            "feature_selector": Mock(),
            "feature_weighter": Mock()
        }
        
        # Effectuer la prédiction
        result = predict_from_declarations("chap30", self.test_declarations[:1])
        
        # Vérifier la structure du résultat
        self.assertIn("predictions", result)
        self.assertIn("summary", result)
        
        # Vérifier la structure des prédictions
        predictions = result["predictions"]
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0)
        
        # Vérifier la structure du résumé
        summary = result["summary"]
        self.assertIn("total_declarations", summary)
        self.assertIn("fraud_count", summary)
        self.assertIsInstance(summary["total_declarations"], int)
        self.assertIsInstance(summary["fraud_count"], int)

class TestOCRPipelineIntegration(unittest.TestCase):
    """Tests d'intégration pour l'OCR Pipeline"""
    
    def setUp(self):
        """Configuration pour les tests d'intégration"""
        self.test_data = pd.DataFrame({
            "numero_declaration": ["TEST001", "TEST002", "TEST003"],
            "valeur_fob": [1000, 5000, 100],
            "pays_origine": ["FR", "DE", "XX"],
            "bureau_douane": ["TEST1", "TEST2", "RISK1"]
        })
    
    @patch('src.shared.ocr_pipeline._load_yaml_config')
    @patch('src.shared.ocr_pipeline._load_ml_models')
    def test_end_to_end_prediction_flow(self, mock_load_models, mock_load_config):
        """Test du flux de prédiction complet"""
        # Configuration complète des mocks
        mock_load_config.return_value = {
            "hybrid": {
                "method_default": "HYBRID_SCORE_BASED",
                "seuils": {"ml_conforme": 0.30, "ml_fraude": 0.65}
            },
            "anomaly_thresholds": {"valeur_fob_min": 100, "valeur_fob_max": 10000},
            "pays_risque": ["XX"],
            "bureaux_risque": ["RISK1"]
        }
        
        # Mock du modèle ML avec prédictions variées
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([
            [0.8, 0.2],  # 20% fraude
            [0.3, 0.7],  # 70% fraude
            [0.9, 0.1]   # 10% fraude
        ])
        
        mock_load_models.return_value = {
            "best_model": mock_model,
            "preprocessor": Mock(),
            "feature_selector": Mock(),
            "feature_weighter": Mock()
        }
        
        # Test du flux complet
        result = predict_from_csv("chap30", self.test_data)
        
        # Vérifications
        self.assertIsNotNone(result)
        self.assertEqual(len(result["predictions"]), 3)
        self.assertEqual(result["summary"]["total_declarations"], 3)
        
        # Vérifier que les prédictions sont cohérentes
        predictions = result["predictions"]
        for i, pred in enumerate(predictions):
            self.assertIn("fraud_probability", pred)
            self.assertIn("decision", pred)
            self.assertIsInstance(pred["fraud_probability"], (int, float))
    
    def test_data_quality_validation(self):
        """Test de la validation de la qualité des données"""
        # Test avec des données de bonne qualité
        good_quality_data = pd.DataFrame({
            "numero_declaration": ["TEST001", "TEST002"],
            "valeur_fob": [1000, 2000],
            "pays_origine": ["FR", "DE"],
            "bureau_douane": ["TEST1", "TEST2"]
        })
        
        # Vérifier la qualité des données
        self.assertFalse(good_quality_data.isnull().any().any())
        self.assertTrue((good_quality_data["valeur_fob"] > 0).all())
        self.assertTrue(good_quality_data["numero_declaration"].str.len() > 0).all()
        
        # Test avec des données de mauvaise qualité
        bad_quality_data = pd.DataFrame({
            "numero_declaration": ["", "TEST002"],
            "valeur_fob": [-100, 2000],
            "pays_origine": [None, "DE"],
            "bureau_douane": ["TEST1", ""]
        })
        
        # Vérifier que les problèmes sont détectés
        self.assertTrue(bad_quality_data.isnull().any().any())
        self.assertTrue((bad_quality_data["valeur_fob"] < 0).any())
        self.assertTrue((bad_quality_data["numero_declaration"] == "").any())

if __name__ == '__main__':
    # Configuration des tests
    unittest.main(verbosity=2, failfast=True)
