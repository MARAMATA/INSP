"""
Test d'intégration complet pour la base de données INSPECT_IA
Teste toutes les fonctionnalités et intégrations
"""

import psycopg2
import psycopg2.extras
import logging
import sys
from pathlib import Path
import json
from datetime import datetime

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from database.database_config import get_database_url
from database.models import Base, engine, SessionLocal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test de connexion à la base de données"""
    try:
        db_url = get_database_url()
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        logger.info(f"✅ Connexion PostgreSQL réussie: {version}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur connexion base de données: {e}")
        return False

def test_tables_exist():
    """Test de l'existence des tables"""
    try:
        db_url = get_database_url()
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        expected_tables = [
            'chapters', 'models', 'features', 'chapter_features', 'declarations',
            'predictions', 'rl_decisions', 'inspector_profiles', 'feedback_history',
            'analysis_results', 'model_thresholds', 'performance_metrics',
            'system_logs', 'pv_inspections', 'advanced_decisions', 'advanced_feedbacks',
            'advanced_policies'
        ]
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        
        existing_tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"📋 Tables existantes: {len(existing_tables)}")
        
        missing_tables = []
        for table in expected_tables:
            if table in existing_tables:
                logger.info(f"   ✅ {table}")
            else:
                logger.warning(f"   ❌ {table} - MANQUANTE")
                missing_tables.append(table)
        
        cursor.close()
        conn.close()
        
        if missing_tables:
            logger.error(f"❌ Tables manquantes: {missing_tables}")
            return False
        
        logger.info("✅ Toutes les tables existent")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test tables: {e}")
        return False

def test_chapters_data():
    """Test des données des chapitres"""
    try:
        db_url = get_database_url()
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT chapter_id, chapter_name, best_model, optimal_threshold, features_count
            FROM chapters 
            ORDER BY chapter_number
        """)
        
        chapters = cursor.fetchall()
        logger.info(f"📊 Chapitres trouvés: {len(chapters)}")
        
        for chapter_id, name, model, threshold, features in chapters:
            logger.info(f"   - {chapter_id}: {name}")
            logger.info(f"     Meilleur modèle: {model}")
            logger.info(f"     Seuil optimal: {threshold}")
            logger.info(f"     Features: {features}")
        
        cursor.close()
        conn.close()
        
        if len(chapters) >= 3:
            logger.info("✅ Données des chapitres correctes")
            return True
        else:
            logger.error("❌ Données des chapitres insuffisantes")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur test chapitres: {e}")
        return False

def test_features_data():
    """Test des données des features"""
    try:
        db_url = get_database_url()
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Test features par catégorie
        categories = ['basic', 'business', 'fraud_detection']
        
        for category in categories:
            cursor.execute("""
                SELECT COUNT(*) 
                FROM features 
                WHERE feature_category = %s
            """, (category,))
            
            count = cursor.fetchone()[0]
            logger.info(f"📊 Features {category}: {count}")
        
        # Test associations chapitres-features
        cursor.execute("""
            SELECT c.chapter_id, COUNT(cf.feature_id) as feature_count
            FROM chapters c
            LEFT JOIN chapter_features cf ON c.chapter_id = cf.chapter_id
            GROUP BY c.chapter_id
            ORDER BY c.chapter_number
        """)
        
        associations = cursor.fetchall()
        logger.info("🔗 Associations chapitres-features:")
        for chapter_id, count in associations:
            logger.info(f"   - {chapter_id}: {count} features")
        
        cursor.close()
        conn.close()
        
        logger.info("✅ Données des features correctes")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur test features: {e}")
        return False

def test_rl_integration():
    """Test de l'intégration RL avec PostgreSQL"""
    try:
        # Test des tables RL
        db_url = get_database_url()
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        rl_tables = ['advanced_decisions', 'advanced_feedbacks', 'inspector_profiles', 'advanced_policies']
        
        for table in rl_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"📊 Table RL {table}: {count} enregistrements")
        
        cursor.close()
        conn.close()
        
        # Test de l'import du module RL
        try:
            from src.shared.advanced_reinforcement_learning import AdvancedRLStore
            logger.info("✅ Module RL importé avec succès")
            
            # Test création d'un store RL
            rl_store = AdvancedRLStore("chap30")
            logger.info("✅ Store RL créé avec succès")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur import module RL: {e}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Erreur test intégration RL: {e}")
        return False

def test_ml_integration():
    """Test de l'intégration ML"""
    try:
        # Test des tables ML
        db_url = get_database_url()
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        ml_tables = ['models', 'predictions', 'model_thresholds', 'performance_metrics']
        
        for table in ml_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"📊 Table ML {table}: {count} enregistrements")
        
        cursor.close()
        conn.close()
        
        # Test de l'import des modules ML
        try:
            from src.chapters.chap30.ml_model_advanced import Chap30MLPipelineAdvanced
            from src.chapters.chap84.ml_model_advanced import Chap84MLPipelineAdvanced
            from src.chapters.chap85.ml_model_advanced import Chap85MLPipelineAdvanced
            logger.info("✅ Modules ML importés avec succès")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur import modules ML: {e}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Erreur test intégration ML: {e}")
        return False

def test_ocr_integration():
    """Test de l'intégration OCR"""
    try:
        # Test des tables OCR
        db_url = get_database_url()
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        ocr_tables = ['declarations', 'predictions']
        
        for table in ocr_tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            logger.info(f"📊 Table OCR {table}: {count} enregistrements")
        
        cursor.close()
        conn.close()
        
        # Test de l'import des modules OCR
        try:
            from src.shared.ocr_ingest import process_declaration_file, OCRDataContract
            from src.shared.ocr_pipeline import process_file_with_ml_prediction
            logger.info("✅ Modules OCR importés avec succès")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur import modules OCR: {e}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Erreur test intégration OCR: {e}")
        return False

def test_api_integration():
    """Test de l'intégration API"""
    try:
        # Test de l'import des modules API
        try:
            from api.main import app
            from api.routes_predict import router
            logger.info("✅ Modules API importés avec succès")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur import modules API: {e}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Erreur test intégration API: {e}")
        return False

def run_complete_integration_test():
    """Exécute tous les tests d'intégration"""
    logger.info("🚀 Démarrage des tests d'intégration INSPECT_IA")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion base de données", test_database_connection),
        ("Existence des tables", test_tables_exist),
        ("Données des chapitres", test_chapters_data),
        ("Données des features", test_features_data),
        ("Intégration RL", test_rl_integration),
        ("Intégration ML", test_ml_integration),
        ("Intégration OCR", test_ocr_integration),
        ("Intégration API", test_api_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Test: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}: RÉUSSI")
            else:
                logger.error(f"❌ {test_name}: ÉCHEC")
                
        except Exception as e:
            logger.error(f"💥 {test_name}: ERREUR - {e}")
            results.append((test_name, False))
    
    # Résumé final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Le système nécessite des corrections")
        return False

if __name__ == "__main__":
    success = run_complete_integration_test()
    if not success:
        sys.exit(1)
def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test d'intégration SQLAlchemy avec PostgreSQL
Vérifie que toutes les fonctionnalités fonctionnent correctement
"""

import sys
import logging
from datetime import datetime
from typing import Dict, Any

# Ajouter le chemin du projet
sys.path.append('/Users/macbook/Desktop/inspectia_app/backend')

from database import get_db_session, test_connection, init_database
from backend_integration import InspectIADatabase
from models import (
    Declaration, Prediction, Feature, Chapter, Model,
    RLDecision, FeedbackHistory, AnalysisResult
)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test de connexion à la base de données"""
    logger.info("🔍 Test de connexion à la base de données...")
    
    if test_connection():
        logger.info("✅ Connexion réussie")
        return True
    else:
        logger.error("❌ Échec de la connexion")
        return False

def test_database_initialization():
    """Test d'initialisation de la base de données"""
    logger.info("🏗️ Test d'initialisation de la base de données...")
    
    if init_database():
        logger.info("✅ Initialisation réussie")
        return True
    else:
        logger.error("❌ Échec de l'initialisation")
        return False

def test_basic_operations():
    """Test des opérations de base"""
    logger.info("📝 Test des opérations de base...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Test de récupération des chapitres
            chapters = db.query(Chapter).all()
            logger.info(f"✅ Chapitres trouvés: {len(chapters)}")
            
            # Test de récupération des features
            features = db.query(Feature).all()
            logger.info(f"✅ Features trouvées: {len(features)}")
            
            # Test de récupération des modèles
            models = db.query(Model).all()
            logger.info(f"✅ Modèles trouvés: {len(models)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations de base: {e}")
        return False

def test_declaration_operations():
    """Test des opérations sur les déclarations"""
    logger.info("📋 Test des opérations sur les déclarations...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Données de test avec ID unique
            import time
            unique_id = f"TEST_DECL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "poids_net_kg": 100.5,
                "nombre_colis": 10,
                "code_sh_complet": "30049000",
                "code_pays_origine": "FR",
                "created_at": datetime.utcnow()
            }
            
            # Créer une déclaration
            declaration = inspectia_db.create_declaration(test_declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            
            # Récupérer la déclaration
            retrieved_declaration = inspectia_db.get_declaration(unique_id)
            if retrieved_declaration:
                logger.info(f"✅ Déclaration récupérée: {retrieved_declaration.declaration_id}")
            else:
                logger.error("❌ Déclaration non trouvée")
                return False
            
            # Nettoyer
            db.delete(declaration)
            db.commit()
            logger.info("✅ Déclaration de test supprimée")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les déclarations: {e}")
        return False

def test_prediction_operations():
    """Test des opérations sur les prédictions"""
    logger.info("🎯 Test des opérations sur les prédictions...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer une déclaration de test avec un ID unique
            import time
            unique_id = f"TEST_PRED_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_prediction.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Créer un modèle de test
            test_model = {
                "chapter_id": "chap30",
                "model_name": "test_model",
                "model_type": "xgboost",
                "version": "1.0.0",
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            model = db.query(Model).filter(Model.chapter_id == "chap30").first()
            if not model:
                model = Model(**test_model)
                db.add(model)
                db.commit()
                db.refresh(model)
            
            # Données de prédiction
            test_prediction = {
                "declaration_id": unique_id,
                "model_id": str(model.model_id),
                "chapter_id": "chap30",
                "predicted_fraud": True,
                "fraud_probability": 0.85,
                "confidence_score": 0.85,
                "decision": "fraude",
                "decision_source": "ml",
                "ml_threshold": 0.5,
                "created_at": datetime.utcnow()
            }
            
            # Créer une prédiction
            prediction = inspectia_db.create_prediction(test_prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            
            # Récupérer les prédictions
            predictions = inspectia_db.get_predictions_by_declaration(unique_id)
            logger.info(f"✅ Prédictions récupérées: {len(predictions)}")
            
            # Nettoyer
            db.delete(prediction)
            db.delete(declaration)
            if model:
                db.delete(model)
            db.commit()
            logger.info("✅ Données de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations sur les prédictions: {e}")
        return False

def test_rl_operations():
    """Test des opérations RL"""
    logger.info("🤖 Test des opérations RL...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Créer d'abord une déclaration pour la décision RL avec ID unique
            import time
            unique_id = f"TEST_RL_{int(time.time())}"
            
            test_declaration = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "file_name": "test_rl_declaration.csv",
                "file_type": "csv",
                "source_type": "manual",
                "created_at": datetime.utcnow()
            }
            declaration = inspectia_db.create_declaration(test_declaration)
            
            # Données de décision RL
            test_decision = {
                "declaration_id": unique_id,
                "chapter_id": "chap30",
                "action": "inspect",
                "rl_probability": 0.75,
                "confidence_score": 0.75,
                "context_key": "test_context",
                "context_json": {
                    "declaration_id": unique_id,
                    "risk_factors": ["high_value", "suspicious_origin"]
                },
                "created_at": datetime.utcnow()
            }
            
            # Créer une décision RL
            decision = inspectia_db.create_rl_decision(test_decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            
            # Récupérer les décisions RL
            decisions = inspectia_db.get_rl_decisions_by_chapter("chap30", limit=10)
            logger.info(f"✅ Décisions RL récupérées: {len(decisions)}")
            
            # Nettoyer
            db.delete(decision)
            db.delete(declaration)
            db.commit()
            logger.info("✅ Données RL de test supprimées")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors des opérations RL: {e}")
        return False

def test_database_stats():
    """Test des statistiques de base de données"""
    logger.info("📊 Test des statistiques de base de données...")
    
    try:
        with get_db_session() as db:
            inspectia_db = InspectIADatabase(db)
            
            # Récupérer les statistiques
            stats = inspectia_db.get_database_stats()
            logger.info(f"✅ Statistiques récupérées:")
            logger.info(f"   - Déclarations: {stats.get('declarations', 0)}")
            logger.info(f"   - Prédictions: {stats.get('predictions', 0)}")
            logger.info(f"   - Décisions RL: {stats.get('rl_decisions', 0)}")
            logger.info(f"   - Features: {stats.get('features', 0)}")
            logger.info(f"   - Modèles: {stats.get('models', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la récupération des statistiques: {e}")
        return False

def main():
    """Fonction principale de test"""
    logger.info("🚀 Test d'intégration SQLAlchemy avec PostgreSQL")
    logger.info("=" * 60)
    
    tests = [
        ("Connexion à la base de données", test_database_connection),
        ("Initialisation de la base de données", test_database_initialization),
        ("Opérations de base", test_basic_operations),
        ("Opérations sur les déclarations", test_declaration_operations),
        ("Opérations sur les prédictions", test_prediction_operations),
        ("Opérations RL", test_rl_operations),
        ("Statistiques de base de données", test_database_stats)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name} - RÉUSSI")
            else:
                logger.error(f"❌ {test_name} - ÉCHOUÉ")
        except Exception as e:
            logger.error(f"❌ {test_name} - ERREUR: {e}")
            results.append((test_name, False))
    
    # Résumé des résultats
    logger.info("\n" + "=" * 60)
    logger.info("📋 RÉSUMÉ DES TESTS")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        logger.info(f"   {test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'intégration SQLAlchemy avec PostgreSQL fonctionne parfaitement")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
