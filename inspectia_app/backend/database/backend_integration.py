"""
Script d'intégration backend pour INSPECT_IA
Intègre tous les composants du système de détection de fraude
"""

import sys
import logging
from pathlib import Path
import importlib.util

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_module_import(module_path: str, module_name: str) -> bool:
    """Teste l'import d'un module"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            logger.error(f"❌ Module {module_name}: spécification non trouvée")
            return False
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        logger.info(f"✅ Module {module_name}: importé avec succès")
        return True
        
    except Exception as e:
        logger.error(f"❌ Module {module_name}: erreur d'import - {e}")
        return False

def test_database_integration() -> bool:
    """Teste l'intégration de la base de données"""
    try:
        logger.info("🧪 Test intégration base de données")
        
        # Test des modèles SQLAlchemy
        from database.models import Base, engine, SessionLocal
        logger.info("✅ Modèles SQLAlchemy: importés")
        
        # Test de la configuration
        from database.database_config import get_database_url
        db_url = get_database_url()
        logger.info(f"✅ Configuration DB: {db_url}")
        
        # Test de connexion
        import psycopg2
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        logger.info(f"✅ Connexion DB: {version}")
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur intégration base de données: {e}")
        return False

def test_ml_integration() -> bool:
    """Teste l'intégration ML"""
    try:
        logger.info("🧪 Test intégration ML")
        
        # Test des pipelines ML
        from src.chapters.chap30.ml_model_advanced import Chap30MLPipelineAdvanced
        from src.chapters.chap84.ml_model_advanced import Chap84MLPipelineAdvanced
        from src.chapters.chap85.ml_model_advanced import Chap85MLPipelineAdvanced
        logger.info("✅ Pipelines ML: importés")
        
        # Test de création des pipelines
        for chapter, pipeline_class in [
            ("chap30", Chap30MLPipelineAdvanced),
            ("chap84", Chap84MLPipelineAdvanced),
            ("chap85", Chap85MLPipelineAdvanced)
        ]:
            pipeline = pipeline_class()
            logger.info(f"✅ Pipeline {chapter}: créé")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur intégration ML: {e}")
        return False

def test_rl_integration() -> bool:
    """Teste l'intégration RL"""
    try:
        logger.info("🧪 Test intégration RL")
        
        # Test du module RL
        from src.shared.advanced_reinforcement_learning import AdvancedRLManager, AdvancedRLStore
        logger.info("✅ Module RL: importé")
        
        # Test de création d'un manager RL
        rl_manager = AdvancedRLManager("chap30")
        logger.info("✅ Manager RL: créé")
        
        # Test de création d'un store RL
        rl_store = AdvancedRLStore("chap30")
        logger.info("✅ Store RL: créé")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur intégration RL: {e}")
        return False

def test_ocr_integration() -> bool:
    """Teste l'intégration OCR"""
    try:
        logger.info("🧪 Test intégration OCR")
        
        # Test des modules OCR
        from src.shared.ocr_ingest import process_declaration_file, OCRDataContract
        from src.shared.ocr_pipeline import process_file_with_ml_prediction
        logger.info("✅ Modules OCR: importés")
        
        # Test des configurations
        from src.shared.ocr_pipeline import CHAPTER_CONFIGS
        logger.info(f"✅ Configurations OCR: {len(CHAPTER_CONFIGS)} chapitres")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur intégration OCR: {e}")
        return False

def test_api_integration() -> bool:
    """Teste l'intégration API"""
    try:
        logger.info("🧪 Test intégration API")
        
        # Test de l'application FastAPI
        from api.main import app
        logger.info("✅ Application FastAPI: importée")
        
        # Test des routes
        from api.routes_predict import router
        logger.info("✅ Routes API: importées")
        
        # Test des endpoints
        routes = [route.path for route in app.routes if hasattr(route, 'path')]
        logger.info(f"✅ Endpoints API: {len(routes)} routes")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur intégration API: {e}")
        return False

def test_utils_integration() -> bool:
    """Teste l'intégration des utilitaires"""
    try:
        logger.info("🧪 Test intégration utilitaires")
        
        # Test de la détection de fraude avancée
        from src.utils.advanced_fraud_detection import AdvancedFraudDetection
        logger.info("✅ Détection de fraude avancée: importée")
        
        # Test de création
        fraud_detector = AdvancedFraudDetection()
        logger.info("✅ Détecteur de fraude: créé")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur intégration utilitaires: {e}")
        return False

def test_chapters_integration() -> bool:
    """Teste l'intégration des chapitres"""
    try:
        logger.info("🧪 Test intégration chapitres")
        
        # Test des préprocesseurs
        from src.chapters.chap30.preprocess import Chap30PreprocessorComprehensive
        from src.chapters.chap84.preprocess import Chap84PreprocessorComprehensive
        from src.chapters.chap85.preprocess import Chap85PreprocessorComprehensive
        logger.info("✅ Préprocesseurs: importés")
        
        # Test des intégrations RL
        from src.chapters.chap30.rl_integration import get_rl_manager
        from src.chapters.chap84.rl_integration import get_rl_manager
        from src.chapters.chap85.rl_integration import get_rl_manager
        logger.info("✅ Intégrations RL: importées")
        
        # Test des modules OCR NLP
        from src.chapters.chap30.ocr_nlp import predict_from_uploads
        from src.chapters.chap84.ocr_nlp import predict_from_uploads
        from src.chapters.chap85.ocr_nlp import predict_from_uploads
        logger.info("✅ Modules OCR NLP: importés")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur intégration chapitres: {e}")
        return False

def test_data_flow() -> bool:
    """Teste le flux de données complet"""
    try:
        logger.info("🧪 Test flux de données")
        
        # Test du flux OCR -> ML -> RL
        from src.shared.ocr_ingest import OCRDataContract
        from src.shared.ocr_pipeline import process_file_with_ml_prediction
        from src.shared.advanced_reinforcement_learning import AdvancedRLManager
        
        logger.info("✅ Flux de données: modules importés")
        
        # Test de création des composants
        rl_manager = AdvancedRLManager("chap30")
        logger.info("✅ Flux de données: composants créés")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur flux de données: {e}")
        return False

def run_integration_tests():
    """Exécute tous les tests d'intégration"""
    logger.info("🚀 TESTS D'INTÉGRATION BACKEND INSPECT_IA")
    logger.info("=" * 80)
    
    tests = [
        ("Base de données", test_database_integration),
        ("ML", test_ml_integration),
        ("RL", test_rl_integration),
        ("OCR", test_ocr_integration),
        ("API", test_api_integration),
        ("Utilitaires", test_utils_integration),
        ("Chapitres", test_chapters_integration),
        ("Flux de données", test_data_flow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Test: {test_name}")
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
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ DES TESTS D'INTÉGRATION")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} tests réussis")
    
    if passed == total:
        logger.info("🎉 TOUS LES TESTS D'INTÉGRATION SONT RÉUSSIS!")
        logger.info("✅ Le backend INSPECT_IA est correctement intégré")
        return True
    else:
        logger.error(f"💥 {total - passed} test(s) ont échoué")
        logger.error("❌ Des corrections sont nécessaires")
        return False

if __name__ == "__main__":
    success = run_integration_tests()
    if not success:
        sys.exit(1)
class InspectIADatabase:
    """
    Classe principale pour l'intégration avec la base de données PostgreSQL
    """
    
    def __init__(self, db: Session = None):
        self.db = db
    
    # =============================================================================
    # MÉTHODES POUR LES DÉCLARATIONS
    # =============================================================================
    
    def create_declaration(self, declaration_data: Dict[str, Any]) -> Declaration:
        """Créer une nouvelle déclaration"""
        try:
            declaration = Declaration(**declaration_data)
            self.db.add(declaration)
            self.db.commit()
            self.db.refresh(declaration)
            logger.info(f"✅ Déclaration créée: {declaration.declaration_id}")
            return declaration
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création déclaration: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création déclaration: {e}")
    
    def get_declaration(self, declaration_id: str) -> Optional[Declaration]:
        """Récupérer une déclaration par ID"""
        try:
            return self.db.query(Declaration).filter(
                Declaration.declaration_id == declaration_id
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération déclaration: {e}")
            return None
    
    def get_declarations_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Declaration]:
        """Récupérer les déclarations par chapitre"""
        try:
            return self.db.query(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).order_by(Declaration.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération déclarations: {e}")
            return []
    
    def get_recent_declarations(self, limit: int = 100) -> List[Declaration]:
        """Récupérer les déclarations récentes de tous les chapitres"""
        try:
            return self.db.query(Declaration).order_by(
                Declaration.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération déclarations récentes: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES PRÉDICTIONS
    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass
    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

    # =============================================================================
    
    def create_prediction(self, prediction_data: Dict[str, Any]) -> Prediction:
        """Créer une nouvelle prédiction"""
        try:
            prediction = Prediction(**prediction_data)
            self.db.add(prediction)
            self.db.commit()
            self.db.refresh(prediction)
            logger.info(f"✅ Prédiction créée: {prediction.prediction_id}")
            return prediction
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création prédiction: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création prédiction: {e}")
    
    def get_predictions_by_declaration(self, declaration_id: str) -> List[Prediction]:
        """Récupérer les prédictions pour une déclaration"""
        try:
            return self.db.query(Prediction).filter(
                Prediction.declaration_id == declaration_id
            ).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    def get_predictions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[Prediction]:
        """Récupérer les prédictions par chapitre"""
        try:
            return self.db.query(Prediction).join(Declaration).filter(
                Declaration.chapter_id == chapter_id
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération prédictions: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES FEATURES
    # =============================================================================
    
    def get_features_by_chapter(self, chapter_id: str) -> List[Feature]:
        """Récupérer les features pour un chapitre"""
        try:
            return self.db.query(Feature).join(ChapterFeature).filter(
                ChapterFeature.chapter_id == chapter_id
            ).order_by(ChapterFeature.feature_order).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération features: {e}")
            return []
    
    def get_feature_by_name(self, feature_name: str) -> Optional[Feature]:
        """Récupérer une feature par nom"""
        try:
            return self.db.query(Feature).filter(
                Feature.feature_name == feature_name
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feature: {e}")
            return None
    
    # =============================================================================
    # MÉTHODES POUR LES MODÈLES
    # =============================================================================
    
    def get_model_by_chapter(self, chapter_id: str) -> Optional[Model]:
        """Récupérer le modèle actuel pour un chapitre"""
        try:
            return self.db.query(Model).filter(
                Model.chapter_id == chapter_id,
                Model.is_active == True
            ).first()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèle: {e}")
            return None
    
    def get_all_models(self) -> List[Model]:
        """Récupérer tous les modèles"""
        try:
            return self.db.query(Model).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LE SYSTÈME RL
    # =============================================================================
    
    def create_rl_decision(self, decision_data: Dict[str, Any]) -> RLDecision:
        """Créer une décision RL"""
        try:
            decision = RLDecision(**decision_data)
            self.db.add(decision)
            self.db.commit()
            self.db.refresh(decision)
            logger.info(f"✅ Décision RL créée: {decision.decision_id}")
            return decision
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création décision RL: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création décision RL: {e}")
    
    def get_rl_decisions_by_chapter(self, chapter_id: str, limit: int = 100) -> List[RLDecision]:
        """Récupérer les décisions RL par chapitre"""
        try:
            return self.db.query(RLDecision).filter(
                RLDecision.chapter_id == chapter_id
            ).order_by(RLDecision.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération décisions RL: {e}")
            return []
    
    def create_feedback(self, feedback_data: Dict[str, Any]) -> FeedbackHistory:
        """Créer un feedback"""
        try:
            feedback = FeedbackHistory(**feedback_data)
            self.db.add(feedback)
            self.db.commit()
            self.db.refresh(feedback)
            logger.info(f"✅ Feedback créé: {feedback.feedback_id}")
            return feedback
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création feedback: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création feedback: {e}")
    
    def get_feedback_history(self, limit: int = 100) -> List[FeedbackHistory]:
        """Récupérer l'historique des feedbacks"""
        try:
            return self.db.query(FeedbackHistory).order_by(
                FeedbackHistory.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération feedbacks: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES ANALYSES
    # =============================================================================
    
    def create_analysis_result(self, analysis_data: Dict[str, Any]) -> AnalysisResult:
        """Créer un résultat d'analyse"""
        try:
            analysis = AnalysisResult(**analysis_data)
            self.db.add(analysis)
            self.db.commit()
            self.db.refresh(analysis)
            logger.info(f"✅ Analyse créée: {analysis.analysis_id}")
            return analysis
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création analyse: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création analyse: {e}")
    
    def get_analysis_results(self, limit: int = 100) -> List[AnalysisResult]:
        """Récupérer les résultats d'analyse"""
        try:
            return self.db.query(AnalysisResult).order_by(
                AnalysisResult.created_at.desc()
            ).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération analyses: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES POUR LES MÉTRIQUES
    # =============================================================================
    
    def create_performance_metric(self, metric_data: Dict[str, Any]) -> PerformanceMetric:
        """Créer une métrique de performance"""
        try:
            metric = PerformanceMetric(**metric_data)
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            logger.info(f"✅ Métrique créée: {metric.metric_id}")
            return metric
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur création métrique: {e}")
            raise HTTPException(status_code=500, detail=f"Erreur création métrique: {e}")
    
    def get_performance_metrics(self, chapter_id: str = None, limit: int = 100) -> List[PerformanceMetric]:
        """Récupérer les métriques de performance"""
        try:
            query = self.db.query(PerformanceMetric)
            if chapter_id:
                query = query.filter(PerformanceMetric.chapter_id == chapter_id)
            return query.order_by(PerformanceMetric.created_at.desc()).limit(limit).all()
        except Exception as e:
            logger.error(f"❌ Erreur récupération métriques: {e}")
            return []
    
    # =============================================================================
    # MÉTHODES UTILITAIRES
    # =============================================================================
    
    def log_system_event(self, event_type: str, message: str, details: Dict[str, Any] = None):
        """Logger un événement système"""
        try:
            log_entry = SystemLog(
                event_type=event_type,
                message=message,
                details=details or {},
                created_at=datetime.utcnow()
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.info(f"📝 Événement système loggé: {event_type}")
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Erreur logging: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Récupérer les statistiques de la base de données"""
        try:
            stats = {}
            
            # Compter les enregistrements par table
            stats['declarations'] = self.db.query(Declaration).count()
            stats['predictions'] = self.db.query(Prediction).count()
            stats['rl_decisions'] = self.db.query(RLDecision).count()
            stats['feedbacks'] = self.db.query(FeedbackHistory).count()
            stats['analyses'] = self.db.query(AnalysisResult).count()
            stats['features'] = self.db.query(Feature).count()
            stats['models'] = self.db.query(Model).count()
            
            # Statistiques par chapitre
            stats['by_chapter'] = {}
            for chapter in self.db.query(Chapter).all():
                stats['by_chapter'][chapter.chapter_id] = {
                    'declarations': self.db.query(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count(),
                    'predictions': self.db.query(Prediction).join(Declaration).filter(
                        Declaration.chapter_id == chapter.chapter_id
                    ).count()
                }
            
            return stats
        except Exception as e:
            logger.error(f"❌ Erreur récupération stats: {e}")
            return {}

# Instance globale pour l'utilisation dans FastAPI
def get_inspectia_db(db: Session = Depends(get_db)) -> InspectIADatabase:
    """Dependency pour FastAPI"""
    return InspectIADatabase(db)

# Fonctions utilitaires pour l'ancien système
def migrate_from_sqlite():
    """
    Fonction pour migrer les données de l'ancien système SQLite
    (À implémenter si nécessaire)
    """
    logger.info("🔄 Migration depuis SQLite vers PostgreSQL")
    # TODO: Implémenter la migration si nécessaire
    pass

def backup_database():
    """
    Fonction pour créer une sauvegarde de la base de données
    """
    logger.info("💾 Sauvegarde de la base de données")
    # TODO: Implémenter la sauvegarde
    pass

