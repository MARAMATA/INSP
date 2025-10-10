"""
Script de configuration de l'environnement pour INSPECT_IA
Configure toutes les variables d'environnement et les chemins nécessaires
"""

import os
import sys
import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_project_paths():
    """Configure les chemins du projet"""
    try:
        # Chemin racine du projet - /Users/macbook/Desktop/INSP/inspectia_app/backend
        backend_root = Path(__file__).parent.parent
        project_root = backend_root.parent.parent
        os.environ['PROJECT_ROOT'] = str(project_root)
        
        # Chemins des dossiers principaux
        paths = {
            'BACKEND_PATH': backend_root,
            'DATA_PATH': backend_root / 'data',
            'RESULTS_PATH': backend_root / 'results',
            'MODELS_PATH': backend_root / 'results',
            'LOGS_PATH': backend_root / 'logs',
            'CONFIGS_PATH': backend_root / 'configs',
            'API_PATH': backend_root / 'api',
            'SRC_PATH': backend_root / 'src',
            'DATABASE_PATH': backend_root / 'database'
        }
        
        for env_var, path in paths.items():
            os.environ[env_var] = str(path)
            logger.info(f"✅ {env_var}: {path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur configuration chemins: {e}")
        return False

def setup_database_environment():
    """Configure l'environnement de base de données"""
    try:
        # Configuration PostgreSQL INSPECT_IA
        db_config = {
            'DATABASE_URL': 'postgresql://maramata:maramata@localhost:5432/INSPECT_IA',
            'SQLALCHEMY_DATABASE_URL': 'postgresql://maramata:maramata@localhost:5432/INSPECT_IA',
            'ALEMBIC_DATABASE_URL': 'postgresql://maramata:maramata@localhost:5432/INSPECT_IA',
            'DB_HOST': 'localhost',
            'DB_PORT': '5432',
            'DB_USER': 'maramata',
            'DB_PASSWORD': 'maramata',
            'DB_NAME': 'INSPECT_IA',
            'DB_POOL_SIZE': '15',
            'DB_MAX_OVERFLOW': '25',
            'DB_ECHO': 'false',
            'DB_ECHO_POOL': 'false'
        }
        
        for key, value in db_config.items():
            os.environ[key] = value
            logger.info(f"✅ {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur configuration base de données: {e}")
        return False

def setup_ml_environment():
    """Configure l'environnement ML"""
    try:
        ml_config = {
            'ML_MODELS_PATH': os.environ.get('MODELS_PATH', 'inspectia_app/backend/results'),
            'ML_DATA_PATH': os.environ.get('DATA_PATH', 'inspectia_app/backend/data'),
            'ML_RANDOM_STATE': '42',
            'ML_TRAIN_SIZE': '0.7',
            'ML_VAL_SIZE': '0.15',
            'ML_TEST_SIZE': '0.15',
            'ML_N_JOBS': '4',
            'ML_VERBOSE': '1'
        }
        
        for key, value in ml_config.items():
            os.environ[key] = value
            logger.info(f"✅ {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur configuration ML: {e}")
        return False

def setup_rl_environment():
    """Configure l'environnement RL"""
    try:
        rl_config = {
            'RL_EPSILON_DECAY': '0.995',
            'RL_LEARNING_RATE': '0.1',
            'RL_EXPLORATION_RATE': '0.1',
            'RL_CONTEXT_WINDOW': '100',
            'RL_REWARD_SCALE': '1.0',
            'RL_PENALTY_SCALE': '0.5'
        }
        
        for key, value in rl_config.items():
            os.environ[key] = value
            logger.info(f"✅ {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur configuration RL: {e}")
        return False

def setup_ocr_environment():
    """Configure l'environnement OCR"""
    try:
        ocr_config = {
            'OCR_INPUT_PATH': os.environ.get('DATA_PATH', 'inspectia_app/backend/data') + '/ocr/input',
            'OCR_PROCESSED_PATH': os.environ.get('DATA_PATH', 'inspectia_app/backend/data') + '/ocr/processed',
            'OCR_CONFIDENCE_THRESHOLD': '0.8',
            'OCR_MAX_FILE_SIZE': '10485760',  # 10MB
            'OCR_SUPPORTED_FORMATS': 'pdf,csv,image'
        }
        
        for key, value in ocr_config.items():
            os.environ[key] = value
            logger.info(f"✅ {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur configuration OCR: {e}")
        return False

def setup_api_environment():
    """Configure l'environnement API"""
    try:
        api_config = {
            'API_HOST': '0.0.0.0',
            'API_PORT': '8000',
            'API_DEBUG': 'false',
            'API_RELOAD': 'true',
            'API_WORKERS': '1',
            'API_LOG_LEVEL': 'info',
            'API_CORS_ORIGINS': '*',
            'API_SECRET_KEY': 'inspectia-secret-key-2024',
            'API_JWT_SECRET': 'inspectia-jwt-secret-2024'
        }
        
        for key, value in api_config.items():
            os.environ[key] = value
            logger.info(f"✅ {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur configuration API: {e}")
        return False

def setup_chapters_environment():
    """Configure l'environnement des chapitres avec les NOUVEAUX modèles et métriques"""
    try:
        chapters_config = {
            'CHAPTERS': 'chap30,chap84,chap85',
            # CHAPITRE 30 - XGBoost (nouvelles métriques après réentraînement)
            'CHAP30_OPTIMAL_THRESHOLD': '0.35',
            'CHAP30_BEST_MODEL': 'xgboost',
            'CHAP30_FEATURES_COUNT': '41',
            # CHAPITRE 84 - XGBoost
            'CHAP84_OPTIMAL_THRESHOLD': '0.25',
            'CHAP84_BEST_MODEL': 'xgboost',
            'CHAP84_FEATURES_COUNT': '43',
            # CHAPITRE 85 - XGBoost
            'CHAP85_OPTIMAL_THRESHOLD': '0.20',
            'CHAP85_BEST_MODEL': 'xgboost',
            'CHAP85_FEATURES_COUNT': '43'
        }
        
        for key, value in chapters_config.items():
            os.environ[key] = value
            logger.info(f"✅ {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur configuration chapitres: {e}")
        return False

def setup_logging_environment():
    """Configure l'environnement de logging"""
    try:
        logging_config = {
            'LOG_LEVEL': 'INFO',
            'LOG_FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'LOG_FILE': os.environ.get('LOGS_PATH', 'inspectia_app/backend/logs') + '/inspectia.log',
            'LOG_MAX_SIZE': '10485760',  # 10MB
            'LOG_BACKUP_COUNT': '5'
        }
        
        for key, value in logging_config.items():
            os.environ[key] = value
            logger.info(f"✅ {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur configuration logging: {e}")
        return False

def create_directories():
    """Crée les répertoires nécessaires"""
    try:
        directories = [
            os.environ.get('DATA_PATH', 'inspectia_app/backend/data'),
            os.environ.get('RESULTS_PATH', 'inspectia_app/backend/results'),
            os.environ.get('LOGS_PATH', 'inspectia_app/backend/logs'),
            os.environ.get('DATA_PATH', 'inspectia_app/backend/data') + '/raw',
            os.environ.get('DATA_PATH', 'inspectia_app/backend/data') + '/processed',
            os.environ.get('DATA_PATH', 'inspectia_app/backend/data') + '/ml_splits',
            os.environ.get('DATA_PATH', 'inspectia_app/backend/data') + '/ocr/input',
            os.environ.get('DATA_PATH', 'inspectia_app/backend/data') + '/ocr/processed',
            os.environ.get('DATA_PATH', 'inspectia_app/backend/data') + '/rl',
            os.environ.get('RESULTS_PATH', 'inspectia_app/backend/results') + '/chap30',
            os.environ.get('RESULTS_PATH', 'inspectia_app/backend/results') + '/chap84',
            os.environ.get('RESULTS_PATH', 'inspectia_app/backend/results') + '/chap85',
            os.environ.get('RESULTS_PATH', 'inspectia_app/backend/results') + '/chap30/models',
            os.environ.get('RESULTS_PATH', 'inspectia_app/backend/results') + '/chap84/models',
            os.environ.get('RESULTS_PATH', 'inspectia_app/backend/results') + '/chap85/models'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Répertoire créé/vérifié: {directory}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur création répertoires: {e}")
        return False

def create_env_file():
    """Crée le fichier .env avec toutes les variables"""
    try:
        project_root = Path(__file__).parent.parent.parent
        env_file = project_root / '.env'
        
        env_content = f"""# Configuration INSPECT_IA - Système de détection de fraude douanière
# Généré automatiquement par setup_env.py

# Base de données PostgreSQL
DATABASE_URL=postgresql://maramata:maramata@localhost:5432/INSPECT_IA
SQLALCHEMY_DATABASE_URL=postgresql://maramata:maramata@localhost:5432/INSPECT_IA
ALEMBIC_DATABASE_URL=postgresql://maramata:maramata@localhost:5432/INSPECT_IA

# Configuration base de données
DB_HOST=localhost
DB_PORT=5432
DB_USER=maramata
DB_PASSWORD=maramata
DB_NAME=INSPECT_IA
DB_POOL_SIZE=15
DB_MAX_OVERFLOW=25
DB_ECHO=false
DB_ECHO_POOL=false

# Configuration API
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false
API_RELOAD=true
API_WORKERS=1
API_LOG_LEVEL=info
API_CORS_ORIGINS=*
API_SECRET_KEY=inspectia-secret-key-2024
API_JWT_SECRET=inspectia-jwt-secret-2024

# Configuration chapitres (NOUVEAUX MODÈLES ET MÉTRIQUES)
CHAPTERS=chap30,chap84,chap85
# Chapitre 30 - XGBoost (Validation F1: 0.9815)
CHAP30_OPTIMAL_THRESHOLD=0.35
CHAP30_BEST_MODEL=xgboost
CHAP30_FEATURES_COUNT=41
# Chapitre 84 - XGBoost (Validation F1: 0.9891)
CHAP84_OPTIMAL_THRESHOLD=0.25
CHAP84_BEST_MODEL=xgboost
CHAP84_FEATURES_COUNT=43
# Chapitre 85 - XGBoost (Validation F1: 0.9808)
CHAP85_OPTIMAL_THRESHOLD=0.20
CHAP85_BEST_MODEL=xgboost
CHAP85_FEATURES_COUNT=43

# Configuration ML
ML_MODELS_PATH=inspectia_app/backend/results
ML_DATA_PATH=inspectia_app/backend/data
ML_RANDOM_STATE=42
ML_TRAIN_SIZE=0.7
ML_VAL_SIZE=0.15
ML_TEST_SIZE=0.15
ML_N_JOBS=4
ML_VERBOSE=1

# Configuration RL
RL_EPSILON_DECAY=0.995
RL_LEARNING_RATE=0.1
RL_EXPLORATION_RATE=0.1
RL_CONTEXT_WINDOW=100
RL_REWARD_SCALE=1.0
RL_PENALTY_SCALE=0.5

# Configuration OCR
OCR_INPUT_PATH=inspectia_app/backend/data/ocr/input
OCR_PROCESSED_PATH=inspectia_app/backend/data/ocr/processed
OCR_CONFIDENCE_THRESHOLD=0.8
OCR_MAX_FILE_SIZE=10485760
OCR_SUPPORTED_FORMATS=pdf,csv,image

# Configuration logging
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOG_FILE=inspectia_app/backend/logs/inspectia.log
LOG_MAX_SIZE=10485760
LOG_BACKUP_COUNT=5

# Chemins du projet
PROJECT_ROOT={str(project_root)}
BACKEND_PATH={str(project_root / 'inspectia_app' / 'backend')}
DATA_PATH={str(project_root / 'inspectia_app' / 'backend' / 'data')}
RESULTS_PATH={str(project_root / 'inspectia_app' / 'backend' / 'results')}
MODELS_PATH={str(project_root / 'inspectia_app' / 'backend' / 'results')}
LOGS_PATH={str(project_root / 'inspectia_app' / 'backend' / 'logs')}
CONFIGS_PATH={str(project_root / 'inspectia_app' / 'backend' / 'configs')}
API_PATH={str(project_root / 'inspectia_app' / 'backend' / 'api')}
SRC_PATH={str(project_root / 'inspectia_app' / 'backend' / 'src')}
DATABASE_PATH={str(project_root / 'inspectia_app' / 'backend' / 'database')}
"""
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        logger.info(f"✅ Fichier .env créé: {env_file}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur création fichier .env: {e}")
        return False

def setup_complete_environment():
    """Configure l'environnement complet"""
    logger.info("🚀 Configuration de l'environnement INSPECT_IA")
    logger.info("=" * 60)
    
    setup_functions = [
        ("Chemins du projet", setup_project_paths),
        ("Base de données", setup_database_environment),
        ("ML", setup_ml_environment),
        ("RL", setup_rl_environment),
        ("OCR", setup_ocr_environment),
        ("API", setup_api_environment),
        ("Chapitres", setup_chapters_environment),
        ("Logging", setup_logging_environment),
        ("Répertoires", create_directories),
        ("Fichier .env", create_env_file)
    ]
    
    results = []
    
    for setup_name, setup_func in setup_functions:
        logger.info(f"\n🔧 Configuration: {setup_name}")
        logger.info("-" * 40)
        
        try:
            result = setup_func()
            results.append((setup_name, result))
            
            if result:
                logger.info(f"✅ {setup_name}: RÉUSSI")
            else:
                logger.error(f"❌ {setup_name}: ÉCHEC")
                
        except Exception as e:
            logger.error(f"💥 {setup_name}: ERREUR - {e}")
            results.append((setup_name, False))
    
    # Résumé final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RÉSUMÉ DE LA CONFIGURATION")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for setup_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{setup_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} configurations réussies")
    
    if passed == total:
        logger.info("🎉 ENVIRONNEMENT CONFIGURÉ AVEC SUCCÈS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        logger.info("\n📋 NOUVEAUX MODÈLES ET MÉTRIQUES:")
        logger.info("   - Chapitre 30: XGBoost (F1: 0.9796, AUC: 0.9995, Seuil: 0.35, Features: 41)")
        logger.info("   - Chapitre 84: XGBoost (F1: 0.9887, AUC: 0.9997, Seuil: 0.25, Features: 43)")
        logger.info("   - Chapitre 85: XGBoost (F1: 0.9808, AUC: 0.9993, Seuil: 0.20, Features: 43)")
        return True
    else:
        logger.error(f"💥 {total - passed} configuration(s) ont échoué")
        logger.error("❌ Des corrections sont nécessaires")
        return False

if __name__ == "__main__":
    success = setup_complete_environment()
    if not success:
        sys.exit(1)
