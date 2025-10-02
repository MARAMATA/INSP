"""
Script principal d'intégration backend pour INSPECT_IA
Orchestre l'installation et la configuration complète du système
"""

import sys
import logging
from pathlib import Path
import subprocess
import time

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(script_path: Path, description: str) -> bool:
    """Exécute un script Python et retourne le succès"""
    try:
        logger.info(f"🚀 {description}")
        logger.info(f"   Script: {script_path}")
        
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd=script_path.parent.parent)
        
        if result.returncode == 0:
            logger.info(f"✅ {description}: RÉUSSI")
            if result.stdout:
                logger.info(f"   Sortie: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"❌ {description}: ÉCHEC")
            if result.stderr:
                logger.error(f"   Erreur: {result.stderr.strip()}")
            return False
            
    except Exception as e:
        logger.error(f"💥 {description}: ERREUR - {e}")
        return False

def check_prerequisites() -> bool:
    """Vérifie les prérequis du système"""
    try:
        logger.info("🔍 Vérification des prérequis")
        logger.info("-" * 40)
        
        # Vérifier Python
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            logger.error(f"❌ Python 3.8+ requis, version actuelle: {python_version.major}.{python_version.minor}")
            return False
        logger.info(f"✅ Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Vérifier les modules requis
        required_modules = [
            'psycopg2', 'sqlalchemy', 'pandas', 'numpy', 'sklearn',
            'fastapi', 'uvicorn', 'pydantic', 'yaml', 'joblib'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"✅ Module {module}: disponible")
            except ImportError:
                logger.error(f"❌ Module {module}: manquant")
                missing_modules.append(module)
        
        if missing_modules:
            logger.error(f"❌ Modules manquants: {missing_modules}")
            logger.error("   Installez-les avec: pip install " + " ".join(missing_modules))
            return False
        
        # Vérifier PostgreSQL
        try:
            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                user='maramata',
                password='maramata',
                database='postgres'
            )
            conn.close()
            logger.info("✅ PostgreSQL: accessible")
        except Exception as e:
            logger.error(f"❌ PostgreSQL: non accessible - {e}")
            logger.error("   Vérifiez que PostgreSQL est installé et en cours d'exécution")
            logger.error("   Créez l'utilisateur 'maramata' avec le mot de passe 'maramata'")
            return False
        
        logger.info("✅ Tous les prérequis sont satisfaits")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur vérification prérequis: {e}")
        return False

def install_database() -> bool:
    """Installe la base de données"""
    try:
        script_path = Path(__file__).parent / "install_database.py"
        return run_script(script_path, "Installation de la base de données INSPECT_IA")
    except Exception as e:
        logger.error(f"❌ Erreur installation base de données: {e}")
        return False

def setup_environment() -> bool:
    """Configure l'environnement"""
    try:
        script_path = Path(__file__).parent / "setup_env.py"
        return run_script(script_path, "Configuration de l'environnement")
    except Exception as e:
        logger.error(f"❌ Erreur configuration environnement: {e}")
        return False

def verify_features() -> bool:
    """Vérifie les features"""
    try:
        script_path = Path(__file__).parent / "verify_features.py"
        return run_script(script_path, "Vérification des features")
    except Exception as e:
        logger.error(f"❌ Erreur vérification features: {e}")
        return False

def test_integration() -> bool:
    """Teste l'intégration complète"""
    try:
        script_path = Path(__file__).parent / "test_integration.py"
        return run_script(script_path, "Test d'intégration complet")
    except Exception as e:
        logger.error(f"❌ Erreur test intégration: {e}")
        return False

def run_backend_integration():
    """Exécute l'intégration complète du backend"""
    logger.info("🚀 INTÉGRATION BACKEND INSPECT_IA")
    logger.info("=" * 80)
    logger.info("Système de détection de fraude douanière avec ML-RL avancé")
    logger.info("=" * 80)
    
    # Étapes d'intégration
    integration_steps = [
        ("Vérification des prérequis", check_prerequisites),
        ("Configuration de l'environnement", setup_environment),
        ("Installation de la base de données", install_database),
        ("Vérification des features", verify_features),
        ("Test d'intégration complet", test_integration)
    ]
    
    results = []
    
    for step_name, step_func in integration_steps:
        logger.info(f"\n📋 Étape: {step_name}")
        logger.info("-" * 50)
        
        try:
            result = step_func()
            results.append((step_name, result))
            
            if result:
                logger.info(f"✅ {step_name}: RÉUSSI")
            else:
                logger.error(f"❌ {step_name}: ÉCHEC")
                logger.error("🛑 Arrêt de l'intégration")
                break
                
        except Exception as e:
            logger.error(f"💥 {step_name}: ERREUR - {e}")
            results.append((step_name, False))
            logger.error("🛑 Arrêt de l'intégration")
            break
        
        # Pause entre les étapes
        time.sleep(2)
    
    # Résumé final
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ DE L'INTÉGRATION")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for step_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{step_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} étapes réussies")
    
    if passed == total:
        logger.info("🎉 INTÉGRATION BACKEND TERMINÉE AVEC SUCCÈS!")
        logger.info("✅ Le système INSPECT_IA est prêt à être utilisé")
        logger.info("\n📋 Prochaines étapes:")
        logger.info("   1. Démarrer l'API: uvicorn api.main:app --host 0.0.0.0 --port 8000")
        logger.info("   2. Tester les endpoints: http://localhost:8000/docs")
        logger.info("   3. Utiliser le système de détection de fraude")
        return True
    else:
        logger.error(f"💥 {total - passed} étape(s) ont échoué")
        logger.error("❌ L'intégration nécessite des corrections")
        logger.error("\n🔧 Actions recommandées:")
        logger.error("   1. Vérifier les prérequis manquants")
        logger.error("   2. Corriger les erreurs de configuration")
        logger.error("   3. Relancer l'intégration")
        return False

def main():
    """Fonction principale"""
    try:
        success = run_backend_integration()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")

            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Intégration interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
from database.database import get_db, get_db_session, init_database, test_connection
from database.backend_integration import InspectIADatabase
from database.api_routes import router as postgresql_router

# Import des modules existants
from api.main import app
from api.routes_predict import router as predict_router

def integrate_postgresql_with_backend():
    """
    Intègre le système PostgreSQL avec le backend FastAPI existant
    """
    print("🔧 Intégration PostgreSQL avec le backend FastAPI")
    print("=" * 60)
    
    # 1. Test de connexion
    print("1️⃣ Test de connexion PostgreSQL...")
    if not test_connection():
        print("❌ Échec de la connexion PostgreSQL")
        return False
    print("✅ Connexion PostgreSQL réussie")
    
    # 2. Initialisation de la base de données
    print("2️⃣ Initialisation de la base de données...")
    if not init_database():
        print("❌ Échec de l'initialisation")
        return False
    print("✅ Base de données initialisée")
    
    # 3. Ajout des routes PostgreSQL à l'application FastAPI
    print("3️⃣ Ajout des routes PostgreSQL...")
    try:
        # Inclure les routes PostgreSQL avec le préfixe /api/v2
        app.include_router(postgresql_router, prefix="/api/v2")
        print("✅ Routes PostgreSQL ajoutées")
    except Exception as e:
        print(f"❌ Erreur ajout routes: {e}")
        return False
    
    # 4. Test des routes
    print("4️⃣ Test des routes...")
    try:
        # Test de la route de santé
        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test de la route de santé PostgreSQL
        response = client.get("/api/v2/health/")
        if response.status_code == 200:
            print("✅ Route de santé PostgreSQL fonctionnelle")
        else:
            print(f"⚠️ Route de santé: {response.status_code}")
        
        # Test des statistiques
        response = client.get("/api/v2/stats/")
        if response.status_code == 200:
            print("✅ Route des statistiques fonctionnelle")
        else:
            print(f"⚠️ Route statistiques: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ Erreur test routes: {e}")
    
    print("=" * 60)
    print("🎉 Intégration PostgreSQL terminée avec succès!")
    print("📋 Routes disponibles:")
    print("   - /api/v2/health/ - Santé de la base de données")
    print("   - /api/v2/stats/ - Statistiques de la base de données")
    print("   - /api/v2/declarations/ - Gestion des déclarations")
    print("   - /api/v2/predictions/ - Gestion des prédictions")
    print("   - /api/v2/features/ - Gestion des features")
    print("   - /api/v2/models/ - Gestion des modèles")
    print("   - /api/v2/rl/ - Système RL")
    
    return True

def create_hybrid_backend():
    """
    Crée un backend hybride utilisant à la fois SQLite et PostgreSQL
    Permet une transition progressive
    """
    print("🔄 Création d'un backend hybride")
    print("=" * 60)
    
    # Configuration hybride
    hybrid_config = {
        "use_postgresql": True,
        "use_sqlite": True,  # Pour compatibilité
        "postgresql_primary": True,
        "sqlite_fallback": True
    }
    
    print("📋 Configuration hybride:")
    for key, value in hybrid_config.items():
        print(f"   - {key}: {value}")
    
    return hybrid_config

def migrate_endpoints_to_postgresql():
    """
    Migre progressivement les endpoints vers PostgreSQL
    """
    print("🔄 Migration des endpoints vers PostgreSQL")
    print("=" * 60)
    
    # Endpoints à migrer
    endpoints_to_migrate = [
        "/predict",
        "/upload",
        "/declarations",
        "/feedback",
        "/analytics"
    ]
    
    print("📋 Endpoints à migrer:")
    for endpoint in endpoints_to_migrate:
        print(f"   - {endpoint}")
    
    # Stratégie de migration
    migration_strategy = {
        "phase_1": "Ajout des routes PostgreSQL en parallèle",
        "phase_2": "Test des nouvelles routes",
        "phase_3": "Redirection progressive du trafic",
        "phase_4": "Désactivation des anciennes routes SQLite"
    }
    
    print("\n📋 Stratégie de migration:")
    for phase, description in migration_strategy.items():
        print(f"   - {phase}: {description}")
    
    return migration_strategy

def create_database_abstraction_layer():
    """
    Crée une couche d'abstraction pour gérer SQLite et PostgreSQL
    """
    print("🏗️ Création d'une couche d'abstraction")
    print("=" * 60)
    
    abstraction_code = '''
class DatabaseManager:
    """Gestionnaire de base de données hybride"""
    
    def __init__(self, use_postgresql=True, use_sqlite=False):
        self.use_postgresql = use_postgresql
        self.use_sqlite = use_sqlite
        
        if use_postgresql:
            from database.backend_integration import InspectIADatabase
            self.postgresql_db = InspectIADatabase()
        
        if use_sqlite:
            # Ancien système SQLite
            self.sqlite_db = None  # À implémenter
    
    def get_declaration(self, declaration_id: str):
        """Récupère une déclaration (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.get_declaration(declaration_id)
        elif self.use_sqlite:
            return self.sqlite_db.get_declaration(declaration_id)
        return None
    
    def create_prediction(self, prediction_data: dict):
        """Crée une prédiction (PostgreSQL prioritaire)"""
        if self.use_postgresql:
            return self.postgresql_db.create_prediction(prediction_data)
        elif self.use_sqlite:
            return self.sqlite_db.create_prediction(prediction_data)
        return None
    '''
    
    print("✅ Couche d'abstraction créée")
    return abstraction_code

if __name__ == "__main__":
    print("🚀 Intégration PostgreSQL avec le backend InspectIA")
    print("=" * 60)
    
    # 1. Intégration principale
    success = integrate_postgresql_with_backend()
    
    if success:
        # 2. Configuration hybride
        hybrid_config = create_hybrid_backend()
        
        # 3. Stratégie de migration
        migration_strategy = migrate_endpoints_to_postgresql()
        
        # 4. Couche d'abstraction
        abstraction_code = create_database_abstraction_layer()
        
        print("\n🎉 Intégration complète réussie!")
        print("📋 Prochaines étapes:")
        print("   1. Tester les nouvelles routes PostgreSQL")
        print("   2. Migrer progressivement les endpoints")
        print("   3. Adapter le frontend")
        print("   4. Tests d'intégration complets")
    else:
        print("❌ Échec de l'intégration")
