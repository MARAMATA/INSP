"""
Configuration d'intégration pour INSPECT_IA
Centralise toutes les configurations de base de données et d'intégration
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntegrationConfig:
    """Configuration centralisée pour l'intégration INSPECT_IA"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent / "configs" / "integration.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier YAML"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Fichier de configuration non trouvé: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Erreur chargement configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuration par défaut"""
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'user': 'maramata',
                'password': 'maramata',
                'database': 'INSPECT_IA',
                'pool_size': 15,
                'max_overflow': 25,
                'echo': False,
                'echo_pool': False
            },
            'chapters': {
                'chap30': {
                    'name': 'Produits pharmaceutiques',
                    'best_model': 'catboost',
                    'optimal_threshold': 0.20,
                    'features_count': 43,
                    'business_features_count': 15
                },
                'chap84': {
                    'name': 'Machines et appareils mécaniques',
                    'best_model': 'xgboost',
                    'optimal_threshold': 0.20,
                    'features_count': 43,
                    'business_features_count': 18
                },
                'chap85': {
                    'name': 'Machines et appareils électriques',
                    'best_model': 'lightgbm',
                    'optimal_threshold': 0.20,
                    'features_count': 43,
                    'business_features_count': 16
                }
            },
            'ml': {
                'models_path': 'backend/results',
                'data_path': 'backend/data',
                'train_val_test_split': [0.7, 0.15, 0.15],
                'random_state': 42
            },
            'rl': {
                'epsilon_decay': 0.995,
                'learning_rate': 0.1,
                'exploration_rate': 0.1,
                'context_window': 100
            },
            'ocr': {
                'input_path': 'backend/data/ocr/input',
                'processed_path': 'backend/data/ocr/processed',
                'confidence_threshold': 0.8
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': False,
                'reload': False
            }
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """Retourne la configuration de base de données"""
        return self.config.get('database', {})
    
    def get_database_url(self) -> str:
        """Retourne l'URL de connexion à la base de données"""
        db_config = self.get_database_config()
        return f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    def get_chapter_config(self, chapter_id: str) -> Dict[str, Any]:
        """Retourne la configuration d'un chapitre"""
        return self.config.get('chapters', {}).get(chapter_id, {})
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Retourne la configuration ML"""
        return self.config.get('ml', {})
    
    def get_rl_config(self) -> Dict[str, Any]:
        """Retourne la configuration RL"""
        return self.config.get('rl', {})
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """Retourne la configuration OCR"""
        return self.config.get('ocr', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Retourne la configuration API"""
        return self.config.get('api', {})
    
    def get_all_chapters(self) -> list:
        """Retourne la liste de tous les chapitres"""
        return list(self.config.get('chapters', {}).keys())
    
    def get_best_model_for_chapter(self, chapter_id: str) -> str:
        """Retourne le meilleur modèle pour un chapitre"""
        chapter_config = self.get_chapter_config(chapter_id)
        return chapter_config.get('best_model', 'xgboost')
    
    def get_optimal_threshold_for_chapter(self, chapter_id: str) -> float:
        """Retourne le seuil optimal pour un chapitre"""
        chapter_config = self.get_chapter_config(chapter_id)
        return chapter_config.get('optimal_threshold', 0.5)
    
    def get_features_count_for_chapter(self, chapter_id: str) -> int:
        """Retourne le nombre de features pour un chapitre"""
        chapter_config = self.get_chapter_config(chapter_id)
        return chapter_config.get('features_count', 0)
    
    def save_config(self) -> bool:
        """Sauvegarde la configuration dans le fichier YAML"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"Configuration sauvegardée: {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Erreur sauvegarde configuration: {e}")
            return False
    
    def update_chapter_config(self, chapter_id: str, updates: Dict[str, Any]) -> bool:
        """Met à jour la configuration d'un chapitre"""
        try:
            if 'chapters' not in self.config:
                self.config['chapters'] = {}
            
            if chapter_id not in self.config['chapters']:
                self.config['chapters'][chapter_id] = {}
            
            self.config['chapters'][chapter_id].update(updates)
            return self.save_config()
        except Exception as e:
            logger.error(f"Erreur mise à jour configuration chapitre {chapter_id}: {e}")
            return False
    
    def validate_config(self) -> bool:
        """Valide la configuration"""
        try:
            # Vérifier la configuration de base de données
            db_config = self.get_database_config()
            required_db_keys = ['host', 'port', 'user', 'password', 'database']
            for key in required_db_keys:
                if key not in db_config:
                    logger.error(f"Configuration base de données manquante: {key}")
                    return False
            
            # Vérifier la configuration des chapitres
            chapters = self.get_all_chapters()
            if not chapters:
                logger.error("Aucun chapitre configuré")
                return False
            
            for chapter_id in chapters:
                chapter_config = self.get_chapter_config(chapter_id)
                required_chapter_keys = ['name', 'best_model', 'optimal_threshold']
                for key in required_chapter_keys:
                    if key not in chapter_config:
                        logger.error(f"Configuration chapitre {chapter_id} manquante: {key}")
                        return False
            
            logger.info("✅ Configuration validée avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur validation configuration: {e}")
            return False

# Instance globale de configuration
config = IntegrationConfig()

# Fonctions utilitaires
def get_database_url() -> str:
    """Retourne l'URL de connexion à la base de données"""
    return config.get_database_url()

def get_chapter_config(chapter_id: str) -> Dict[str, Any]:
    """Retourne la configuration d'un chapitre"""
    return config.get_chapter_config(chapter_id)

def get_best_model_for_chapter(chapter_id: str) -> str:
    """Retourne le meilleur modèle pour un chapitre"""
    return config.get_best_model_for_chapter(chapter_id)

def get_optimal_threshold_for_chapter(chapter_id: str) -> float:
    """Retourne le seuil optimal pour un chapitre"""
    return config.get_optimal_threshold_for_chapter(chapter_id)

def get_all_chapters() -> list:
    """Retourne la liste de tous les chapitres"""
    return config.get_all_chapters()

def validate_integration_config() -> bool:
    """Valide la configuration d'intégration"""
    return config.validate_config()

# Configuration des variables d'environnement
def setup_environment_variables():
    """Configure les variables d'environnement à partir de la configuration"""
    try:
        db_config = config.get_database_config()
        
        os.environ['DATABASE_URL'] = config.get_database_url()
        os.environ['DB_HOST'] = str(db_config['host'])
        os.environ['DB_PORT'] = str(db_config['port'])
        os.environ['DB_USER'] = db_config['user']
        os.environ['DB_PASSWORD'] = db_config['password']
        os.environ['DB_NAME'] = db_config['database']
        
        api_config = config.get_api_config()
        os.environ['API_HOST'] = api_config['host']
        os.environ['API_PORT'] = str(api_config['port'])
        os.environ['API_DEBUG'] = str(api_config['debug'])
        
        logger.info("✅ Variables d'environnement configurées")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

        
    except Exception as e:
        logger.error(f"❌ Erreur configuration variables d'environnement: {e}")
        return False

if __name__ == "__main__":
    # Test de la configuration
    logger.info("🧪 Test de la configuration d'intégration")
    
    if validate_integration_config():
        logger.info("✅ Configuration valide")
        
        # Afficher la configuration
        logger.info(f"Base de données: {get_database_url()}")
        logger.info(f"Chapitres: {get_all_chapters()}")
        
        for chapter_id in get_all_chapters():
            chapter_config = get_chapter_config(chapter_id)
            logger.info(f"  {chapter_id}: {chapter_config['name']} - {chapter_config['best_model']}")
        
        # Configurer les variables d'environnement
        setup_environment_variables()
        
    else:
        logger.error("❌ Configuration invalide")
        exit(1)
# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")

"""
Configuration d'intégration PostgreSQL avec le backend InspectIA
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

# Configuration de l'intégration
INTEGRATION_CONFIG = {
    # Base de données
    "database": {
        "postgresql_enabled": True,
        "sqlite_enabled": False,  # Désactivé pour utiliser PostgreSQL
        "postgresql_primary": True,
        "connection_pool_size": 10,
        "connection_timeout": 30
    },
    
    # API Routes
    "api": {
        "postgresql_prefix": "/api/v2",
        "legacy_prefix": "/api/v1",
        "enable_hybrid_mode": False,  # Mode hybride désactivé
        "postgresql_only": True  # Utiliser uniquement PostgreSQL
    },
    
    # Migration
    "migration": {
        "auto_migrate": True,
        "backup_before_migrate": True,
        "migrate_sqlite_data": False,  # Pas de migration SQLite
        "validate_after_migrate": True
    },
    
    # Logging
    "logging": {
        "log_database_queries": False,
        "log_performance": True,
        "log_errors": True
    },
    
    # Performance
    "performance": {
        "enable_caching": True,
        "cache_ttl": 300,  # 5 minutes
        "enable_connection_pooling": True,
        "max_connections": 20
    }
}

def get_integration_config() -> Dict[str, Any]:
    """Retourne la configuration d'intégration"""
    return INTEGRATION_CONFIG

def get_database_config() -> Dict[str, Any]:
    """Retourne la configuration de base de données"""
    return INTEGRATION_CONFIG["database"]

def get_api_config() -> Dict[str, Any]:
    """Retourne la configuration API"""
    return INTEGRATION_CONFIG["api"]

def is_postgresql_enabled() -> bool:
    """Vérifie si PostgreSQL est activé"""
    return INTEGRATION_CONFIG["database"]["postgresql_enabled"]

def is_sqlite_enabled() -> bool:
    """Vérifie si SQLite est activé"""
    return INTEGRATION_CONFIG["database"]["sqlite_enabled"]

def is_hybrid_mode() -> bool:
    """Vérifie si le mode hybride est activé"""
    return INTEGRATION_CONFIG["api"]["enable_hybrid_mode"]

def get_postgresql_prefix() -> str:
    """Retourne le préfixe des routes PostgreSQL"""
    return INTEGRATION_CONFIG["api"]["postgresql_prefix"]

def get_legacy_prefix() -> str:
    """Retourne le préfixe des routes legacy"""
    return INTEGRATION_CONFIG["api"]["legacy_prefix"]

# Configuration des endpoints
ENDPOINT_MAPPING = {
    # Endpoints PostgreSQL (nouveaux)
    "postgresql": {
        "declarations": "/api/v2/declarations/",
        "predictions": "/api/v2/predictions/",
        "features": "/api/v2/features/",
        "models": "/api/v2/models/",
        "rl_decisions": "/api/v2/rl/decisions/",
        "rl_feedback": "/api/v2/rl/feedback/",
        "stats": "/api/v2/stats/",
        "health": "/api/v2/health/"
    },
    
    # Endpoints legacy (anciens)
    "legacy": {
        "predict": "/predict",
        "upload": "/upload",
        "feedback": "/feedback",
        "analytics": "/analytics"
    }
}

def get_endpoint_mapping() -> Dict[str, Dict[str, str]]:
    """Retourne le mapping des endpoints"""
    return ENDPOINT_MAPPING

def get_postgresql_endpoints() -> Dict[str, str]:
    """Retourne les endpoints PostgreSQL"""
    return ENDPOINT_MAPPING["postgresql"]

def get_legacy_endpoints() -> Dict[str, str]:
    """Retourne les endpoints legacy"""
    return ENDPOINT_MAPPING["legacy"]

# Configuration des features
FEATURE_CONFIG = {
    "chapters": {
        "chap30": {
            "enabled": True,
            "features_count": 22,
            "model_type": "xgboost",
            "fraud_rate": 0.108
        },
        "chap84": {
            "enabled": True,
            "features_count": 21,
            "model_type": "catboost",
            "fraud_rate": 0.108
        },
        "chap85": {
            "enabled": True,
            "features_count": 23,
            "model_type": "xgboost",
            "fraud_rate": 0.192
        }
    }
}

def get_feature_config() -> Dict[str, Any]:
    """Retourne la configuration des features"""
    return FEATURE_CONFIG

def get_chapter_config(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Retourne la configuration d'un chapitre"""
    return FEATURE_CONFIG["chapters"].get(chapter_id)

def is_chapter_enabled(chapter_id: str) -> bool:
    """Vérifie si un chapitre est activé"""
    chapter_config = get_chapter_config(chapter_id)
    return chapter_config.get("enabled", False) if chapter_config else False

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "environment": os.getenv("ENVIRONMENT", "development"),
    "debug": os.getenv("DEBUG", "false").lower() == "true",
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "workers": int(os.getenv("API_WORKERS", "4"))
}

def get_deployment_config() -> Dict[str, Any]:
    """Retourne la configuration de déploiement"""
    return DEPLOYMENT_CONFIG

def is_production() -> bool:
    """Vérifie si on est en production"""
    return DEPLOYMENT_CONFIG["environment"] == "production"

def is_development() -> bool:
    """Vérifie si on est en développement"""
    return DEPLOYMENT_CONFIG["environment"] == "development"

# Configuration de sécurité
SECURITY_CONFIG = {
    "cors_origins": ["*"],  # À restreindre en production
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "rate_limiting": {
        "enabled": True,
        "requests_per_minute": 100
    }
}

def get_security_config() -> Dict[str, Any]:
    """Retourne la configuration de sécurité"""
    return SECURITY_CONFIG

# Fonction utilitaire pour valider la configuration
def validate_config() -> bool:
    """Valide la configuration d'intégration"""
    try:
        # Vérifier la configuration de base de données
        db_config = get_database_config()
        if not db_config["postgresql_enabled"]:
            print("❌ PostgreSQL doit être activé")
            return False
        
        # Vérifier la configuration API
        api_config = get_api_config()
        if not api_config["postgresql_only"]:
            print("⚠️ Mode hybride détecté - PostgreSQL uniquement recommandé")
        
        # Vérifier les features
        feature_config = get_feature_config()
        for chapter_id, config in feature_config["chapters"].items():
            if not config["enabled"]:
                print(f"⚠️ Chapitre {chapter_id} désactivé")
        
        print("✅ Configuration validée")
        return True
        
    except Exception as e:
        print(f"❌ Erreur validation configuration: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuration d'intégration PostgreSQL")
    print("=" * 50)
    
    # Afficher la configuration
    print("📊 Configuration de base de données:")
    db_config = get_database_config()
    for key, value in db_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Configuration API:")
    api_config = get_api_config()
    for key, value in api_config.items():
        print(f"   - {key}: {value}")
    
    print("\n📊 Endpoints PostgreSQL:")
    postgresql_endpoints = get_postgresql_endpoints()
    for name, endpoint in postgresql_endpoints.items():
        print(f"   - {name}: {endpoint}")
    
    print("\n📊 Configuration des chapitres:")
    feature_config = get_feature_config()
    for chapter_id, config in feature_config["chapters"].items():
        print(f"   - {chapter_id}: {config['features_count']} features, {config['model_type']}")
    
    # Valider la configuration
    print("\n🔍 Validation de la configuration...")
    if validate_config():
        print("🎉 Configuration valide!")
    else:
        print("❌ Configuration invalide!")
