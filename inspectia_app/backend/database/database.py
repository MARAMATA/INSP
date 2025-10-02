"""
Module principal de base de données pour INSPECT_IA
Gère les connexions et les opérations de base de données
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gestionnaire de base de données INSPECT_IA"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or self._get_database_url()
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _get_database_url(self) -> str:
        """Obtient l'URL de la base de données depuis les variables d'environnement"""
        return os.getenv(
            'DATABASE_URL',
            'postgresql://maramata:maramata@localhost:5432/INSPECT_IA'
        )
    
    def _initialize_engine(self):
        """Initialise le moteur SQLAlchemy"""
        try:
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=int(os.getenv('DB_POOL_SIZE', '15')),
                max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '25')),
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=os.getenv('DB_ECHO', 'false').lower() == 'true',
                echo_pool=os.getenv('DB_ECHO_POOL', 'false').lower() == 'true'
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info("✅ Moteur de base de données initialisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation moteur DB: {e}")
            raise
    
    def get_session(self) -> Session:
        """Obtient une session de base de données"""
        if self.SessionLocal is None:
            self._initialize_engine()
        return self.SessionLocal()
    
    @contextmanager
    def get_session_context(self):
        """Contexte de session de base de données"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Erreur session DB: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """Teste la connexion à la base de données"""
        try:
            with self.get_session_context() as session:
                result = session.execute(text("SELECT 1"))
                result.fetchone()
            logger.info("✅ Connexion à la base de données réussie")
            return True
        except Exception as e:
            logger.error(f"❌ Erreur connexion DB: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """Obtient les informations sur la base de données"""
        try:
            with self.get_session_context() as session:
                # Version PostgreSQL
                version_result = session.execute(text("SELECT version()"))
                version = version_result.fetchone()[0]
                
                # Nombre de tables
                tables_result = session.execute(text("""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                tables_count = tables_result.fetchone()[0]
                
                # Taille de la base de données
                size_result = session.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """))
                db_size = size_result.fetchone()[0]
                
                return {
                    'version': version,
                    'tables_count': tables_count,
                    'database_size': db_size,
                    'connection_url': self.database_url
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération infos DB: {e}")
            return {}
    
    def close(self):
        """Ferme la connexion à la base de données"""
        if self.engine:
            self.engine.dispose()
            logger.info("✅ Connexion à la base de données fermée")

class PostgreSQLConnection:
    """Connexion PostgreSQL directe pour les opérations spécifiques"""
    
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or self._get_database_url()
        self.connection = None
    
    def _get_database_url(self) -> str:
        """Obtient l'URL de la base de données"""
        return os.getenv(
            'DATABASE_URL',
            'postgresql://maramata:maramata@localhost:5432/INSPECT_IA'
        )
    
    def connect(self):
        """Établit une connexion PostgreSQL"""
        try:
            self.connection = psycopg2.connect(self.database_url)
            logger.info("✅ Connexion PostgreSQL établie")
        except Exception as e:
            logger.error(f"❌ Erreur connexion PostgreSQL: {e}")
            raise
    
    def get_cursor(self, dict_cursor: bool = False):
        """Obtient un curseur PostgreSQL"""
        if self.connection is None:
            self.connect()
        
        if dict_cursor:
            return self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            return self.connection.cursor()
    
    def execute_query(self, query: str, params: tuple = None, fetch: bool = True):
        """Exécute une requête PostgreSQL"""
        try:
            cursor = self.get_cursor()
            cursor.execute(query, params)
            
            if fetch:
                result = cursor.fetchall()
                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")

                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur exécution requête: {e}")
            if self.connection:
                self.connection.rollback()
            raise
    
    def close(self):
        """Ferme la connexion PostgreSQL"""
        if self.connection:
            self.connection.close()
            logger.info("✅ Connexion PostgreSQL fermée")

# Instances globales
db_manager = DatabaseManager()
postgres_connection = PostgreSQLConnection()

# Fonctions utilitaires
def get_database_session() -> Session:
    """Obtient une session de base de données"""
    return db_manager.get_session()

@contextmanager
def get_database_session_context():
    """Contexte de session de base de données"""
    with db_manager.get_session_context() as session:
        yield session

def test_database_connection() -> bool:
    """Teste la connexion à la base de données"""
    return db_manager.test_connection()

def get_database_info() -> Dict[str, Any]:
    """Obtient les informations sur la base de données"""
    return db_manager.get_database_info()

def execute_postgresql_query(query: str, params: tuple = None, fetch: bool = True):
    """Exécute une requête PostgreSQL"""
    return postgres_connection.execute_query(query, params, fetch)

# Configuration de l'engine SQLAlchemy pour les modèles
engine = db_manager.engine
SessionLocal = db_manager.SessionLocal

# Fonction de dépendance pour FastAPI
def get_db():
    """Dépendance FastAPI pour obtenir une session de base de données"""
    db = get_database_session()
    try:
        yield db
    finally:
        db.close()

# Initialisation des tables
def create_tables():
    """Crée toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables de base de données créées")
    except Exception as e:
        logger.error(f"❌ Erreur création tables: {e}")
        raise

def drop_tables():
    """Supprime toutes les tables de la base de données"""
    try:
        from database.models import Base
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Tables de base de données supprimées")
    except Exception as e:
        logger.error(f"❌ Erreur suppression tables: {e}")
        raise

# Fonctions de maintenance
def vacuum_database():
    """Effectue un VACUUM sur la base de données"""
    try:
        with get_database_session_context() as session:
            session.execute(text("VACUUM ANALYZE"))
        logger.info("✅ VACUUM de la base de données effectué")
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")

def get_table_stats():
    """Obtient les statistiques des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_tup_ins as inserts,
            n_tup_upd as updates,
            n_tup_del as deletes,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples
        FROM pg_stat_user_tables
        ORDER BY n_live_tup DESC
        """
        
        results = execute_postgresql_query(query)
        return results
        
    except Exception as e:
        logger.error(f"❌ Erreur statistiques tables: {e}")
        return []

# Fonction de nettoyage
def cleanup_connections():
    """Nettoie toutes les connexions"""
    try:
        db_manager.close()
        postgres_connection.close()
        logger.info("✅ Connexions nettoyées")
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage connexions: {e}")

if __name__ == "__main__":
    # Test de la base de données
    logger.info("🧪 Test de la base de données INSPECT_IA")
    
    if test_database_connection():
        logger.info("✅ Connexion réussie")
        
        info = get_database_info()
        logger.info(f"📊 Informations base de données:")
        logger.info(f"   Version: {info.get('version', 'N/A')}")
        logger.info(f"   Tables: {info.get('tables_count', 'N/A')}")
        logger.info(f"   Taille: {info.get('database_size', 'N/A')}")
        
        # Test des statistiques
        stats = get_table_stats()
        if stats:
            logger.info(f"📈 Statistiques des tables: {len(stats)} tables")
        
    else:
        logger.error("❌ Connexion échouée")
        sys.exit(1)
# Création de l'engine SQLAlchemy
engine = create_engine(
    db_config['sqlalchemy_database_url'],
    poolclass=QueuePool,
    pool_size=db_config['pool_size'],
    max_overflow=db_config['max_overflow'],
    pool_timeout=db_config['pool_timeout'],
    pool_recycle=db_config['pool_recycle'],
    echo=db_config['echo'],
    future=True  # Utilise SQLAlchemy 2.0 style
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base pour les modèles
Base = declarative_base()

# Métadonnées
metadata = MetaData()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour FastAPI - fournit une session de base de données
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager pour les sessions de base de données
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Erreur de base de données: {e}")
        raise
    finally:
        db.close()

def init_database():
    """
    Initialise la base de données (crée les tables si elles n'existent pas)
    """
    try:
        # Créer toutes les tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de données initialisée avec succès")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation: {e}")
        return False

def test_connection():
    """
    Teste la connexion à la base de données
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Connexion à la base de données réussie")
            return True
    except Exception as e:
        logger.error(f"❌ Erreur de connexion: {e}")
        return False

def get_engine():
    """
    Retourne l'engine SQLAlchemy
    """
    return engine

def get_session_local():
    """
    Retourne la session factory
    """
    return SessionLocal

# Configuration pour Alembic (migrations)
alembic_config = {
    'script_location': 'backend/database/migrations',
    'sqlalchemy.url': db_config['alembic_database_url'],
    'file_template': '%(year)d_%(month).2d_%(day).2d_%(hour).2d%(minute).2d_%(rev)s_%(slug)s',
    'timezone': 'UTC'
}

# Informations de connexion pour le debugging
def get_connection_info():
    """
    Retourne les informations de connexion (sans mot de passe)
    """
    url = db_config['sqlalchemy_database_url']
    # Masquer le mot de passe
    if '@' in url:
        parts = url.split('@')
        if ':' in parts[0]:
            user_pass = parts[0].split('://')[1]
            if ':' in user_pass:
                user = user_pass.split(':')[0]
                masked_url = url.replace(f':{user_pass.split(":")[1]}', ':***')
            else:
                masked_url = url
        else:
            masked_url = url
    else:
        masked_url = url
    
    return {
        'database_url': masked_url,
        'pool_size': db_config['pool_size'],
        'max_overflow': db_config['max_overflow'],
        'echo': db_config['echo']
    }

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration SQLAlchemy InspectIA")
    print("=" * 50)
    
    # Informations de connexion
    conn_info = get_connection_info()
    print(f"📊 URL de base de données: {conn_info['database_url']}")
    print(f"📊 Pool size: {conn_info['pool_size']}")
    print(f"📊 Max overflow: {conn_info['max_overflow']}")
    print(f"📊 Echo: {conn_info['echo']}")
    
    # Test de connexion
    print("\n🔍 Test de connexion...")
    if test_connection():
        print("✅ Connexion réussie")
    else:
        print("❌ Échec de la connexion")
    
    # Test d'initialisation
    print("\n🏗️ Test d'initialisation...")
    if init_database():
        print("✅ Initialisation réussie")
    else:
        print("❌ Échec de l'initialisation")
