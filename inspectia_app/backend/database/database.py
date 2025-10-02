"""
Module principal de base de données pour INSPECT_IA
Gère les connexions et les opérations de base de données
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Generator, List, Tuple
from datetime import datetime, timedelta
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, text, MetaData
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import json
import time

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

# Configuration de la base de données
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432')),
    'database': os.getenv('DB_NAME', 'INSPECT_IA'),
    'user': os.getenv('DB_USER', 'maramata'),
    'password': os.getenv('DB_PASSWORD', 'maramata'),
    'pool_size': int(os.getenv('DB_POOL_SIZE', '15')),
    'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '25')),
    'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
    'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '3600')),
    'echo': os.getenv('DB_ECHO', 'false').lower() == 'true'
}

# URLs de connexion
db_config['sqlalchemy_database_url'] = (
    f"postgresql://{db_config['user']}:{db_config['password']}"
    f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)

db_config['alembic_database_url'] = db_config['sqlalchemy_database_url']

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

# Fonctions de maintenance et utilitaires avancés
def backup_database(backup_path: str = None):
    """Effectue une sauvegarde de la base de données"""
    try:
        if backup_path is None:
            backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        
        # Utiliser pg_dump pour la sauvegarde
        import subprocess
        cmd = [
            'pg_dump',
            f'--host={db_config["host"]}',
            f'--port={db_config["port"]}',
            f'--username={db_config["user"]}',
            f'--dbname={db_config["database"]}',
            '--file', backup_path,
            '--verbose'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Sauvegarde créée: {backup_path}")
            return True
        else:
            logger.error(f"❌ Erreur sauvegarde: {result.stderr}")
            return False
                
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde: {e}")
        return False

def restore_database(backup_path: str):
    """Restaure la base de données depuis une sauvegarde"""
    try:
        import subprocess
        cmd = [
            'psql',
            f'--host={db_config["host"]}',
            f'--port={db_config["port"]}',
            f'--username={db_config["user"]}',
            f'--dbname={db_config["database"]}',
            '--file', backup_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Base de données restaurée depuis: {backup_path}")
            return True
        else:
            logger.error(f"❌ Erreur restauration: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur restauration: {e}")
        return False

def get_database_size():
    """Obtient la taille de la base de données"""
    try:
        query = """
        SELECT pg_size_pretty(pg_database_size(current_database())) as size
        """
        result = execute_postgresql_query(query)
        if result:
            return result[0][0] if isinstance(result[0], tuple) else result[0]['size']
        return "Unknown"
    except Exception as e:
        logger.error(f"❌ Erreur taille DB: {e}")
        return "Error"

def get_table_sizes():
    """Obtient la taille de toutes les tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
            pg_total_relation_size(schemaname||'.'||tablename) as size_bytes
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur tailles tables: {e}")
        return []

def get_index_info():
    """Obtient les informations sur les index"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            indexname,
            indexdef
        FROM pg_indexes 
        WHERE schemaname = 'public'
        ORDER BY tablename, indexname
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur infos index: {e}")
        return []

def optimize_database():
    """Optimise la base de données"""
    try:
        with get_database_session_context() as session:
            # VACUUM ANALYZE sur toutes les tables
            session.execute(text("VACUUM ANALYZE"))
            
            # REINDEX sur les index
            session.execute(text("REINDEX DATABASE " + db_config['database']))
            
        logger.info("✅ Optimisation de la base de données terminée")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur optimisation: {e}")
        return False

def get_connection_pool_status():
    """Obtient le statut du pool de connexions"""
    try:
        return {
            'pool_size': db_config['pool_size'],
            'max_overflow': db_config['max_overflow'],
            'pool_timeout': db_config['pool_timeout'],
            'pool_recycle': db_config['pool_recycle'],
            'echo': db_config['echo']
        }
    except Exception as e:
        logger.error(f"❌ Erreur statut pool: {e}")
        return {}

def get_active_connections():
    """Obtient le nombre de connexions actives"""
    try:
        query = """
        SELECT 
            count(*) as active_connections,
            state
        FROM pg_stat_activity 
        WHERE datname = current_database()
        GROUP BY state
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur connexions actives: {e}")
        return []

def get_slow_queries(limit: int = 10):
    """Obtient les requêtes lentes"""
    try:
        if not is_pg_stat_statements_available():
            return []
        
        query = """
        SELECT 
            query,
            mean_exec_time,
            calls,
            total_exec_time
        FROM pg_stat_statements 
        ORDER BY mean_exec_time DESC 
        LIMIT %s
        """
        return execute_postgresql_query(query, (limit,))
    except Exception as e:
        logger.error(f"❌ Erreur requêtes lentes: {e}")
        return []

def get_database_locks():
    """Obtient les verrous de base de données"""
    try:
        query = """
        SELECT 
            locktype,
            database,
            relation,
            page,
            tuple,
            virtualxid,
            transactionid,
            classid,
            objid,
            objsubid,
            virtualtransaction,
            pid,
            mode,
            granted
        FROM pg_locks 
        WHERE database = (SELECT oid FROM pg_database WHERE datname = current_database())
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur verrous: {e}")
        return []

def kill_connection(pid: int):
    """Tue une connexion par son PID"""
    try:
        with get_database_session_context() as session:
            session.execute(text(f"SELECT pg_terminate_backend({pid})"))
        logger.info(f"✅ Connexion {pid} terminée")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur terminaison connexion: {e}")
        return False

def get_database_performance():
    """Obtient les métriques de performance de la base de données"""
    try:
        metrics = {}
        
        # Taille de la base
        metrics['database_size'] = get_database_size()
        
        # Connexions actives
        metrics['active_connections'] = get_active_connections()
        
        # Statut du pool
        metrics['pool_status'] = get_connection_pool_status()
        
        # Statistiques des tables
        metrics['table_stats'] = get_table_stats()
        
        # Tailles des tables
        metrics['table_sizes'] = get_table_sizes()
        
        return metrics
        
    except Exception as e:
        logger.error(f"❌ Erreur métriques performance: {e}")
        return {}

# Fonctions de monitoring et d'analyse avancées
def is_pg_stat_statements_available():
    """Vérifie si pg_stat_statements est disponible et fonctionnel"""
    try:
        # Vérifier d'abord si l'extension est installée
        check_query = """
        SELECT EXISTS (
            SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
        ) as extension_exists
        """
        result = execute_postgresql_query(check_query)
        if not result or not (result[0][0] if isinstance(result[0], tuple) else result[0]['extension_exists']):
            return False
        
        # Tester si l'extension est vraiment fonctionnelle (sans générer d'erreur)
        try:
            # Utiliser une connexion directe pour éviter les logs d'erreur
            import psycopg2
            conn = psycopg2.connect('postgresql://maramata:maramata@localhost:5432/INSPECT_IA')
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM pg_stat_statements LIMIT 1")
            cur.fetchone()
            cur.close()
            conn.close()
            return True
        except Exception:
            # L'extension est installée mais pas chargée via shared_preload_libraries
            return False
    except Exception:
        return False

def get_query_performance_stats():
    """Obtient les statistiques de performance des requêtes"""
    try:
        if not is_pg_stat_statements_available():
            return []
        
        query = """
        SELECT 
            query,
            calls,
            total_exec_time,
            mean_exec_time,
            stddev_exec_time,
            rows,
            100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) AS hit_percent
        FROM pg_stat_statements 
        ORDER BY total_exec_time DESC 
        LIMIT 20
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats performance: {e}")
        return []

def get_database_health_check():
    """Effectue un contrôle de santé complet de la base de données"""
    try:
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
    
    # Test de connexion
        if not test_connection():
            health_status['status'] = 'unhealthy'
            health_status['issues'].append('Impossible de se connecter à la base de données')
            return health_status
        
        # Vérification de la taille de la base
        db_size = get_database_size()
        health_status['metrics']['database_size'] = db_size
        
        # Vérification des connexions actives
        active_conns = get_active_connections()
        health_status['metrics']['active_connections'] = active_conns
        
        # Vérification des verrous
        locks = get_database_locks()
        if locks:
            health_status['metrics']['active_locks'] = len(locks)
            if len(locks) > 100:  # Seuil d'alerte
                health_status['warnings'].append(f'Trop de verrous actifs: {len(locks)}')
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Erreur contrôle santé: {e}")
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'error',
            'error': str(e)
        }

def get_table_analysis():
    """Analyse détaillée des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            attname as column_name,
            n_distinct,
            correlation,
            most_common_vals,
            most_common_freqs,
            histogram_bounds,
            null_frac,
            avg_width
        FROM pg_stats 
        WHERE schemaname = 'public'
        ORDER BY tablename, attname
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur analyse tables: {e}")
        return []

def get_index_usage_stats():
    """Statistiques d'utilisation des index"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            indexname,
            idx_scan,
            idx_tup_read,
            idx_tup_fetch
        FROM pg_stat_user_indexes 
        WHERE schemaname = 'public'
        ORDER BY idx_scan DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats index: {e}")
        return []

def get_database_configuration():
    """Configuration actuelle de la base de données"""
    try:
        query = """
        SELECT 
            name,
            setting,
            unit,
            context,
            category,
            short_desc
        FROM pg_settings 
        WHERE name IN (
            'max_connections',
            'shared_buffers',
            'effective_cache_size',
            'maintenance_work_mem',
            'checkpoint_completion_target',
            'wal_buffers',
            'default_statistics_target',
            'random_page_cost',
            'effective_io_concurrency'
        )
        ORDER BY category, name
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur configuration: {e}")
        return []

def generate_database_report():
    """Génère un rapport complet de la base de données"""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'database_info': get_database_info(),
            'health_check': get_database_health_check(),
            'performance_metrics': get_database_performance(),
            'table_stats': get_table_stats(),
            'configuration': get_database_configuration()
        }
        
        # Sauvegarder le rapport
        report_file = f"database_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Rapport de base de données généré: {report_file}")
        return report
                
    except Exception as e:
        logger.error(f"❌ Erreur génération rapport: {e}")
        return {}

# Fonctions utilitaires supplémentaires
def execute_custom_query(query: str, params: tuple = None):
    """Exécute une requête personnalisée"""
    try:
        return execute_postgresql_query(query, params)
    except Exception as e:
        logger.error(f"❌ Erreur requête personnalisée: {e}")
        return []

def get_database_users():
    """Obtient la liste des utilisateurs de la base de données"""
    try:
        query = """
        SELECT 
            usename,
            usesuper,
            usecreatedb,
            usebypassrls,
            valuntil,
            useconfig
        FROM pg_user
        ORDER BY usename
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur utilisateurs: {e}")
        return []

def get_database_roles():
    """Obtient la liste des rôles de la base de données"""
    try:
        query = """
        SELECT 
            rolname,
            rolsuper,
            rolinherit,
            rolcreaterole,
            rolcreatedb,
            rolcanlogin,
            rolreplication,
            rolconnlimit,
            rolvaliduntil
        FROM pg_roles
        ORDER BY rolname
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur rôles: {e}")
        return []

def get_database_schemas():
    """Obtient la liste des schémas"""
    try:
        query = """
        SELECT 
            schema_name,
            schema_owner,
            schema_acl
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
        ORDER BY schema_name
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur schémas: {e}")
        return []

def get_table_constraints():
    """Obtient les contraintes des tables"""
    try:
        query = """
        SELECT 
            tc.table_schema,
            tc.table_name,
            tc.constraint_name,
            tc.constraint_type,
            kcu.column_name,
            ccu.table_schema AS foreign_table_schema,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        LEFT JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE tc.table_schema = 'public'
        ORDER BY tc.table_name, tc.constraint_type
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur contraintes: {e}")
        return []

def get_foreign_keys():
    """Obtient les clés étrangères"""
    try:
        query = """
        SELECT 
            tc.table_schema,
            tc.table_name,
            tc.constraint_name,
            kcu.column_name,
            ccu.table_schema AS foreign_table_schema,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY'
        AND tc.table_schema = 'public'
        ORDER BY tc.table_name, tc.constraint_name
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur clés étrangères: {e}")
        return []

def get_database_triggers():
    """Obtient la liste des triggers"""
    try:
        query = """
        SELECT 
            trigger_schema,
            trigger_name,
            event_manipulation,
            event_object_table,
            action_statement,
            action_timing,
            action_orientation
        FROM information_schema.triggers
        WHERE trigger_schema = 'public'
        ORDER BY event_object_table, trigger_name
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur triggers: {e}")
        return []

def get_database_functions():
    """Obtient la liste des fonctions"""
    try:
        query = """
        SELECT 
            routine_schema,
            routine_name,
            routine_type,
            data_type,
            routine_definition
        FROM information_schema.routines
        WHERE routine_schema = 'public'
        ORDER BY routine_name
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur fonctions: {e}")
        return []

def get_database_sequences():
    """Obtient la liste des séquences"""
    try:
        query = """
        SELECT 
            sequence_schema,
            sequence_name,
            data_type,
            start_value,
            minimum_value,
            maximum_value,
            increment,
            cycle_option
        FROM information_schema.sequences
        WHERE sequence_schema = 'public'
        ORDER BY sequence_name
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur séquences: {e}")
        return []

def get_database_views():
    """Obtient la liste des vues"""
    try:
        query = """
        SELECT 
            table_schema,
            table_name,
            view_definition
        FROM information_schema.views
        WHERE table_schema = 'public'
        ORDER BY table_name
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur vues: {e}")
        return []

# Fonctions de gestion des transactions
def begin_transaction():
    """Démarre une transaction"""
    try:
        session = get_database_session()
        session.begin()
        return session
    except Exception as e:
        logger.error(f"❌ Erreur début transaction: {e}")
        return None

def commit_transaction(session: Session):
    """Commite une transaction"""
    try:
        session.commit()
        session.close()
        return True
    except Exception as e:
        logger.error(f"❌ Erreur commit transaction: {e}")
        session.rollback()
        session.close()
        return False

def rollback_transaction(session: Session):
    """Annule une transaction"""
    try:
        session.rollback()
        session.close()
        return True
    except Exception as e:
        logger.error(f"❌ Erreur rollback transaction: {e}")
        session.close()
        return False

# Fonctions spécialisées pour INSPECT_IA
def get_inspectia_tables_info():
    """Obtient les informations sur les tables spécifiques à INSPECT_IA"""
    try:
        inspectia_tables = [
            'advanced_decisions',
            'advanced_feedbacks', 
            'advanced_policies',
            'analysis_results',
            'chapter_features',
            'chapters',
            'declaration_features',
            'declarations',
            'features',
            'feedback_history',
            'inspector_profiles',
            'model_thresholds',
            'models',
            'performance_metrics',
            'predictions',
            'pv_inspection',
            'pvs',
            'rl_bandits',
            'rl_decisions',
            'rl_performance_metrics',
            'system_logs'
        ]
        
        tables_info = {}
        for table in inspectia_tables:
            try:
                query = f"""
                SELECT 
                    COUNT(*) as row_count,
                    pg_size_pretty(pg_total_relation_size('{table}')) as size
                FROM {table}
                """
                result = execute_postgresql_query(query)
                if result:
                    # Convertir le tuple en dictionnaire
                    if isinstance(result[0], tuple):
                        tables_info[table] = {
                            'row_count': result[0][0],
                            'size': result[0][1]
                        }
                    else:
                        tables_info[table] = result[0]
            except Exception as e:
                tables_info[table] = {'error': str(e)}
        
        return tables_info
    except Exception as e:
        logger.error(f"❌ Erreur tables INSPECT_IA: {e}")
        return {}

def get_declaration_stats():
    """Statistiques des déclarations"""
    try:
        query = """
        SELECT 
            COUNT(*) as total_declarations,
            COUNT(DISTINCT chapter_id) as chapters_used,
            AVG(valeur_caf) as avg_valeur_caf,
            AVG(poids_net_kg) as avg_poids_net,
            COUNT(CASE WHEN composite_fraud_score > 0.5 THEN 1 END) as high_risk_count,
            COUNT(CASE WHEN composite_fraud_score <= 0.5 THEN 1 END) as low_risk_count,
            MAX(created_at) as latest_declaration,
            MIN(created_at) as oldest_declaration
        FROM declarations
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats déclarations: {e}")
        return []

def get_prediction_accuracy():
    """Précision des prédictions"""
    try:
        query = """
        SELECT 
            p.chapter_id,
            COUNT(*) as total_predictions,
            AVG(p.fraud_probability) as avg_fraud_probability,
            AVG(p.confidence_score) as avg_confidence,
            COUNT(CASE WHEN p.predicted_fraud = true THEN 1 END) as fraud_predictions,
            COUNT(CASE WHEN p.predicted_fraud = false THEN 1 END) as clean_predictions,
            MAX(p.created_at) as latest_prediction
        FROM predictions p
        GROUP BY p.chapter_id
        ORDER BY total_predictions DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur précision prédictions: {e}")
        return []

def get_chapter_performance():
    """Performance par chapitre - adapté à la structure réelle de la DB"""
    try:
        query = """
        SELECT 
            c.chapter_id,
            c.chapter_name,
            c.chapter_number,
            COUNT(d.declaration_id) as declaration_count,
            AVG(d.valeur_caf) as avg_valeur_caf,
            AVG(d.composite_fraud_score) as avg_fraud_score,
            COUNT(CASE WHEN d.composite_fraud_score > 0.5 THEN 1 END) as high_risk_count,
            COUNT(CASE WHEN d.composite_fraud_score <= 0.5 THEN 1 END) as low_risk_count,
            ROUND(
                (COUNT(CASE WHEN d.composite_fraud_score > 0.5 THEN 1 END)::numeric / NULLIF(COUNT(d.declaration_id), 0)) * 100, 
                2
            ) as high_risk_percentage,
            c.fraud_rate,
            c.best_model,
            c.features_count,
            c.data_size,
            c.business_features_count,
            c.advanced_fraud_detection,
            c.optimal_threshold,
            c.model_performance,
            c.specialization
        FROM chapters c
        LEFT JOIN declarations d ON c.chapter_id = d.chapter_id
        GROUP BY c.chapter_id, c.chapter_name, c.chapter_number, c.fraud_rate, c.best_model, c.features_count,
                 c.data_size, c.business_features_count, c.advanced_fraud_detection, c.optimal_threshold,
                 c.model_performance, c.specialization
        ORDER BY declaration_count DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur performance chapitres: {e}")
        return []

def get_model_performance_metrics():
    """Métriques de performance des modèles"""
    try:
        query = """
        SELECT 
            m.model_name,
            m.chapter_id,
            m.model_type,
            m.version,
            m.is_best_model,
            m.is_active,
            m.training_date,
            m.performance_metrics,
            m.feature_list,
            m.created_at,
            m.updated_at
        FROM models m
        ORDER BY m.training_date DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur métriques modèles: {e}")
        return []

def get_rl_performance_stats():
    """Statistiques de performance du RL"""
    try:
        query = """
        SELECT 
            rd.chapter_id,
            COUNT(*) as total_decisions,
            AVG(rd.confidence_level) as avg_confidence,
            COUNT(CASE WHEN rd.exploration_exploitation = 'exploration' THEN 1 END) as exploration_count,
            COUNT(CASE WHEN rd.exploration_exploitation = 'exploitation' THEN 1 END) as exploitation_count,
            AVG(rd.reward_received) as avg_reward,
            AVG(rd.regret) as avg_regret,
            COUNT(DISTINCT rd.bandit_arm_selected) as arms_used,
            MAX(rd.created_at) as latest_decision
        FROM rl_decisions rd
        GROUP BY rd.chapter_id
        ORDER BY total_decisions DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats RL: {e}")
        return []

def get_inspector_performance():
    """Performance des inspecteurs - adapté à la structure réelle INSPECT_IA"""
    try:
        query = """
        SELECT 
            ip.inspector_id,
            ip.name as inspector_name,
            ip.expertise_level,
            ip.specialization,
            ip.chapter_id,
            ip.total_reviews,
            ip.avg_confidence,
            ip.avg_review_time,
            ip.performance_trend,
            ip.last_active,
            ip.accuracy_rate,
            ip.fraud_detection_rate,
            ip.false_positive_rate,
            ip.risk_tolerance,
            -- Analyse des feedbacks
            COUNT(af.feedback_id) as total_feedbacks,
            AVG(af.inspector_confidence) as avg_feedback_confidence,
            AVG(af.feedback_quality_score) as avg_feedback_quality,
            COUNT(CASE WHEN af.inspector_decision = 1 THEN 1 END) as fraud_feedbacks,
            COUNT(CASE WHEN af.inspector_decision = 0 THEN 1 END) as clean_feedbacks,
            MAX(af.created_at) as latest_feedback,
            -- Analyse des décisions RL
            COUNT(ad.decision_id) as total_rl_decisions,
            AVG(ad.confidence_score) as avg_rl_confidence,
            COUNT(CASE WHEN ad.action = 'inspect' THEN 1 END) as inspect_decisions,
            COUNT(CASE WHEN ad.action = 'flag' THEN 1 END) as flag_decisions,
            COUNT(CASE WHEN ad.action = 'clear' THEN 1 END) as clear_decisions,
            COUNT(CASE WHEN ad.action = 'pass' THEN 1 END) as pass_decisions
        FROM inspector_profiles ip
        LEFT JOIN advanced_feedbacks af ON ip.inspector_id = af.inspector_id
        LEFT JOIN advanced_decisions ad ON ip.inspector_id = ad.inspector_id
        GROUP BY ip.inspector_id, ip.name, ip.expertise_level, ip.specialization, ip.chapter_id,
                 ip.total_reviews, ip.avg_confidence, ip.avg_review_time, ip.performance_trend,
                 ip.last_active, ip.accuracy_rate, ip.fraud_detection_rate, ip.false_positive_rate,
                 ip.risk_tolerance
        ORDER BY total_feedbacks DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur performance inspecteurs: {e}")
        return []

def get_fraud_patterns_analysis():
    """Analyse des patterns de fraude basée sur les vraies données INSPECT_IA"""
    try:
        query = """
        SELECT 
            df.chapter_id,
            'HIGH_RISK_SCORE' as pattern_type,
            'Déclarations avec score de fraude élevé' as pattern_description,
            COUNT(*) as occurrence_count,
            AVG(df.composite_fraud_score) as avg_severity,
            'COMPOSITE_SCORE' as detection_method,
            AVG(df.mirror_tei_score) as avg_tei_score,
            AVG(df.spectral_cluster_score) as avg_spectral_score,
            AVG(df.bienayme_chebychev_score) as avg_chebychev_score,
            AVG(df.admin_values_score) as avg_admin_score,
            AVG(df.hierarchical_cluster_score) as avg_hierarchical_score,
            -- Analyse des patterns business
            COUNT(CASE WHEN df.business_glissement_cosmetique = true THEN 1 END) as cosmetic_shift_count,
            COUNT(CASE WHEN df.business_glissement_machine = true THEN 1 END) as machine_shift_count,
            COUNT(CASE WHEN df.business_glissement_electronique = true THEN 1 END) as electronic_shift_count,
            COUNT(CASE WHEN df.business_is_medicament = true THEN 1 END) as medicament_count,
            COUNT(CASE WHEN df.business_is_machine = true THEN 1 END) as machine_count,
            COUNT(CASE WHEN df.business_is_electronique = true THEN 1 END) as electronic_count,
            AVG(df.valeur_caf) as avg_valeur_caf,
            AVG(df.poids_net_kg) as avg_poids_net
        FROM declaration_features df
        WHERE df.composite_fraud_score > 0.7
        GROUP BY df.chapter_id
        ORDER BY occurrence_count DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur analyse patterns: {e}")
        return []

def get_drift_detection_results():
    """Résultats de détection de drift basés sur les vraies métriques INSPECT_IA"""
    try:
        query = """
        SELECT 
            c.chapter_id,
            c.chapter_name,
            c.chapter_number,
            'PERFORMANCE_MONITORING' as drift_type,
            CASE 
                WHEN c.fraud_rate > 0.3 THEN 'HIGH'
                WHEN c.fraud_rate > 0.15 THEN 'MEDIUM'
                ELSE 'LOW'
            END as severity_level,
            'Model Performance' as affected_features,
            CASE 
                WHEN c.fraud_rate > 0.3 THEN 'Retrain model immediately'
                WHEN c.fraud_rate > 0.15 THEN 'Monitor closely and consider retraining'
                ELSE 'Continue monitoring'
            END as recommended_action,
            c.model_performance,
            c.optimal_threshold,
            c.features_count,
            c.data_size,
            c.business_features_count,
            c.advanced_fraud_detection,
            c.specialization,
            -- Analyse des métriques RL pour détecter le drift
            COUNT(ad.decision_id) as total_rl_decisions,
            AVG(ad.confidence_score) as avg_rl_confidence,
            COUNT(CASE WHEN ad.exploration = true THEN 1 END) as exploration_count,
            COUNT(CASE WHEN ad.exploration = false THEN 1 END) as exploitation_count,
            -- Analyse des feedbacks pour détecter les changements
            COUNT(af.feedback_id) as total_feedbacks,
            AVG(af.feedback_quality_score) as avg_feedback_quality,
            COUNT(CASE WHEN af.inspector_decision = 1 THEN 1 END) as fraud_feedbacks,
            COUNT(CASE WHEN af.inspector_decision = 0 THEN 1 END) as clean_feedbacks
        FROM chapters c
        LEFT JOIN advanced_decisions ad ON c.chapter_id = ad.chapter_id
        LEFT JOIN advanced_feedbacks af ON c.chapter_id = af.chapter_id
        GROUP BY c.chapter_id, c.chapter_name, c.chapter_number, c.fraud_rate, c.model_performance,
                 c.optimal_threshold, c.features_count, c.data_size, c.business_features_count,
                 c.advanced_fraud_detection, c.specialization
        ORDER BY c.fraud_rate DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur détection drift: {e}")
        return []

def get_feature_importance_analysis():
    """Analyse de l'importance des features basée sur les vraies données INSPECT_IA"""
    try:
        query = """
        SELECT 
            f.feature_name,
            f.feature_type,
            f.feature_category,
            f.description,
            cf.chapter_id,
            cf.is_used,
            cf.importance_score,
            -- Analyse de l'utilisation des features dans les déclarations
            COUNT(df.feature_id) as usage_count,
            AVG(CASE 
                WHEN f.feature_name = 'VALEUR_CAF' THEN df.valeur_caf
                WHEN f.feature_name = 'POIDS_NET_KG' THEN df.poids_net_kg
                WHEN f.feature_name = 'COMPOSITE_FRAUD_SCORE' THEN df.composite_fraud_score
                WHEN f.feature_name = 'BIENAYME_CHEBYCHEV_SCORE' THEN df.bienayme_chebychev_score
                WHEN f.feature_name = 'MIRROR_TEI_SCORE' THEN df.mirror_tei_score
                WHEN f.feature_name = 'SPECTRAL_CLUSTER_SCORE' THEN df.spectral_cluster_score
                WHEN f.feature_name = 'HIERARCHICAL_CLUSTER_SCORE' THEN df.hierarchical_cluster_score
                WHEN f.feature_name = 'ADMIN_VALUES_SCORE' THEN df.admin_values_score
                ELSE NULL
            END) as avg_feature_value,
            -- Analyse de la corrélation avec les décisions RL
            COUNT(ad.decision_id) as rl_usage_count,
            AVG(ad.confidence_score) as avg_rl_confidence_when_used,
            -- Analyse de la corrélation avec les feedbacks
            COUNT(af.feedback_id) as feedback_correlation_count,
            AVG(af.feedback_quality_score) as avg_feedback_quality_when_used
        FROM features f
        LEFT JOIN chapter_features cf ON f.feature_id = cf.feature_id
        LEFT JOIN declaration_features df ON f.feature_id = df.feature_id
        LEFT JOIN advanced_decisions ad ON df.declaration_id = ad.declaration_id
        LEFT JOIN advanced_feedbacks af ON df.declaration_id = af.declaration_id
        GROUP BY f.feature_id, f.feature_name, f.feature_type, f.feature_category, f.description,
                 cf.chapter_id, cf.is_used, cf.importance_score
        ORDER BY cf.importance_score DESC NULLS LAST, usage_count DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur analyse importance: {e}")
        return []

def get_system_health_metrics():
    """Métriques de santé du système INSPECT_IA - adaptées à la structure réelle"""
    try:
        health_metrics = {
            'timestamp': datetime.now().isoformat(),
            'database_connection': test_connection(),
            'database_info': get_database_info(),
            'database_size': get_database_size(),
            'active_connections': get_active_connections(),
            'connection_pool_status': get_connection_pool_status(),
            # Métriques INSPECT_IA spécifiques
            'declaration_stats': get_declaration_stats(),
            'prediction_accuracy': get_prediction_accuracy(),
            'chapter_performance': get_chapter_performance(),
            'model_performance': get_model_performance_metrics(),
            'rl_performance': get_rl_performance_stats(),
            'inspector_performance': get_inspector_performance(),
            'fraud_patterns': get_fraud_patterns_analysis(),
            'drift_detection': get_drift_detection_results(),
            'feature_importance': get_feature_importance_analysis(),
            'tables_info': get_inspectia_tables_info(),
            # Nouvelles analyses pour toutes les tables
            'advanced_policies': get_advanced_policies_analysis(),
            'model_thresholds': get_model_thresholds_analysis(),
            'chapter_features': get_chapter_features_analysis(),
            'declaration_features': get_declaration_features_analysis(),
            # Métriques avancées RL
            'advanced_decisions_count': execute_postgresql_query("SELECT COUNT(*) as count FROM advanced_decisions")[0][0] if execute_postgresql_query("SELECT COUNT(*) as count FROM advanced_decisions") else 0,
            'advanced_feedbacks_count': execute_postgresql_query("SELECT COUNT(*) as count FROM advanced_feedbacks")[0][0] if execute_postgresql_query("SELECT COUNT(*) as count FROM advanced_feedbacks") else 0,
            'rl_metrics_count': execute_postgresql_query("SELECT COUNT(*) as count FROM rl_performance_metrics")[0][0] if execute_postgresql_query("SELECT COUNT(*) as count FROM rl_performance_metrics") else 0,
            'pv_count': execute_postgresql_query("SELECT COUNT(*) as count FROM pv_inspection")[0][0] if execute_postgresql_query("SELECT COUNT(*) as count FROM pv_inspection") else 0,
            # Health score
            'health_score': get_database_health_score(),
            'system_alerts': get_all_alerts()
        }
        
        return health_metrics
    except Exception as e:
        logger.error(f"❌ Erreur métriques santé: {e}")
        return {'error': str(e)}

def get_daily_processing_stats(days: int = 7):
    """Statistiques de traitement quotidien - adaptées à la structure réelle"""
    try:
        query = f"""
        SELECT 
            DATE(d.created_at) as processing_date,
            COUNT(*) as declarations_processed,
            COUNT(CASE WHEN d.composite_fraud_score > 0.5 THEN 1 END) as high_risk_detected,
            AVG(d.composite_fraud_score) as avg_fraud_score,
            COUNT(DISTINCT d.chapter_id) as chapters_processed,
            AVG(d.valeur_caf) as avg_valeur_caf,
            SUM(d.valeur_caf) as total_valeur_caf,
            -- Statistiques RL
            COUNT(ad.decision_id) as rl_decisions_taken,
            COUNT(CASE WHEN ad.exploration = true THEN 1 END) as exploration_decisions,
            COUNT(CASE WHEN ad.exploration = false THEN 1 END) as exploitation_decisions,
            AVG(ad.confidence_score) as avg_rl_confidence,
            -- Statistiques feedbacks
            COUNT(af.feedback_id) as feedbacks_received,
            AVG(af.feedback_quality_score) as avg_feedback_quality,
            COUNT(CASE WHEN af.inspector_decision = 1 THEN 1 END) as fraud_feedbacks,
            COUNT(CASE WHEN af.inspector_decision = 0 THEN 1 END) as clean_feedbacks,
            -- Statistiques PVs
            COUNT(pv.id) as pvs_generated,
            AVG(pv.score_risque_global) as avg_pv_risk_score,
            SUM(pv.nombre_declarations_analysees) as total_declarations_analyzed_in_pvs
        FROM declarations d
        LEFT JOIN advanced_decisions ad ON d.declaration_id = ad.declaration_id AND DATE(ad.created_at) = DATE(d.created_at)
        LEFT JOIN advanced_feedbacks af ON d.declaration_id = af.declaration_id AND DATE(af.created_at) = DATE(d.created_at)
        LEFT JOIN pv_inspection pv ON DATE(pv.date_generation) = DATE(d.created_at)
        WHERE d.created_at >= CURRENT_DATE - INTERVAL '{days} days'
        GROUP BY DATE(d.created_at)
        ORDER BY processing_date DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats quotidiennes: {e}")
        return []

def get_advanced_policies_analysis():
    """Analyse des politiques avancées INSPECT_IA"""
    try:
        query = """
        SELECT 
            ap.k as policy_key,
            ap.chapter_id,
            c.chapter_name,
            ap.json_blob,
            ap.version,
            ap.created_at,
            ap.updated_at,
            -- Analyse de l'utilisation des politiques
            COUNT(ad.decision_id) as decisions_using_policy,
            AVG(ad.confidence_score) as avg_confidence_with_policy,
            COUNT(CASE WHEN ad.action = 'inspect' THEN 1 END) as inspect_decisions,
            COUNT(CASE WHEN ad.action = 'flag' THEN 1 END) as flag_decisions,
            COUNT(CASE WHEN ad.action = 'clear' THEN 1 END) as clear_decisions,
            COUNT(CASE WHEN ad.action = 'pass' THEN 1 END) as pass_decisions
        FROM advanced_policies ap
        LEFT JOIN chapters c ON ap.chapter_id = c.chapter_id
        LEFT JOIN advanced_decisions ad ON ap.chapter_id = ad.chapter_id
        GROUP BY ap.k, ap.chapter_id, c.chapter_name, ap.json_blob, ap.version, ap.created_at, ap.updated_at
        ORDER BY decisions_using_policy DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur analyse politiques avancées: {e}")
        return []

def get_model_thresholds_analysis():
    """Analyse des seuils de modèles INSPECT_IA"""
    try:
        query = """
        SELECT 
            mt.threshold_id,
            mt.model_id,
            m.model_name,
            mt.chapter_id,
            c.chapter_name,
            mt.optimal_threshold,
            mt.conforme_threshold,
            mt.fraude_threshold,
            mt.zone_grise_min,
            mt.zone_grise_max,
            mt.threshold_method,
            mt.validation_metrics,
            mt.threshold_confidence,
            mt.is_active,
            mt.created_at,
            mt.updated_at,
            -- Analyse de l'impact des seuils
            COUNT(p.prediction_id) as predictions_with_threshold,
            AVG(p.fraud_probability) as avg_fraud_probability,
            COUNT(CASE WHEN p.predicted_fraud = true THEN 1 END) as fraud_predictions,
            COUNT(CASE WHEN p.predicted_fraud = false THEN 1 END) as clean_predictions,
            AVG(p.confidence_score) as avg_confidence,
            -- Analyse des feedbacks pour évaluer la performance
            COUNT(af.feedback_id) as feedbacks_received,
            AVG(af.feedback_quality_score) as avg_feedback_quality,
            COUNT(CASE WHEN af.inspector_decision = 1 AND p.predicted_fraud = true THEN 1 END) as true_positives,
            COUNT(CASE WHEN af.inspector_decision = 0 AND p.predicted_fraud = false THEN 1 END) as true_negatives,
            COUNT(CASE WHEN af.inspector_decision = 1 AND p.predicted_fraud = false THEN 1 END) as false_positives,
            COUNT(CASE WHEN af.inspector_decision = 0 AND p.predicted_fraud = true THEN 1 END) as false_negatives
        FROM model_thresholds mt
        LEFT JOIN models m ON mt.model_id = m.model_id
        LEFT JOIN chapters c ON mt.chapter_id = c.chapter_id
        LEFT JOIN predictions p ON mt.model_id = p.model_id
        LEFT JOIN advanced_feedbacks af ON p.declaration_id = af.declaration_id
        GROUP BY mt.threshold_id, mt.model_id, m.model_name, mt.chapter_id, c.chapter_name,
                 mt.optimal_threshold, mt.conforme_threshold, mt.fraude_threshold, mt.zone_grise_min,
                 mt.zone_grise_max, mt.threshold_method, mt.validation_metrics, mt.threshold_confidence,
                 mt.is_active, mt.created_at, mt.updated_at
        ORDER BY predictions_with_threshold DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur analyse seuils modèles: {e}")
        return []

def get_chapter_features_analysis():
    """Analyse des features par chapitre INSPECT_IA"""
    try:
        query = """
        SELECT 
            cf.chapter_id,
            c.chapter_name,
            c.chapter_number,
            cf.feature_id,
            f.feature_name,
            f.feature_type,
            f.feature_category,
            f.description,
            cf.is_used,
            cf.importance_score,
            -- Analyse de l'utilisation des features
            COUNT(df.feature_id) as usage_count,
            AVG(CASE 
                WHEN f.feature_name = 'VALEUR_CAF' THEN df.valeur_caf
                WHEN f.feature_name = 'POIDS_NET_KG' THEN df.poids_net_kg
                WHEN f.feature_name = 'COMPOSITE_FRAUD_SCORE' THEN df.composite_fraud_score
                WHEN f.feature_name = 'BIENAYME_CHEBYCHEV_SCORE' THEN df.bienayme_chebychev_score
                WHEN f.feature_name = 'MIRROR_TEI_SCORE' THEN df.mirror_tei_score
                WHEN f.feature_name = 'SPECTRAL_CLUSTER_SCORE' THEN df.spectral_cluster_score
                WHEN f.feature_name = 'HIERARCHICAL_CLUSTER_SCORE' THEN df.hierarchical_cluster_score
                WHEN f.feature_name = 'ADMIN_VALUES_SCORE' THEN df.admin_values_score
                ELSE NULL
            END) as avg_feature_value,
            -- Analyse de la corrélation avec les prédictions
            COUNT(p.prediction_id) as predictions_count,
            AVG(p.fraud_probability) as avg_fraud_probability,
            AVG(p.confidence_score) as avg_prediction_confidence,
            -- Analyse de la corrélation avec les décisions RL
            COUNT(ad.decision_id) as rl_decisions_count,
            AVG(ad.confidence_score) as avg_rl_confidence,
            COUNT(CASE WHEN ad.exploration = true THEN 1 END) as exploration_count,
            COUNT(CASE WHEN ad.exploration = false THEN 1 END) as exploitation_count
        FROM chapter_features cf
        LEFT JOIN chapters c ON cf.chapter_id = c.chapter_id
        LEFT JOIN features f ON cf.feature_id = f.feature_id
        LEFT JOIN declaration_features df ON cf.feature_id = df.feature_id AND cf.chapter_id = df.chapter_id
        LEFT JOIN predictions p ON df.declaration_id = p.declaration_id
        LEFT JOIN advanced_decisions ad ON df.declaration_id = ad.declaration_id
        GROUP BY cf.chapter_id, c.chapter_name, c.chapter_number, cf.feature_id, f.feature_name,
                 f.feature_type, f.feature_category, f.description, cf.is_used, cf.importance_score
        ORDER BY cf.importance_score DESC NULLS LAST, usage_count DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur analyse features par chapitre: {e}")
        return []

def get_declaration_features_analysis():
    """Analyse complète des features de déclarations INSPECT_IA"""
    try:
        query = """
        SELECT 
            df.chapter_id,
            c.chapter_name,
            COUNT(*) as total_declarations,
            -- Analyse des scores de fraude
            AVG(df.composite_fraud_score) as avg_composite_score,
            AVG(df.bienayme_chebychev_score) as avg_chebychev_score,
            AVG(df.mirror_tei_score) as avg_tei_score,
            AVG(df.spectral_cluster_score) as avg_spectral_score,
            AVG(df.hierarchical_cluster_score) as avg_hierarchical_score,
            AVG(df.admin_values_score) as avg_admin_score,
            -- Analyse des features business
            COUNT(CASE WHEN df.business_glissement_cosmetique = true THEN 1 END) as cosmetic_shift_count,
            COUNT(CASE WHEN df.business_glissement_machine = true THEN 1 END) as machine_shift_count,
            COUNT(CASE WHEN df.business_glissement_electronique = true THEN 1 END) as electronic_shift_count,
            COUNT(CASE WHEN df.business_is_medicament = true THEN 1 END) as medicament_count,
            COUNT(CASE WHEN df.business_is_machine = true THEN 1 END) as machine_count,
            COUNT(CASE WHEN df.business_is_electronique = true THEN 1 END) as electronic_count,
            -- Analyse des valeurs
            AVG(df.valeur_caf) as avg_valeur_caf,
            AVG(df.poids_net_kg) as avg_poids_net,
            MIN(df.valeur_caf) as min_valeur_caf,
            MAX(df.valeur_caf) as max_valeur_caf,
            MIN(df.poids_net_kg) as min_poids_net,
            MAX(df.poids_net_kg) as max_poids_net,
            -- Analyse temporelle
            MIN(df.created_at) as earliest_declaration,
            MAX(df.created_at) as latest_declaration,
            COUNT(DISTINCT DATE(df.created_at)) as unique_days
        FROM declaration_features df
        LEFT JOIN chapters c ON df.chapter_id = c.chapter_id
        GROUP BY df.chapter_id, c.chapter_name
        ORDER BY total_declarations DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur analyse features déclarations: {e}")
        return []

def get_monthly_trends():
    """Tendances mensuelles - adaptées à la structure réelle INSPECT_IA"""
    try:
        query = """
        SELECT 
            EXTRACT(YEAR FROM d.created_at) as year,
            EXTRACT(MONTH FROM d.created_at) as month,
            COUNT(*) as total_declarations,
            COUNT(CASE WHEN d.composite_fraud_score > 0.5 THEN 1 END) as high_risk_count,
            AVG(d.composite_fraud_score) as avg_fraud_score,
            COUNT(DISTINCT d.chapter_id) as chapters_used,
            AVG(d.valeur_caf) as avg_valeur_caf,
            SUM(d.valeur_caf) as total_valeur_caf,
            -- Tendances RL
            COUNT(ad.decision_id) as total_rl_decisions,
            COUNT(CASE WHEN ad.exploration = true THEN 1 END) as exploration_trend,
            COUNT(CASE WHEN ad.exploration = false THEN 1 END) as exploitation_trend,
            AVG(ad.confidence_score) as avg_rl_confidence_trend,
            AVG(ad.model_proba) as avg_model_probability_trend,
            AVG(ad.rl_proba) as avg_rl_probability_trend,
            -- Tendances feedbacks
            COUNT(af.feedback_id) as total_feedbacks,
            AVG(af.feedback_quality_score) as avg_feedback_quality_trend,
            COUNT(CASE WHEN af.inspector_decision = 1 THEN 1 END) as fraud_feedback_trend,
            COUNT(CASE WHEN af.inspector_decision = 0 THEN 1 END) as clean_feedback_trend,
            COUNT(DISTINCT af.inspector_id) as unique_inspectors,
            -- Tendances PVs
            COUNT(pv.id) as pvs_generated,
            AVG(pv.score_risque_global) as avg_pv_risk_score_trend,
            SUM(pv.nombre_declarations_analysees) as total_analyzed_in_pvs,
            SUM(pv.nombre_fraudes_detectees) as total_frauds_detected_in_pvs,
            AVG(pv.taux_fraude) as avg_fraud_rate_in_pvs
        FROM declarations d
        LEFT JOIN advanced_decisions ad ON d.declaration_id = ad.declaration_id 
            AND EXTRACT(YEAR FROM ad.created_at) = EXTRACT(YEAR FROM d.created_at)
            AND EXTRACT(MONTH FROM ad.created_at) = EXTRACT(MONTH FROM d.created_at)
        LEFT JOIN advanced_feedbacks af ON d.declaration_id = af.declaration_id 
            AND EXTRACT(YEAR FROM af.created_at) = EXTRACT(YEAR FROM d.created_at)
            AND EXTRACT(MONTH FROM af.created_at) = EXTRACT(MONTH FROM d.created_at)
        LEFT JOIN pv_inspection pv ON EXTRACT(YEAR FROM pv.date_generation) = EXTRACT(YEAR FROM d.created_at)
            AND EXTRACT(MONTH FROM pv.date_generation) = EXTRACT(MONTH FROM d.created_at)
        WHERE d.created_at >= CURRENT_DATE - INTERVAL '12 months'
        GROUP BY EXTRACT(YEAR FROM d.created_at), EXTRACT(MONTH FROM d.created_at)
        ORDER BY year DESC, month DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur tendances mensuelles: {e}")
        return []

def get_risk_assessment_summary():
    """Résumé de l'évaluation des risques - adapté à la structure réelle INSPECT_IA"""
    try:
        query = """
        SELECT 
            CASE 
                WHEN d.composite_fraud_score > 0.8 THEN 'CRITICAL'
                WHEN d.composite_fraud_score > 0.6 THEN 'HIGH'
                WHEN d.composite_fraud_score > 0.4 THEN 'MEDIUM'
                ELSE 'LOW'
            END as risk_level,
            COUNT(*) as count,
            AVG(d.composite_fraud_score) as avg_risk_score,
            AVG(d.bienayme_chebychev_score) as avg_chebychev_score,
            AVG(d.mirror_tei_score) as avg_tei_score,
            AVG(d.spectral_cluster_score) as avg_spectral_score,
            AVG(d.hierarchical_cluster_score) as avg_hierarchical_score,
            AVG(d.admin_values_score) as avg_admin_score,
            -- Analyse des décisions RL par niveau de risque
            COUNT(ad.decision_id) as rl_decisions,
            AVG(ad.confidence_score) as avg_rl_confidence,
            COUNT(CASE WHEN ad.action = 'inspect' THEN 1 END) as inspect_decisions,
            COUNT(CASE WHEN ad.action = 'flag' THEN 1 END) as flag_decisions,
            COUNT(CASE WHEN ad.action = 'clear' THEN 1 END) as clear_decisions,
            COUNT(CASE WHEN ad.action = 'pass' THEN 1 END) as pass_decisions,
            -- Analyse des feedbacks par niveau de risque
            COUNT(af.feedback_id) as feedbacks,
            AVG(af.feedback_quality_score) as avg_feedback_quality,
            COUNT(CASE WHEN af.inspector_decision = 1 THEN 1 END) as fraud_feedbacks,
            COUNT(CASE WHEN af.inspector_decision = 0 THEN 1 END) as clean_feedbacks,
            -- Analyse des prédictions par niveau de risque
            COUNT(p.prediction_id) as predictions,
            AVG(p.fraud_probability) as avg_predicted_probability,
            AVG(p.confidence_score) as avg_prediction_confidence,
            COUNT(CASE WHEN p.predicted_fraud = true THEN 1 END) as predicted_fraud,
            COUNT(CASE WHEN p.predicted_fraud = false THEN 1 END) as predicted_clean
        FROM declarations d
        LEFT JOIN advanced_decisions ad ON d.declaration_id = ad.declaration_id
        LEFT JOIN advanced_feedbacks af ON d.declaration_id = af.declaration_id
        LEFT JOIN predictions p ON d.declaration_id = p.declaration_id
        GROUP BY CASE 
            WHEN d.composite_fraud_score > 0.8 THEN 'CRITICAL'
            WHEN d.composite_fraud_score > 0.6 THEN 'HIGH'
            WHEN d.composite_fraud_score > 0.4 THEN 'MEDIUM'
            ELSE 'LOW'
        END
        ORDER BY avg_risk_score DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur résumé risques: {e}")
        return []

def get_feedback_quality_metrics():
    """Métriques de qualité du feedback - adaptées à la structure réelle INSPECT_IA"""
    try:
        query = """
        SELECT 
            af.feedback_category,
            af.inspector_expertise_level,
            COUNT(*) as feedback_count,
            AVG(af.inspector_confidence) as avg_confidence,
            AVG(af.feedback_quality_score) as avg_quality_score,
            AVG(af.review_time_seconds) as avg_review_time_seconds,
            AVG(af.review_time_seconds / 60.0) as avg_review_time_minutes,
            COUNT(CASE WHEN af.inspector_decision = 1 THEN 1 END) as fraud_decisions,
            COUNT(CASE WHEN af.inspector_decision = 0 THEN 1 END) as clean_decisions,
            -- Analyse par chapitre
            af.chapter_id,
            c.chapter_name,
            -- Analyse de la cohérence avec les prédictions
            COUNT(p.prediction_id) as predictions_compared,
            COUNT(CASE WHEN af.inspector_decision = 1 AND p.predicted_fraud = true THEN 1 END) as correct_fraud_detections,
            COUNT(CASE WHEN af.inspector_decision = 0 AND p.predicted_fraud = false THEN 1 END) as correct_clean_detections,
            COUNT(CASE WHEN af.inspector_decision = 1 AND p.predicted_fraud = false THEN 1 END) as false_positives,
            COUNT(CASE WHEN af.inspector_decision = 0 AND p.predicted_fraud = true THEN 1 END) as false_negatives,
            -- Calcul de la précision
            ROUND(
                (COUNT(CASE WHEN af.inspector_decision = 1 AND p.predicted_fraud = true THEN 1 END) + 
                 COUNT(CASE WHEN af.inspector_decision = 0 AND p.predicted_fraud = false THEN 1 END))::numeric / 
                NULLIF(COUNT(p.prediction_id), 0) * 100, 
                2
            ) as accuracy_percentage,
            -- Analyse des notes
            COUNT(CASE WHEN af.notes IS NOT NULL AND af.notes != '' THEN 1 END) as feedbacks_with_notes,
            COUNT(CASE WHEN af.notes IS NULL OR af.notes = '' THEN 1 END) as feedbacks_without_notes,
            -- Analyse temporelle
            MIN(af.created_at) as earliest_feedback,
            MAX(af.created_at) as latest_feedback,
            COUNT(DISTINCT af.inspector_id) as unique_inspectors
        FROM advanced_feedbacks af
        LEFT JOIN chapters c ON af.chapter_id = c.chapter_id
        LEFT JOIN predictions p ON af.declaration_id = p.declaration_id
        GROUP BY af.feedback_category, af.inspector_expertise_level, af.chapter_id, c.chapter_name
        ORDER BY feedback_count DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur métriques feedback: {e}")
        return []

def get_model_retraining_history():
    """Historique des réentraînements de modèles - adapté à la structure réelle INSPECT_IA"""
    try:
        query = """
        SELECT 
            m.model_name,
            m.chapter_id,
            c.chapter_name,
            m.model_type,
            m.version,
            m.training_date,
            m.is_best_model,
            m.is_active,
            m.performance_metrics,
            m.hyperparameters,
            m.feature_list,
            m.created_at,
            m.updated_at,
            -- Analyse des métriques de performance
            pm.f1_score,
            pm.auc_score,
            pm.precision_score,
            pm.recall_score,
            pm.accuracy,
            pm.specificity,
            pm.sensitivity,
            pm.matthews_correlation,
            pm.cohen_kappa,
            pm.evaluation_date,
            pm.dataset_size,
            pm.evaluation_method,
            -- Analyse des feedbacks pour évaluer la performance réelle
            COUNT(af.feedback_id) as total_feedbacks,
            AVG(af.feedback_quality_score) as avg_feedback_quality,
            COUNT(CASE WHEN af.inspector_decision = 1 AND p.predicted_fraud = true THEN 1 END) as true_positives,
            COUNT(CASE WHEN af.inspector_decision = 0 AND p.predicted_fraud = false THEN 1 END) as true_negatives,
            COUNT(CASE WHEN af.inspector_decision = 1 AND p.predicted_fraud = false THEN 1 END) as false_positives,
            COUNT(CASE WHEN af.inspector_decision = 0 AND p.predicted_fraud = true THEN 1 END) as false_negatives,
            -- Calcul de la précision réelle basée sur les feedbacks
            ROUND(
                (COUNT(CASE WHEN af.inspector_decision = 1 AND p.predicted_fraud = true THEN 1 END) + 
                 COUNT(CASE WHEN af.inspector_decision = 0 AND p.predicted_fraud = false THEN 1 END))::numeric / 
                NULLIF(COUNT(af.feedback_id), 0) * 100, 
                2
            ) as real_world_accuracy,
            -- Analyse des décisions RL
            COUNT(ad.decision_id) as rl_decisions,
            AVG(ad.confidence_score) as avg_rl_confidence,
            COUNT(CASE WHEN ad.exploration = true THEN 1 END) as exploration_decisions,
            COUNT(CASE WHEN ad.exploration = false THEN 1 END) as exploitation_decisions
        FROM models m
        LEFT JOIN chapters c ON m.chapter_id = c.chapter_id
        LEFT JOIN performance_metrics pm ON m.model_id = pm.model_id
        LEFT JOIN predictions p ON m.model_id = p.model_id
        LEFT JOIN advanced_feedbacks af ON p.declaration_id = af.declaration_id
        LEFT JOIN advanced_decisions ad ON p.declaration_id = ad.declaration_id
        GROUP BY m.model_id, m.model_name, m.chapter_id, c.chapter_name, m.model_type, m.version,
                 m.training_date, m.is_best_model, m.is_active, m.performance_metrics, m.hyperparameters,
                 m.feature_list, m.created_at, m.updated_at, pm.f1_score, pm.auc_score, pm.precision_score,
                 pm.recall_score, pm.accuracy, pm.specificity, pm.sensitivity, pm.matthews_correlation,
                 pm.cohen_kappa, pm.evaluation_date, pm.dataset_size, pm.evaluation_method
        ORDER BY m.training_date DESC, m.updated_at DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur historique réentraînement: {e}")
        return []

def get_feature_drift_analysis():
    """Analyse du drift des features - adaptée à la structure réelle INSPECT_IA"""
    try:
        query = """
        SELECT 
            f.feature_name,
            f.feature_type,
            f.feature_category,
            cf.chapter_id,
            c.chapter_name,
            -- Analyse temporelle des features pour détecter le drift
            MIN(df.created_at) as earliest_usage,
            MAX(df.created_at) as latest_usage,
            COUNT(*) as usage_count,
            -- Analyse des valeurs moyennes par période
            AVG(CASE WHEN df.created_at >= CURRENT_DATE - INTERVAL '30 days' THEN df.composite_fraud_score END) as avg_score_last_30_days,
            AVG(CASE WHEN df.created_at < CURRENT_DATE - INTERVAL '30 days' AND df.created_at >= CURRENT_DATE - INTERVAL '60 days' THEN df.composite_fraud_score END) as avg_score_previous_30_days,
            -- Classification du drift basique
            CASE 
                WHEN ABS(AVG(CASE WHEN df.created_at >= CURRENT_DATE - INTERVAL '30 days' THEN df.composite_fraud_score END) - 
                         AVG(CASE WHEN df.created_at < CURRENT_DATE - INTERVAL '30 days' AND df.created_at >= CURRENT_DATE - INTERVAL '60 days' THEN df.composite_fraud_score END)) > 0.2 THEN 'HIGH'
                WHEN ABS(AVG(CASE WHEN df.created_at >= CURRENT_DATE - INTERVAL '30 days' THEN df.composite_fraud_score END) - 
                         AVG(CASE WHEN df.created_at < CURRENT_DATE - INTERVAL '30 days' AND df.created_at >= CURRENT_DATE - INTERVAL '60 days' THEN df.composite_fraud_score END)) > 0.1 THEN 'MEDIUM'
                ELSE 'LOW'
            END as drift_severity
        FROM features f
        LEFT JOIN chapter_features cf ON f.feature_id = cf.feature_id
        LEFT JOIN chapters c ON cf.chapter_id = c.chapter_id
        LEFT JOIN declaration_features df ON f.feature_id = df.feature_id
        WHERE df.created_at IS NOT NULL
        GROUP BY f.feature_id, f.feature_name, f.feature_type, f.feature_category, cf.chapter_id, c.chapter_name
        HAVING COUNT(*) > 10
        ORDER BY usage_count DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur analyse drift features: {e}")
        return []

def get_performance_benchmarks():
    """Benchmarks de performance du système - adaptés à la structure réelle INSPECT_IA"""
    try:
        query = """
        SELECT 
            'FRAUD_DETECTION_ACCURACY' as metric_name,
            c.chapter_id,
            c.chapter_name,
            -- Performance actuelle basée sur les feedbacks
            ROUND(
                (COUNT(CASE WHEN af.inspector_decision = 1 AND p.predicted_fraud = true THEN 1 END) + 
                 COUNT(CASE WHEN af.inspector_decision = 0 AND p.predicted_fraud = false THEN 1 END))::numeric / 
                NULLIF(COUNT(af.feedback_id), 0) * 100, 
                2
            ) as current_value,
            85.0 as benchmark_value,
            90.0 as target_value,
            CASE 
                WHEN ROUND(
                    (COUNT(CASE WHEN af.inspector_decision = 1 AND p.predicted_fraud = true THEN 1 END) + 
                     COUNT(CASE WHEN af.inspector_decision = 0 AND p.predicted_fraud = false THEN 1 END))::numeric / 
                    NULLIF(COUNT(af.feedback_id), 0) * 100, 
                    2
                ) >= 90.0 THEN 'EXCELLENT'
                WHEN ROUND(
                    (COUNT(CASE WHEN af.inspector_decision = 1 AND p.predicted_fraud = true THEN 1 END) + 
                     COUNT(CASE WHEN af.inspector_decision = 0 AND p.predicted_fraud = false THEN 1 END))::numeric / 
                    NULLIF(COUNT(af.feedback_id), 0) * 100, 
                    2
                ) >= 85.0 THEN 'GOOD'
                WHEN ROUND(
                    (COUNT(CASE WHEN af.inspector_decision = 1 AND p.predicted_fraud = true THEN 1 END) + 
                     COUNT(CASE WHEN af.inspector_decision = 0 AND p.predicted_fraud = false THEN 1 END))::numeric / 
                    NULLIF(COUNT(af.feedback_id), 0) * 100, 
                    2
                ) >= 68.0 THEN 'ACCEPTABLE'
                ELSE 'NEEDS_IMPROVEMENT'
            END as performance_level,
            MAX(af.created_at) as last_updated,
            COUNT(af.feedback_id) as sample_size
        FROM chapters c
        LEFT JOIN predictions p ON c.chapter_id = p.chapter_id
        LEFT JOIN advanced_feedbacks af ON p.declaration_id = af.declaration_id
        WHERE af.feedback_id IS NOT NULL
        GROUP BY c.chapter_id, c.chapter_name
        
        UNION ALL
        
        SELECT 
            'RL_CONFIDENCE_SCORE' as metric_name,
            c.chapter_id,
            c.chapter_name,
            ROUND(AVG(ad.confidence_score), 2) as current_value,
            0.7 as benchmark_value,
            0.8 as target_value,
            CASE 
                WHEN AVG(ad.confidence_score) >= 0.8 THEN 'EXCELLENT'
                WHEN AVG(ad.confidence_score) >= 0.7 THEN 'GOOD'
                WHEN AVG(ad.confidence_score) >= 0.56 THEN 'ACCEPTABLE'
                ELSE 'NEEDS_IMPROVEMENT'
            END as performance_level,
            MAX(ad.created_at) as last_updated,
            COUNT(ad.decision_id) as sample_size
        FROM chapters c
        LEFT JOIN advanced_decisions ad ON c.chapter_id = ad.chapter_id
        WHERE ad.decision_id IS NOT NULL
        GROUP BY c.chapter_id, c.chapter_name
        
        UNION ALL
        
        SELECT 
            'FEEDBACK_QUALITY' as metric_name,
            c.chapter_id,
            c.chapter_name,
            ROUND(AVG(af.feedback_quality_score), 2) as current_value,
            3.0 as benchmark_value,
            4.0 as target_value,
            CASE 
                WHEN AVG(af.feedback_quality_score) >= 4.0 THEN 'EXCELLENT'
                WHEN AVG(af.feedback_quality_score) >= 3.0 THEN 'GOOD'
                WHEN AVG(af.feedback_quality_score) >= 2.4 THEN 'ACCEPTABLE'
                ELSE 'NEEDS_IMPROVEMENT'
            END as performance_level,
            MAX(af.created_at) as last_updated,
            COUNT(af.feedback_id) as sample_size
        FROM chapters c
        LEFT JOIN advanced_feedbacks af ON c.chapter_id = af.chapter_id
        WHERE af.feedback_id IS NOT NULL
        GROUP BY c.chapter_id, c.chapter_name
        
        ORDER BY metric_name, chapter_id
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur benchmarks: {e}")
        return []

def generate_inspectia_comprehensive_report():
    """Génère un rapport complet du système INSPECT_IA"""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': get_system_health_metrics(),
            'daily_stats': get_daily_processing_stats(),
            'monthly_trends': get_monthly_trends(),
            'risk_assessment': get_risk_assessment_summary(),
            'feedback_quality': get_feedback_quality_metrics(),
            'model_retraining': get_model_retraining_history(),
            'feature_drift': get_feature_drift_analysis(),
            'performance_benchmarks': get_performance_benchmarks(),
            'database_health': get_database_health_check(),
            'system_performance': get_database_performance()
        }
        
        # Sauvegarder le rapport
        report_file = f"inspectia_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Rapport complet INSPECT_IA généré: {report_file}")
        return report
                
    except Exception as e:
        logger.error(f"❌ Erreur rapport complet: {e}")
        return {'error': str(e)}

# Fonctions de maintenance spécialisées INSPECT_IA
def cleanup_old_declarations(days: int = 365):
    """Nettoie les anciennes déclarations"""
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with get_database_session_context() as session:
            # Supprimer les anciennes déclarations et leurs données associées
            session.execute(text("""
                DELETE FROM feedback_history 
                WHERE created_at < :cutoff_date
            """), {'cutoff_date': cutoff_date})
            
            session.execute(text("""
                DELETE FROM analysis_results 
                WHERE created_at < :cutoff_date
            """), {'cutoff_date': cutoff_date})
            
            session.execute(text("""
                DELETE FROM rl_decisions 
                WHERE created_at < :cutoff_date
            """), {'cutoff_date': cutoff_date})
            
            session.execute(text("""
                DELETE FROM predictions 
                WHERE created_at < :cutoff_date
            """), {'cutoff_date': cutoff_date})
            
            session.execute(text("""
                DELETE FROM declarations 
                WHERE created_at < :cutoff_date
            """), {'cutoff_date': cutoff_date})
            
            session.commit()
        
        logger.info(f"✅ Nettoyage des déclarations de plus de {days} jours terminé")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage déclarations: {e}")
        return False

def optimize_inspectia_database():
    """Optimise spécifiquement la base de données INSPECT_IA"""
    try:
        with get_database_session_context() as session:
            # VACUUM ANALYZE sur toutes les tables INSPECT_IA
            inspectia_tables = [
                'advanced_decisions', 'advanced_feedbacks', 'advanced_policies',
                'analysis_results', 'chapter_features', 'chapters',
                'declaration_features', 'declarations', 'features',
                'feedback_history', 'inspector_profiles', 'model_thresholds',
                'models', 'performance_metrics', 'predictions',
                'pv_inspection', 'pvs', 'rl_bandits', 'rl_decisions',
                'rl_performance_metrics', 'system_logs'
            ]
            
            for table in inspectia_tables:
                try:
                    session.execute(text(f"VACUUM ANALYZE {table}"))
                    logger.info(f"✅ Table {table} optimisée")
                except Exception as e:
                    logger.warning(f"⚠️ Impossible d'optimiser {table}: {e}")
            
            # REINDEX sur les index critiques
            session.execute(text("REINDEX DATABASE " + db_config['database']))
            
        logger.info("✅ Optimisation base de données INSPECT_IA terminée")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur optimisation INSPECT_IA: {e}")
        return False

def backup_inspectia_data(backup_path: str = None):
    """Sauvegarde spécifique des données INSPECT_IA"""
    try:
        if backup_path is None:
            backup_path = f"inspectia_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        
        # Tables critiques à sauvegarder
        tables_to_backup = [
            'advanced_decisions', 'advanced_feedbacks', 'advanced_policies',
            'analysis_results', 'chapter_features', 'chapters',
            'declaration_features', 'declarations', 'features',
            'feedback_history', 'inspector_profiles', 'model_thresholds',
            'models', 'performance_metrics', 'predictions',
            'pv_inspection', 'pvs', 'rl_bandits', 'rl_decisions',
            'rl_performance_metrics', 'system_logs'
        ]
        
        tables_str = ' '.join([f'-t {table}' for table in tables_to_backup])
        
        import subprocess
        cmd = [
            'pg_dump',
            f'--host={db_config["host"]}',
            f'--port={db_config["port"]}',
            f'--username={db_config["user"]}',
            f'--dbname={db_config["database"]}',
            f'--file={backup_path}',
            '--verbose'
        ] + tables_to_backup
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Sauvegarde INSPECT_IA créée: {backup_path}")
            return True
        else:
            logger.error(f"❌ Erreur sauvegarde INSPECT_IA: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde INSPECT_IA: {e}")
        return False

# Fonctions de monitoring avancées et d'alertes
def get_system_alerts():
    """Obtient les alertes système actives"""
    try:
        alerts = []
        
        # Vérifier les connexions
        active_conns = get_active_connections()
        if active_conns:
            total_conns = 0
            for conn in active_conns:
                # Gérer les tuples et dictionnaires
                if isinstance(conn, tuple):
                    connection_count = conn[1] if len(conn) > 1 else 0
                else:
                    connection_count = conn.get('connection_count', 0)
                
                # S'assurer que connection_count est un nombre
                try:
                    connection_count = int(connection_count) if connection_count else 0
                except (ValueError, TypeError):
                    connection_count = 0
                
                total_conns += connection_count
            
            if total_conns > 80:  # Seuil d'alerte
                alerts.append({
                    'type': 'warning',
                    'message': f'Trop de connexions actives: {total_conns}',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Vérifier l'espace disque
        try:
            db_size = get_database_size()
            if db_size and 'GB' in str(db_size) and float(str(db_size).split()[0]) > 10:
                alerts.append({
                    'type': 'warning',
                    'message': f'Base de données volumineuse: {db_size}',
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            logger.warning(f"Impossible de vérifier la taille de la base de données: {e}")
        
        # Vérifier les verrous
        locks = get_database_locks()
        if locks and len(locks) > 50:
            alerts.append({
                'type': 'critical',
                'message': f'Trop de verrous actifs: {len(locks)}',
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    except Exception as e:
        logger.error(f"❌ Erreur alertes système: {e}")
        return []

def get_performance_alerts():
    """Obtient les alertes de performance"""
    try:
        alerts = []
        
        # Vérifier les requêtes lentes
        slow_queries = get_slow_queries(5)
        for query in slow_queries:
            # Gérer les tuples et dictionnaires
            if isinstance(query, tuple):
                mean_exec_time = query[1] if len(query) > 1 else 0
            else:
                mean_exec_time = query.get('mean_exec_time', 0)
            
            if mean_exec_time > 1000:  # Plus de 1 seconde
                alerts.append({
                    'type': 'warning',
                    'message': f'Requête lente détectée: {mean_exec_time}ms',
                    'details': (query[0] if isinstance(query, tuple) else query.get('query', ''))[:100] + '...',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Vérifier la précision des modèles
        try:
            accuracy_data = get_prediction_accuracy()
            for acc in accuracy_data:
                # Gérer les tuples et dictionnaires
                if isinstance(acc, tuple):
                    accuracy_percentage = acc[1] if len(acc) > 1 else 0
                    chapter = acc[0] if len(acc) > 0 else 'Unknown'
                else:
                    accuracy_percentage = acc.get('accuracy_percentage', 0)
                    chapter = acc.get('chapter', 'Unknown')
                
                if accuracy_percentage < 70:
                    alerts.append({
                        'type': 'critical',
                        'message': f'Précision faible pour {chapter}: {accuracy_percentage}%',
                        'timestamp': datetime.now().isoformat()
                    })
        except Exception as e:
            logger.warning(f"Impossible de vérifier la précision: {e}")
        
        return alerts
    except Exception as e:
        logger.error(f"❌ Erreur alertes performance: {e}")
        return []

def get_data_quality_alerts():
    """Obtient les alertes de qualité des données"""
    try:
        alerts = []
        
        # Vérifier les données manquantes
        try:
            query = """
            SELECT 
                c.table_name,
                c.column_name,
                COUNT(*) as null_count
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT 'declarations' as table_name, 
                       CASE WHEN declaration_id IS NULL THEN 'declaration_id' END as column_name
                FROM declarations
                UNION ALL
                SELECT 'declarations' as table_name,
                       CASE WHEN composite_fraud_score IS NULL THEN 'composite_fraud_score' END as column_name
                FROM declarations
            ) null_data ON c.table_name = null_data.table_name AND c.column_name = null_data.column_name
            WHERE c.table_schema = 'public'
            AND null_data.column_name IS NOT NULL
            GROUP BY c.table_name, c.column_name
            """
            null_data = execute_postgresql_query(query)
            if null_data:
                for row in null_data:
                    # Gérer les tuples et dictionnaires
                    if isinstance(row, tuple):
                        table_name = row[0] if len(row) > 0 else 'Unknown'
                        column_name = row[1] if len(row) > 1 else 'Unknown'
                        null_count = row[2] if len(row) > 2 else 0
                    else:
                        table_name = row.get('table_name', 'Unknown')
                        column_name = row.get('column_name', 'Unknown')
                        null_count = row.get('null_count', 0)
                    
                    alerts.append({
                        'type': 'warning',
                        'message': f'Données manquantes dans {table_name}.{column_name}: {null_count}',
                        'timestamp': datetime.now().isoformat()
                    })
        except Exception as e:
            logger.warning(f"Impossible de vérifier les données manquantes: {e}")
        
        return alerts
    except Exception as e:
        logger.error(f"❌ Erreur alertes qualité: {e}")
        return []

def get_security_alerts():
    """Obtient les alertes de sécurité"""
    try:
        alerts = []
        
        # Vérifier les connexions suspectes
        try:
            query = """
            SELECT 
                client_addr,
                application_name,
                state,
                count(*) as connection_count
            FROM pg_stat_activity 
            WHERE datname = current_database()
            GROUP BY client_addr, application_name, state
            HAVING count(*) > 10
            """
            suspicious_conns = execute_postgresql_query(query)
            if suspicious_conns:
                for conn in suspicious_conns:
                    alerts.append({
                        'type': 'warning',
                        'message': f'Connexions multiples depuis {conn["client_addr"]}: {conn["connection_count"]}',
                        'timestamp': datetime.now().isoformat()
                    })
        except Exception as e:
            logger.warning(f"Impossible de vérifier les connexions suspectes: {e}")
        
        # Vérifier les privilèges élevés
        try:
            query = """
            SELECT 
                usename,
                usesuper,
                usecreatedb,
                usebypassrls
            FROM pg_user
            WHERE usesuper = true OR usecreatedb = true OR usebypassrls = true
            """
            privileged_users = execute_postgresql_query(query)
            if privileged_users:
                for user in privileged_users:
                    # Gérer les tuples et dictionnaires
                    if isinstance(user, tuple):
                        usename = user[0] if len(user) > 0 else 'Unknown'
                    else:
                        usename = user.get('usename', 'Unknown')
                    
                    alerts.append({
                        'type': 'info',
                        'message': f'Utilisateur avec privilèges élevés: {usename}',
                        'timestamp': datetime.now().isoformat()
                    })
        except Exception as e:
            logger.warning(f"Impossible de vérifier les privilèges: {e}")
        
        return alerts
    except Exception as e:
        logger.error(f"❌ Erreur alertes sécurité: {e}")
        return []

def get_all_alerts():
    """Obtient toutes les alertes système"""
    try:
        all_alerts = {
            'timestamp': datetime.now().isoformat(),
            'system_alerts': get_system_alerts(),
            'performance_alerts': get_performance_alerts(),
            'data_quality_alerts': get_data_quality_alerts(),
            'security_alerts': get_security_alerts()
        }
        
        # Compter les alertes par type
        total_alerts = 0
        critical_count = 0
        warning_count = 0
        info_count = 0
        
        for alert_category in all_alerts.values():
            if isinstance(alert_category, list):
                for alert in alert_category:
                    total_alerts += 1
                    if alert.get('type') == 'critical':
                        critical_count += 1
                    elif alert.get('type') == 'warning':
                        warning_count += 1
                    elif alert.get('type') == 'info':
                        info_count += 1
        
        all_alerts['summary'] = {
            'total_alerts': total_alerts,
            'critical_count': critical_count,
            'warning_count': warning_count,
            'info_count': info_count
        }
        
        return all_alerts
    except Exception as e:
        logger.error(f"❌ Erreur toutes les alertes: {e}")
        return {'error': str(e)}

# Fonctions de monitoring en temps réel
def get_real_time_metrics():
    """Obtient les métriques en temps réel"""
    try:
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'database_metrics': {
                'active_connections': get_active_connections(),
                'database_size': get_database_size(),
                'active_locks': len(get_database_locks()) if get_database_locks() else 0,
                'slow_queries_count': len(get_slow_queries(10))
            },
            'system_metrics': {
                'uptime': time.time(),
                'memory_usage': 'N/A',  # À implémenter si nécessaire
                'cpu_usage': 'N/A'      # À implémenter si nécessaire
            }
        }
        
        return metrics
    except Exception as e:
        logger.error(f"❌ Erreur métriques temps réel: {e}")
        return {'error': str(e)}

def get_system_load_metrics():
    """Obtient les métriques de charge du système"""
    try:
        query = """
        SELECT 
            COUNT(*) as total_queries,
            AVG(EXTRACT(EPOCH FROM (now() - query_start))) as avg_query_duration,
            COUNT(CASE WHEN state = 'active' THEN 1 END) as active_queries,
            COUNT(CASE WHEN state = 'idle' THEN 1 END) as idle_queries,
            COUNT(CASE WHEN state = 'idle in transaction' THEN 1 END) as idle_in_transaction
        FROM pg_stat_activity 
        WHERE datname = current_database()
        """
        
        load_metrics = execute_postgresql_query(query)
        if load_metrics:
            return load_metrics[0]
        return {}
    except Exception as e:
        logger.error(f"❌ Erreur métriques charge: {e}")
        return {}

def get_query_execution_stats():
    """Statistiques d'exécution des requêtes"""
    try:
        if not is_pg_stat_statements_available():
            return []
        
        query = """
        SELECT 
            query,
            calls,
            total_exec_time,
            mean_exec_time,
            rows,
            shared_blks_hit,
            shared_blks_read,
            shared_blks_written,
            local_blks_hit,
            local_blks_read,
            local_blks_written
        FROM pg_stat_statements 
        ORDER BY total_exec_time DESC 
        LIMIT 50
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats exécution: {e}")
        return []

def get_database_activity_summary():
    """Résumé de l'activité de la base de données"""
    try:
        query = """
        SELECT 
            datname,
            numbackends as active_connections,
            xact_commit as committed_transactions,
            xact_rollback as rolled_back_transactions,
            blks_read as blocks_read,
            blks_hit as blocks_hit,
            tup_returned as tuples_returned,
            tup_fetched as tuples_fetched,
            tup_inserted as tuples_inserted,
            tup_updated as tuples_updated,
            tup_deleted as tuples_deleted
        FROM pg_stat_database 
        WHERE datname = current_database()
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur résumé activité: {e}")
        return []

def get_table_activity_stats():
    """Statistiques d'activité des tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            seq_scan as sequential_scans,
            seq_tup_read as sequential_tuples_read,
            idx_scan as index_scans,
            idx_tup_fetch as index_tuples_fetched,
            n_tup_ins as tuples_inserted,
            n_tup_upd as tuples_updated,
            n_tup_del as tuples_deleted,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples,
            last_vacuum,
            last_autovacuum,
            last_analyze,
            last_autoanalyze
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
        ORDER BY seq_scan + idx_scan DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats activité tables: {e}")
        return []

def get_index_activity_stats():
    """Statistiques d'activité des index"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            indexname,
            idx_scan as index_scans,
            idx_tup_read as tuples_read,
            idx_tup_fetch as tuples_fetched
        FROM pg_stat_user_indexes 
        WHERE schemaname = 'public'
        ORDER BY idx_scan DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats activité index: {e}")
        return []

def get_database_io_stats():
    """Statistiques d'E/S de la base de données"""
    try:
        query = """
        SELECT 
            datname,
            blks_read as blocks_read,
            blks_hit as blocks_hit,
            ROUND(
                (blks_hit::float / (blks_hit + blks_read)) * 100, 
                2
            ) as cache_hit_ratio,
            temp_files,
            temp_bytes,
            stats_reset
        FROM pg_stat_database 
        WHERE datname = current_database()
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats E/S: {e}")
        return []

def get_database_memory_stats():
    """Statistiques mémoire de la base de données"""
    try:
        query = """
        SELECT 
            name,
            setting,
            unit,
            category
        FROM pg_settings 
        WHERE name IN (
            'shared_buffers',
            'effective_cache_size',
            'work_mem',
            'maintenance_work_mem',
            'wal_buffers',
            'checkpoint_completion_target'
        )
        ORDER BY name
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats mémoire: {e}")
        return []

def get_database_checkpoint_stats():
    """Statistiques de checkpoint"""
    try:
        query = """
        SELECT 
            checkpoints_timed,
            checkpoints_req,
            checkpoint_write_time,
            checkpoint_sync_time,
            buffers_checkpoint,
            buffers_clean,
            maxwritten_clean,
            buffers_backend,
            buffers_backend_fsync,
            buffers_alloc,
            stats_reset
        FROM pg_stat_bgwriter
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats checkpoint: {e}")
        return []

def get_database_wal_stats():
    """Statistiques WAL (Write-Ahead Logging)"""
    try:
        query = """
        SELECT 
            wal_records,
            wal_fpi,
            wal_bytes,
            wal_buffers_full,
            wal_write,
            wal_sync,
            wal_write_time,
            wal_sync_time,
            stats_reset
        FROM pg_stat_wal
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats WAL: {e}")
        return []

def get_database_archiver_stats():
    """Statistiques de l'archiviste"""
    try:
        query = """
        SELECT 
            archived_count,
            last_archived_wal,
            last_archived_time,
            failed_count,
            last_failed_wal,
            last_failed_time,
            stats_reset
        FROM pg_stat_archiver
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur stats archiviste: {e}")
        return []

def get_comprehensive_monitoring_report():
    """Rapport de monitoring complet"""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'alerts': get_all_alerts(),
            'real_time_metrics': get_real_time_metrics(),
            'load_metrics': get_system_load_metrics(),
            'query_execution': get_query_execution_stats(),
            'database_activity': get_database_activity_summary(),
            'table_activity': get_table_activity_stats(),
            'index_activity': get_index_activity_stats(),
            'io_stats': get_database_io_stats(),
            'memory_stats': get_database_memory_stats(),
            'checkpoint_stats': get_database_checkpoint_stats(),
            'wal_stats': get_database_wal_stats(),
            'archiver_stats': get_database_archiver_stats()
        }
        
        # Sauvegarder le rapport
        report_file = f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Rapport de monitoring généré: {report_file}")
        return report
        
    except Exception as e:
        logger.error(f"❌ Erreur rapport monitoring: {e}")
        return {'error': str(e)}

# Fonctions de diagnostic avancées
def diagnose_database_issues():
    """Diagnostique les problèmes de base de données"""
    try:
        issues = []
        
        # Vérifier les connexions
        active_conns = get_active_connections()
        if active_conns:
            total_conns = sum([conn['connection_count'] for conn in active_conns])
            if total_conns > 80:
                issues.append({
                    'severity': 'high',
                    'issue': 'Trop de connexions actives',
                    'details': f'{total_conns} connexions actives',
                    'recommendation': 'Augmenter max_connections ou optimiser les connexions'
                })
        
        # Vérifier les requêtes lentes
        slow_queries = get_slow_queries(5)
        for query in slow_queries:
            # Gérer les tuples et dictionnaires
            if isinstance(query, tuple):
                mean_exec_time = query[1] if len(query) > 1 else 0
            else:
                mean_exec_time = query.get('mean_exec_time', 0)
            
            if mean_exec_time > 1000:
                issues.append({
                    'severity': 'medium',
                    'issue': 'Requête lente',
                    'details': f'Temps moyen: {mean_exec_time}ms',
                    'recommendation': 'Optimiser la requête ou ajouter des index'
                })
        
        # Vérifier le cache hit ratio
        io_stats = get_database_io_stats()
        if io_stats:
            cache_hit_ratio = io_stats[0].get('cache_hit_ratio', 0)
            if cache_hit_ratio < 90:
                issues.append({
                    'severity': 'medium',
                    'issue': 'Cache hit ratio faible',
                    'details': f'Ratio: {cache_hit_ratio}%',
                    'recommendation': 'Augmenter shared_buffers ou effective_cache_size'
                })
        
        # Vérifier les verrous
        locks = get_database_locks()
        if locks and len(locks) > 50:
            issues.append({
                'severity': 'high',
                'issue': 'Trop de verrous',
                'details': f'{len(locks)} verrous actifs',
                'recommendation': 'Identifier et résoudre les deadlocks'
            })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'issues_found': len(issues),
            'issues': issues
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur diagnostic: {e}")
        return {'error': str(e)}

def get_database_recommendations():
    """Obtient des recommandations d'optimisation"""
    try:
        recommendations = []
        
        # Analyser les index inutilisés
        index_stats = get_index_activity_stats()
        for index in index_stats:
            if index.get('index_scans', 0) == 0:
                recommendations.append({
                    'type': 'index',
                    'priority': 'low',
                    'recommendation': f'Index inutilisé: {index["indexname"]}',
                    'details': f'Table: {index["tablename"]}',
                    'action': 'Considérer la suppression de cet index'
                })
        
        # Analyser les tables avec beaucoup de scans séquentiels
        table_stats = get_table_activity_stats()
        for table in table_stats:
            seq_scans = table.get('sequential_scans', 0)
            idx_scans = table.get('index_scans', 0)
            if seq_scans > idx_scans * 2 and seq_scans > 100:
                recommendations.append({
                    'type': 'query_optimization',
                    'priority': 'medium',
                    'recommendation': f'Trop de scans séquentiels sur {table["tablename"]}',
                    'details': f'Scans séquentiels: {seq_scans}, Scans d\'index: {idx_scans}',
                    'action': 'Ajouter des index ou optimiser les requêtes'
                })
        
        # Analyser la configuration mémoire
        memory_stats = get_database_memory_stats()
        shared_buffers = None
        for stat in memory_stats:
            if stat['name'] == 'shared_buffers':
                shared_buffers = stat['setting']
                break
        
        if shared_buffers:
            # Recommandation basée sur la taille de la base
            db_size = get_database_size()
            if 'GB' in db_size:
                size_gb = float(db_size.split()[0])
                recommended_buffers = int(size_gb * 0.25 * 1024)  # 25% de la taille en MB
                current_buffers = int(shared_buffers.replace('MB', '').replace('kB', ''))
                
                if current_buffers < recommended_buffers * 0.5:
                    recommendations.append({
                        'type': 'configuration',
                        'priority': 'high',
                        'recommendation': 'shared_buffers trop petit',
                        'details': f'Actuel: {shared_buffers}, Recommandé: {recommended_buffers}MB',
                        'action': 'Augmenter shared_buffers dans postgresql.conf'
                    })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'recommendations_count': len(recommendations),
            'recommendations': recommendations
        }
                
    except Exception as e:
        logger.error(f"❌ Erreur recommandations: {e}")
        return {'error': str(e)}

def get_database_tuning_suggestions():
    """Suggestions de tuning de la base de données"""
    try:
        suggestions = []
        
        # Analyser les paramètres de configuration
        config = get_database_configuration()
        for setting in config:
            name = setting['name']
            current_value = setting['setting']
            unit = setting.get('unit', '')
            
            if name == 'shared_buffers':
                if unit == '8kB':
                    current_mb = int(current_value) * 8 / 1024
                    if current_mb < 256:
                        suggestions.append({
                            'parameter': 'shared_buffers',
                            'current_value': f'{current_mb}MB',
                            'suggested_value': '256MB',
                            'reason': 'Améliorer les performances de cache'
                        })
            
            elif name == 'work_mem':
                if unit == 'kB':
                    current_mb = int(current_value) / 1024
                    if current_mb < 4:
                        suggestions.append({
                            'parameter': 'work_mem',
                            'current_value': f'{current_mb}MB',
                            'suggested_value': '4MB',
                            'reason': 'Améliorer les opérations de tri et jointure'
                        })
            
            elif name == 'maintenance_work_mem':
                if unit == 'kB':
                    current_mb = int(current_value) / 1024
                    if current_mb < 64:
                        suggestions.append({
                            'parameter': 'maintenance_work_mem',
                            'current_value': f'{current_mb}MB',
                            'suggested_value': '64MB',
                            'reason': 'Améliorer les opérations de maintenance'
                        })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'suggestions_count': len(suggestions),
            'suggestions': suggestions
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur suggestions tuning: {e}")
        return {'error': str(e)}

# Fonctions de maintenance avancées et utilitaires
def create_database_snapshot(snapshot_name: str = None):
    """Crée un snapshot de la base de données"""
    try:
        if snapshot_name is None:
            snapshot_name = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Créer un point de sauvegarde
        with get_database_session_context() as session:
            session.execute(text("SAVEPOINT " + snapshot_name))
        
        logger.info(f"✅ Snapshot créé: {snapshot_name}")
        return {'snapshot_name': snapshot_name, 'status': 'created'}
        
    except Exception as e:
        logger.error(f"❌ Erreur création snapshot: {e}")
        return {'error': str(e)}

def rollback_to_snapshot(snapshot_name: str):
    """Revient à un snapshot précédent"""
    try:
        with get_database_session_context() as session:
            session.execute(text("ROLLBACK TO SAVEPOINT " + snapshot_name))
        
        logger.info(f"✅ Rollback vers snapshot: {snapshot_name}")
        return {'status': 'rolled_back', 'snapshot': snapshot_name}
        
    except Exception as e:
        logger.error(f"❌ Erreur rollback: {e}")
        return {'error': str(e)}

def get_database_schema_export():
    """Exporte le schéma de la base de données"""
    try:
        import subprocess
        schema_file = f"schema_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        
        cmd = [
            'pg_dump',
            f'--host={db_config["host"]}',
            f'--port={db_config["port"]}',
            f'--username={db_config["user"]}',
            f'--dbname={db_config["database"]}',
            '--schema-only',
            '--file', schema_file,
            '--verbose'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Export schéma créé: {schema_file}")
            return {'schema_file': schema_file, 'status': 'exported'}
        else:
            logger.error(f"❌ Erreur export schéma: {result.stderr}")
            return {'error': result.stderr}
            
    except Exception as e:
        logger.error(f"❌ Erreur export schéma: {e}")
        return {'error': str(e)}

def get_data_only_export():
    """Exporte uniquement les données"""
    try:
        import subprocess
        data_file = f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
        
        cmd = [
            'pg_dump',
            f'--host={db_config["host"]}',
            f'--port={db_config["port"]}',
            f'--username={db_config["user"]}',
            f'--dbname={db_config["database"]}',
            '--data-only',
            '--file', data_file,
            '--verbose'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"✅ Export données créé: {data_file}")
            return {'data_file': data_file, 'status': 'exported'}
        else:
            logger.error(f"❌ Erreur export données: {result.stderr}")
            return {'error': result.stderr}
            
    except Exception as e:
        logger.error(f"❌ Erreur export données: {e}")
        return {'error': str(e)}

def analyze_table_statistics():
    """Analyse les statistiques des tables"""
    try:
        with get_database_session_context() as session:
            # ANALYZE sur toutes les tables
            session.execute(text("ANALYZE"))
        
        logger.info("✅ Statistiques des tables mises à jour")
        return {'status': 'analyzed'}
                
    except Exception as e:
        logger.error(f"❌ Erreur analyse statistiques: {e}")
        return {'error': str(e)}

def reindex_all_tables():
    """Réindexe toutes les tables"""
    try:
        with get_database_session_context() as session:
            # REINDEX sur toutes les tables
            session.execute(text("REINDEX DATABASE " + db_config['database']))
        
        logger.info("✅ Réindexation terminée")
        return {'status': 'reindexed'}
        
    except Exception as e:
        logger.error(f"❌ Erreur réindexation: {e}")
        return {'error': str(e)}

def vacuum_all_tables():
    """VACUUM sur toutes les tables"""
    try:
        with get_database_session_context() as session:
            # VACUUM sur toutes les tables
            session.execute(text("VACUUM ANALYZE"))
        
        logger.info("✅ VACUUM terminé")
        return {'status': 'vacuumed'}
        
    except Exception as e:
        logger.error(f"❌ Erreur VACUUM: {e}")
        return {'error': str(e)}

def get_database_locks_detailed():
    """Obtient les détails des verrous"""
    try:
        query = """
        SELECT 
            l.locktype,
            l.database,
            l.relation,
            l.page,
            l.tuple,
            l.virtualxid,
            l.transactionid,
            l.classid,
            l.objid,
            l.objsubid,
            l.virtualtransaction,
            l.pid,
            l.mode,
            l.granted,
            a.usename,
            a.application_name,
            a.client_addr,
            a.state,
            a.query_start,
            a.query
        FROM pg_locks l
        LEFT JOIN pg_stat_activity a ON l.pid = a.pid
        WHERE l.database = (SELECT oid FROM pg_database WHERE datname = current_database())
        ORDER BY l.granted, l.pid
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur détails verrous: {e}")
        return []

def get_deadlock_information():
    """Obtient les informations sur les deadlocks"""
    try:
        query = """
        SELECT 
            deadlocks,
            stats_reset
        FROM pg_stat_database 
        WHERE datname = current_database()
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur infos deadlock: {e}")
        return []

def get_database_connections_detailed():
    """Obtient les détails des connexions"""
    try:
        query = """
        SELECT 
            pid,
            usename,
            application_name,
            client_addr,
            client_hostname,
            client_port,
            backend_start,
            xact_start,
            query_start,
            state_change,
            state,
            backend_xid,
            backend_xmin,
            query,
            backend_type
        FROM pg_stat_activity 
        WHERE datname = current_database()
        ORDER BY backend_start DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur détails connexions: {e}")
        return []

def get_database_connections_summary():
    """Obtient un résumé des connexions"""
    try:
        query = """
        SELECT 
            state,
            COUNT(*) as connection_count
        FROM pg_stat_activity 
        WHERE datname = current_database()
        GROUP BY state
        ORDER BY connection_count DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur résumé connexions: {e}")
        return []

def get_replication_status():
    """Obtient le statut de la réplication"""
    try:
        query = """
        SELECT 
            client_addr,
            state,
            sent_lsn,
            write_lsn,
            flush_lsn,
            replay_lsn
        FROM pg_stat_replication
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur statut réplication: {e}")
        return []

def get_database_size_detailed():
    """Obtient la taille détaillée de la base de données"""
    try:
        query = """
        SELECT 
            pg_database.datname,
            pg_size_pretty(pg_database_size(pg_database.datname)) AS size,
            pg_database_size(pg_database.datname) AS size_bytes
        FROM pg_database
        WHERE pg_database.datname = current_database()
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur taille détaillée: {e}")
        return []

def get_table_size_detailed():
    """Obtient la taille détaillée de toutes les tables"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
            pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS index_size,
            pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur taille tables détaillée: {e}")
        return []

def get_index_size_detailed():
    """Obtient la taille détaillée de tous les index"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            indexname,
            pg_size_pretty(pg_relation_size(indexrelid)) AS size,
            pg_relation_size(indexrelid) AS size_bytes
        FROM pg_stat_user_indexes 
        WHERE schemaname = 'public'
        ORDER BY pg_relation_size(indexrelid) DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur taille index détaillée: {e}")
        return []

def get_database_bloat_analysis():
    """Analyse du bloat de la base de données"""
    try:
        query = """
        SELECT 
            schemaname,
            tablename,
            n_live_tup as live_tuples,
            n_dead_tup as dead_tuples,
            CASE 
                WHEN n_live_tup > 0 
                THEN ROUND((n_dead_tup::float / n_live_tup::float) * 100, 2)
                ELSE 0 
            END as bloat_percentage,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
            last_vacuum,
            last_autovacuum
        FROM pg_stat_user_tables
        WHERE schemaname = 'public'
        ORDER BY bloat_percentage DESC
        """
        return execute_postgresql_query(query)
    except Exception as e:
        logger.error(f"❌ Erreur analyse bloat: {e}")
        return []

def get_database_maintenance_recommendations():
    """Recommandations de maintenance de la base de données"""
    try:
        recommendations = []
        
        # Analyser le bloat
        bloat_data = get_database_bloat_analysis()
        for table in bloat_data:
            if table.get('bloat_percentage', 0) > 20:
                recommendations.append({
                    'type': 'vacuum',
                    'priority': 'high',
                    'table': table['tablename'],
                    'bloat_percentage': table['bloat_percentage'],
                    'recommendation': f'Table {table["tablename"]} a {table["bloat_percentage"]}% de bloat',
                    'action': 'Exécuter VACUUM FULL sur cette table'
                })
        
        # Analyser les dernières maintenances
        table_stats = get_table_activity_stats()
        for table in table_stats:
            last_vacuum = table.get('last_vacuum')
            last_autovacuum = table.get('last_autovacuum')
            
            if not last_vacuum and not last_autovacuum:
                recommendations.append({
                    'type': 'maintenance',
                    'priority': 'medium',
                    'table': table['tablename'],
                    'recommendation': f'Table {table["tablename"]} n\'a jamais été vacuumée',
                    'action': 'Exécuter VACUUM ANALYZE sur cette table'
                })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'recommendations_count': len(recommendations),
            'recommendations': recommendations
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur recommandations maintenance: {e}")
        return {'error': str(e)}

def get_database_performance_summary():
    """Résumé des performances de la base de données"""
    try:
        summary = {
            'timestamp': datetime.now().isoformat(),
            'database_info': get_database_size_detailed(),
            'table_sizes': get_table_size_detailed(),
            'index_sizes': get_index_size_detailed(),
            'bloat_analysis': get_database_bloat_analysis(),
            'connection_details': get_database_connections_detailed(),
            'lock_details': get_database_locks_detailed(),
            'deadlock_info': get_deadlock_information(),
            'activity_summary': get_database_activity_summary(),
            'io_stats': get_database_io_stats(),
            'memory_stats': get_database_memory_stats()
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Erreur résumé performances: {e}")
        return {'error': str(e)}

def execute_maintenance_tasks():
    """Exécute toutes les tâches de maintenance"""
    try:
        results = {}
        
        # ANALYZE
        results['analyze'] = analyze_table_statistics()
        
        # VACUUM
        results['vacuum'] = vacuum_all_tables()
        
        # REINDEX
        results['reindex'] = reindex_all_tables()
        
        logger.info("✅ Toutes les tâches de maintenance exécutées")
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'completed',
            'results': results
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur tâches maintenance: {e}")
        return {'error': str(e)}

def get_database_health_score():
    """Calcule un score de santé de la base de données"""
    try:
        score = 100
        issues = []
        
        # Vérifier les connexions
        active_conns = get_active_connections()
        if active_conns:
            total_conns = sum([conn['connection_count'] for conn in active_conns])
            if total_conns > 80:
                score -= 20
                issues.append('Trop de connexions actives')
        
        # Vérifier le cache hit ratio
        io_stats = get_database_io_stats()
        if io_stats:
            cache_hit_ratio = io_stats[0].get('cache_hit_ratio', 0)
            if cache_hit_ratio < 90:
                score -= 15
                issues.append(f'Cache hit ratio faible: {cache_hit_ratio}%')
        
        # Vérifier les verrous
        locks = get_database_locks()
        if locks and len(locks) > 50:
            score -= 25
            issues.append('Trop de verrous actifs')
        
        # Vérifier le bloat
        bloat_data = get_database_bloat_analysis()
        high_bloat_tables = [t for t in bloat_data if t.get('bloat_percentage', 0) > 30]
        if high_bloat_tables:
            score -= 10
            issues.append(f'{len(high_bloat_tables)} tables avec bloat élevé')
        
        # Déterminer le niveau de santé
        if score >= 90:
            health_level = 'EXCELLENT'
        elif score >= 75:
            health_level = 'GOOD'
        elif score >= 60:
            health_level = 'FAIR'
        elif score >= 40:
            health_level = 'POOR'
        else:
            health_level = 'CRITICAL'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health_score': score,
            'health_level': health_level,
            'issues': issues,
            'total_issues': len(issues)
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur score santé: {e}")
        return {'error': str(e)}

def generate_database_maintenance_report():
    """Génère un rapport de maintenance complet"""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'health_score': get_database_health_score(),
            'performance_summary': get_database_performance_summary(),
            'maintenance_recommendations': get_database_maintenance_recommendations(),
            'diagnostics': diagnose_database_issues(),
            'tuning_suggestions': get_database_tuning_suggestions(),
            'optimization_recommendations': get_database_recommendations(),
            'size_analysis': {
                'database_size': get_database_size_detailed(),
                'table_sizes': get_table_size_detailed(),
                'index_sizes': get_index_size_detailed()
            },
            'bloat_analysis': get_database_bloat_analysis(),
            'connection_analysis': get_database_connections_detailed(),
            'lock_analysis': get_database_locks_detailed()
        }
        
        # Sauvegarder le rapport
        report_file = f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Rapport de maintenance généré: {report_file}")
        return report
        
    except Exception as e:
        logger.error(f"❌ Erreur rapport maintenance: {e}")
        return {'error': str(e)}

# Fonctions utilitaires avancées
def get_database_metadata():
    """Obtient toutes les métadonnées de la base de données"""
    try:
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'database_info': get_database_info(),
            'connection_info': get_connection_info(),
            'tables': get_table_stats(),
            'indexes': get_index_info(),
            'schemas': get_database_schemas(),
            'users': get_database_users(),
            'roles': get_database_roles(),
            'sequences': get_database_sequences(),
            'functions': get_database_functions(),
            'triggers': get_database_triggers(),
            'views': get_database_views(),
            'constraints': get_table_constraints(),
            'foreign_keys': get_foreign_keys(),
            'configuration': get_database_configuration()
        }
        
        return metadata
        
    except Exception as e:
        logger.error(f"❌ Erreur métadonnées: {e}")
        return {'error': str(e)}

def validate_database_integrity():
    """Valide l'intégrité de la base de données"""
    try:
        integrity_checks = []
        
        # Vérifier les contraintes de clés étrangères
        try:
            query = """
            SELECT 
                tc.table_name,
                tc.constraint_name,
                tc.constraint_type
            FROM information_schema.table_constraints tc
            WHERE tc.table_schema = 'public'
            AND tc.constraint_type = 'FOREIGN KEY'
            """
            fk_constraints = execute_postgresql_query(query)
            integrity_checks.append({
                'check': 'Foreign Key Constraints',
                'count': len(fk_constraints),
                'status': 'OK'
            })
        except Exception as e:
            integrity_checks.append({
                'check': 'Foreign Key Constraints',
                'status': 'ERROR',
                'error': str(e)
            })
        
        # Vérifier les index
        try:
            indexes = get_index_info()
            integrity_checks.append({
                'check': 'Database Indexes',
                'count': len(indexes),
                'status': 'OK'
            })
        except Exception as e:
            integrity_checks.append({
                'check': 'Database Indexes',
                'status': 'ERROR',
                'error': str(e)
            })
        
        # Vérifier les tables
        try:
            tables = get_table_stats()
            integrity_checks.append({
                'check': 'Database Tables',
                'count': len(tables),
                'status': 'OK'
            })
        except Exception as e:
            integrity_checks.append({
                'check': 'Database Tables',
                'status': 'ERROR',
                'error': str(e)
            })
        
        return {
            'timestamp': datetime.now().isoformat(),
            'integrity_checks': integrity_checks,
            'overall_status': 'OK' if all(check['status'] == 'OK' for check in integrity_checks) else 'ERROR'
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur validation intégrité: {e}")
        return {'error': str(e)}

def get_database_usage_statistics():
    """Statistiques d'utilisation de la base de données"""
    try:
        stats = {
            'timestamp': datetime.now().isoformat(),
            'database_activity': get_database_activity_summary(),
            'table_activity': get_table_activity_stats(),
            'index_activity': get_index_activity_stats(),
            'query_performance': get_query_performance_stats(),
            'connection_stats': get_database_connections_summary(),
            'io_statistics': get_database_io_stats(),
            'memory_usage': get_database_memory_stats()
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Erreur stats utilisation: {e}")
        return {'error': str(e)}

def get_database_security_report():
    """Rapport de sécurité de la base de données"""
    try:
        security_report = {
            'timestamp': datetime.now().isoformat(),
            'users_with_privileges': get_database_users(),
            'roles': get_database_roles(),
            'connections': get_database_connections_detailed(),
            'locks': get_database_locks_detailed(),
            'security_alerts': get_security_alerts(),
            'configuration': get_database_configuration()
        }
        
        return security_report
        
    except Exception as e:
        logger.error(f"❌ Erreur rapport sécurité: {e}")
        return {'error': str(e)}

# Fonctions finales et utilitaires complets
def get_comprehensive_database_report():
    """Rapport complet et définitif de la base de données"""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'database_overview': {
                'connection_status': test_connection(),
                'database_info': get_database_info(),
                'health_score': get_database_health_score(),
                'size_analysis': get_database_size_detailed()
            },
            'performance_metrics': {
                'real_time_metrics': get_real_time_metrics(),
                'load_metrics': get_system_load_metrics(),
                'query_performance': get_query_performance_stats(),
                'database_activity': get_database_activity_summary(),
                'io_statistics': get_database_io_stats(),
                'memory_statistics': get_database_memory_stats()
            },
            'maintenance_status': {
                'bloat_analysis': get_database_bloat_analysis(),
                'maintenance_recommendations': get_database_maintenance_recommendations(),
                'tuning_suggestions': get_database_tuning_suggestions(),
                'optimization_recommendations': get_database_recommendations()
            },
            'security_analysis': {
                'users': get_database_users(),
                'roles': get_database_roles(),
                'connections': get_database_connections_detailed(),
                'locks': get_database_locks_detailed(),
                'security_alerts': get_security_alerts()
            },
            'system_alerts': {
                'all_alerts': get_all_alerts(),
                'system_alerts': get_system_alerts(),
                'performance_alerts': get_performance_alerts(),
                'data_quality_alerts': get_data_quality_alerts()
            },
            'metadata_complete': {
                'tables': get_table_stats(),
                'indexes': get_index_info(),
                'schemas': get_database_schemas(),
                'sequences': get_database_sequences(),
                'functions': get_database_functions(),
                'triggers': get_database_triggers(),
                'views': get_database_views(),
                'constraints': get_table_constraints(),
                'foreign_keys': get_foreign_keys()
            },
            'diagnostics': {
                'database_issues': diagnose_database_issues(),
                'integrity_validation': validate_database_integrity(),
                'usage_statistics': get_database_usage_statistics()
            }
        }
        
        # Sauvegarder le rapport complet
        report_file = f"comprehensive_database_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Rapport complet généré: {report_file}")
        return report
                
    except Exception as e:
        logger.error(f"❌ Erreur rapport complet: {e}")
        return {'error': str(e)}

def get_database_capacity_planning():
    """Planification de la capacité de la base de données"""
    try:
        planning = {
            'timestamp': datetime.now().isoformat(),
            'current_usage': {
                'database_size': get_database_size_detailed(),
                'table_sizes': get_table_size_detailed(),
                'index_sizes': get_index_size_detailed(),
                'connection_usage': get_database_connections_summary()
            },
            'growth_trends': {
                'daily_stats': get_daily_processing_stats(30),
                'monthly_trends': get_monthly_trends(),
                'table_activity': get_table_activity_stats()
            },
            'capacity_recommendations': [],
            'projected_growth': {}
        }
        
        # Analyser la croissance
        db_size = get_database_size()
        if 'GB' in db_size:
            size_gb = float(db_size.split()[0])
            
            # Projeter la croissance sur 6 mois
            projected_6m = size_gb * 1.5  # 50% de croissance
            projected_1y = size_gb * 2.0  # 100% de croissance
            
            planning['projected_growth'] = {
                'current_size_gb': size_gb,
                'projected_6_months_gb': projected_6m,
                'projected_1_year_gb': projected_1y,
                'growth_rate_per_month': 8.3  # 50% sur 6 mois
            }
            
            # Recommandations de capacité
            if size_gb > 10:
                planning['capacity_recommendations'].append({
                    'priority': 'high',
                    'recommendation': 'Base de données volumineuse',
                    'action': 'Considérer le partitionnement et l\'archivage'
                })
            
            if projected_1y > 50:
                planning['capacity_recommendations'].append({
                    'priority': 'critical',
                    'recommendation': 'Croissance rapide prévue',
                    'action': 'Planifier l\'expansion du stockage'
                })
        
        return planning
        
    except Exception as e:
        logger.error(f"❌ Erreur planification capacité: {e}")
        return {'error': str(e)}

def get_database_optimization_plan():
    """Plan d'optimisation de la base de données"""
    try:
        optimization_plan = {
            'timestamp': datetime.now().isoformat(),
            'current_performance': {
                'health_score': get_database_health_score(),
                'query_performance': get_query_performance_stats(),
                'io_stats': get_database_io_stats(),
                'memory_stats': get_database_memory_stats()
            },
            'optimization_opportunities': [],
            'recommended_actions': [],
            'expected_improvements': {}
        }
        
        # Identifier les opportunités d'optimisation
        slow_queries = get_slow_queries(10)
        for query in slow_queries:
            # Gérer les tuples et dictionnaires
            if isinstance(query, tuple):
                mean_exec_time = query[1] if len(query) > 1 else 0
            else:
                mean_exec_time = query.get('mean_exec_time', 0)
            
            if mean_exec_time > 1000:
                optimization_plan['optimization_opportunities'].append({
                    'type': 'query_optimization',
                    'issue': 'Requête lente',
                    'details': f'Temps moyen: {mean_exec_time}ms',
                    'impact': 'high'
                })
        
        # Analyser les index inutilisés
        index_stats = get_index_activity_stats()
        unused_indexes = [idx for idx in index_stats if idx.get('index_scans', 0) == 0]
        if unused_indexes:
            optimization_plan['optimization_opportunities'].append({
                'type': 'index_optimization',
                'issue': f'{len(unused_indexes)} index inutilisés',
                'details': 'Index qui ne sont jamais utilisés',
                'impact': 'medium'
            })
        
        # Analyser le bloat
        bloat_data = get_database_bloat_analysis()
        high_bloat_tables = [t for t in bloat_data if t.get('bloat_percentage', 0) > 30]
        if high_bloat_tables:
            optimization_plan['optimization_opportunities'].append({
                'type': 'maintenance',
                'issue': f'{len(high_bloat_tables)} tables avec bloat élevé',
                'details': 'Tables nécessitant un VACUUM',
                'impact': 'high'
            })
        
        # Recommandations d'actions
        if optimization_plan['optimization_opportunities']:
            optimization_plan['recommended_actions'] = [
                'Exécuter VACUUM ANALYZE sur toutes les tables',
                'Réindexer les tables avec bloat élevé',
                'Supprimer les index inutilisés',
                'Optimiser les requêtes lentes',
                'Ajuster la configuration mémoire'
            ]
        
        return optimization_plan
        
    except Exception as e:
        logger.error(f"❌ Erreur plan optimisation: {e}")
        return {'error': str(e)}

def get_database_backup_strategy():
    """Stratégie de sauvegarde de la base de données"""
    try:
        strategy = {
            'timestamp': datetime.now().isoformat(),
            'current_backup_info': {
                'database_size': get_database_size_detailed(),
                'table_count': len(get_table_stats()),
                'data_volume': get_table_size_detailed()
            },
            'backup_recommendations': [],
            'retention_policy': {},
            'recovery_procedures': []
        }
        
        # Recommandations de sauvegarde
        db_size = get_database_size()
        if 'GB' in db_size:
            size_gb = float(db_size.split()[0])
            
            if size_gb < 1:
                strategy['backup_recommendations'].append({
                    'frequency': 'Daily',
                    'type': 'Full backup',
                    'reason': 'Base de données petite'
                })
            elif size_gb < 10:
                strategy['backup_recommendations'].append({
                    'frequency': 'Daily full + Hourly incremental',
                    'type': 'Full + Incremental',
                    'reason': 'Base de données moyenne'
                })
            else:
                strategy['backup_recommendations'].append({
                    'frequency': 'Weekly full + Daily incremental',
                    'type': 'Full + Incremental',
                    'reason': 'Base de données volumineuse'
                })
        
        # Politique de rétention
        strategy['retention_policy'] = {
            'daily_backups': '30 days',
            'weekly_backups': '12 weeks',
            'monthly_backups': '12 months',
            'yearly_backups': '7 years'
        }
        
        # Procédures de récupération
        strategy['recovery_procedures'] = [
            'Récupération complète de la base de données',
            'Récupération point-in-time (PITR)',
            'Récupération de tables spécifiques',
            'Récupération de données corrompues'
        ]
        
        return strategy
        
    except Exception as e:
        logger.error(f"❌ Erreur stratégie sauvegarde: {e}")
        return {'error': str(e)}

def get_database_monitoring_setup():
    """Configuration du monitoring de la base de données"""
    try:
        monitoring_setup = {
            'timestamp': datetime.now().isoformat(),
            'monitoring_metrics': {
                'performance_metrics': [
                    'Temps de réponse des requêtes',
                    'Utilisation CPU et mémoire',
                    'Activité des connexions',
                    'Hit ratio du cache',
                    'Activité des index'
                ],
                'availability_metrics': [
                    'Disponibilité de la base',
                    'Temps de connexion',
                    'Erreurs de connexion',
                    'Timeout des requêtes'
                ],
                'capacity_metrics': [
                    'Taille de la base de données',
                    'Espace disque disponible',
                    'Nombre de connexions',
                    'Croissance des données'
                ],
                'security_metrics': [
                    'Tentatives de connexion échouées',
                    'Privilèges utilisateur',
                    'Accès aux données sensibles',
                    'Audit des modifications'
                ]
            },
            'alerting_rules': [
                {
                    'metric': 'Query Response Time',
                    'threshold': '> 5 seconds',
                    'severity': 'warning'
                },
                {
                    'metric': 'Database Size',
                    'threshold': '> 10GB',
                    'severity': 'info'
                },
                {
                    'metric': 'Active Connections',
                    'threshold': '> 80% of max_connections',
                    'severity': 'critical'
                },
                {
                    'metric': 'Cache Hit Ratio',
                    'threshold': '< 90%',
                    'severity': 'warning'
                }
            ],
            'monitoring_tools': [
                'pg_stat_statements pour les requêtes',
                'pg_stat_activity pour les connexions',
                'pg_stat_database pour les métriques globales',
                'pg_stat_user_tables pour les tables',
                'pg_stat_user_indexes pour les index'
            ]
        }
        
        return monitoring_setup
                
    except Exception as e:
        logger.error(f"❌ Erreur configuration monitoring: {e}")
        return {'error': str(e)}

def get_database_security_hardening():
    """Renforcement de la sécurité de la base de données"""
    try:
        security_hardening = {
            'timestamp': datetime.now().isoformat(),
            'current_security_status': {
                'users': get_database_users(),
                'roles': get_database_roles(),
                'connections': get_database_connections_detailed(),
                'configuration': get_database_configuration()
            },
            'security_recommendations': [],
            'hardening_checklist': [
                'Changer les mots de passe par défaut',
                'Désactiver les utilisateurs non utilisés',
                'Limiter les privilèges utilisateur',
                'Configurer l\'audit des connexions',
                'Chiffrer les connexions SSL/TLS',
                'Configurer les règles de pare-feu',
                'Mettre à jour PostgreSQL régulièrement',
                'Configurer la sauvegarde sécurisée'
            ],
            'compliance_requirements': {
                'data_encryption': 'Recommandé',
                'access_control': 'Obligatoire',
                'audit_logging': 'Obligatoire',
                'backup_encryption': 'Recommandé',
                'network_security': 'Obligatoire'
            }
        }
        
        # Recommandations spécifiques
        users = get_database_users()
        for user in users:
            if user.get('usesuper', False):
                security_hardening['security_recommendations'].append({
                    'type': 'privilege_review',
                    'user': user['usename'],
                    'recommendation': 'Utilisateur avec privilèges superuser',
                    'action': 'Vérifier si nécessaire'
                })
        
        return security_hardening
        
    except Exception as e:
        logger.error(f"❌ Erreur renforcement sécurité: {e}")
        return {'error': str(e)}

def get_database_disaster_recovery_plan():
    """Plan de récupération après sinistre"""
    try:
        disaster_recovery = {
            'timestamp': datetime.now().isoformat(),
            'current_infrastructure': {
                'database_info': get_database_info(),
                'backup_capability': 'pg_dump disponible',
                'replication_status': get_replication_status()
            },
            'recovery_objectives': {
                'rto': '4 hours',  # Recovery Time Objective
                'rpo': '1 hour',   # Recovery Point Objective
                'availability_target': '99.9%'
            },
            'backup_strategy': {
                'full_backup': 'Daily at 2 AM',
                'incremental_backup': 'Every 4 hours',
                'wal_archiving': 'Continuous',
                'backup_retention': '30 days'
            },
            'recovery_procedures': [
                'Identification du type de sinistre',
                'Évaluation de l\'étendue des dommages',
                'Activation du plan de récupération',
                'Restauration depuis la sauvegarde la plus récente',
                'Validation de l\'intégrité des données',
                'Redémarrage des services applicatifs',
                'Tests de fonctionnement',
                'Documentation de l\'incident'
            ],
            'contact_information': {
                'dba_primary': 'DBA Principal',
                'dba_backup': 'DBA de secours',
                'system_admin': 'Administrateur système',
                'management': 'Direction IT'
            },
            'testing_schedule': {
                'full_recovery_test': 'Quarterly',
                'partial_recovery_test': 'Monthly',
                'backup_validation': 'Weekly'
            }
        }
        
        return disaster_recovery
        
    except Exception as e:
        logger.error(f"❌ Erreur plan récupération: {e}")
        return {'error': str(e)}

def get_final_comprehensive_report():
    """Rapport final et complet de la base de données INSPECT_IA"""
    try:
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'report_type': 'COMPREHENSIVE_DATABASE_ANALYSIS',
            'database_name': db_config['database'],
            'analysis_summary': {
                'total_functions_available': 150,  # Estimation du nombre de fonctions
                'analysis_categories': [
                    'Performance Monitoring',
                    'Maintenance & Optimization',
                    'Security Analysis',
                    'Capacity Planning',
                    'Backup & Recovery',
                    'Health Assessment',
                    'Alerting & Diagnostics'
                ]
            },
            'executive_summary': {
                'database_health': get_database_health_score(),
                'performance_status': 'Analysé',
                'security_status': 'Évalué',
                'maintenance_status': 'Recommandations fournies',
                'capacity_status': 'Planifié'
            },
            'detailed_analysis': {
                'comprehensive_report': get_comprehensive_database_report(),
                'capacity_planning': get_database_capacity_planning(),
                'optimization_plan': get_database_optimization_plan(),
                'backup_strategy': get_database_backup_strategy(),
                'monitoring_setup': get_database_monitoring_setup(),
                'security_hardening': get_database_security_hardening(),
                'disaster_recovery': get_database_disaster_recovery_plan()
            },
            'recommendations_summary': {
                'immediate_actions': [
                    'Exécuter VACUUM ANALYZE',
                    'Vérifier les index inutilisés',
                    'Analyser les requêtes lentes'
                ],
                'short_term_actions': [
                    'Optimiser la configuration mémoire',
                    'Implémenter le monitoring',
                    'Configurer les alertes'
                ],
                'long_term_actions': [
                    'Planifier la croissance',
                    'Implémenter la haute disponibilité',
                    'Mettre en place la réplication'
                ]
            },
            'next_steps': [
                '1. Examiner les recommandations immédiates',
                '2. Planifier les actions court terme',
                '3. Développer la stratégie long terme',
                '4. Mettre en place le monitoring continu',
                '5. Programmer les révisions régulières'
            ]
        }
        
        # Sauvegarder le rapport final
        final_report_file = f"FINAL_COMPREHENSIVE_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(final_report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"✅ RAPPORT FINAL COMPLET GÉNÉRÉ: {final_report_file}")
        logger.info("🎉 ANALYSE COMPLÈTE DE LA BASE DE DONNÉES TERMINÉE")
        
        return final_report
        
    except Exception as e:
        logger.error(f"❌ Erreur rapport final: {e}")
        return {'error': str(e)}

# Fonction principale pour exécuter toutes les analyses
def run_complete_database_analysis():
    """Exécute une analyse complète de la base de données"""
    try:
        logger.info("🚀 DÉBUT DE L'ANALYSE COMPLÈTE DE LA BASE DE DONNÉES")
        logger.info("=" * 60)
        
        # Test de connexion initial
        if not test_connection():
            logger.error("❌ Impossible de se connecter à la base de données")
            return {'error': 'Database connection failed'}
        
        logger.info("✅ Connexion à la base de données établie")
        
        # Exécuter toutes les analyses
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'connection_status': 'SUCCESS',
            'analysis_start_time': datetime.now().isoformat(),
            'final_report': get_final_comprehensive_report()
        }
        
        analysis_results['analysis_end_time'] = datetime.now().isoformat()
        
        logger.info("✅ ANALYSE COMPLÈTE TERMINÉE AVEC SUCCÈS")
        logger.info("=" * 60)
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"❌ Erreur analyse complète: {e}")
        return {'error': str(e)}

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