"""
Script d'installation de la base de données INSPECT_IA
Installe le schéma complet avec toutes les tables et relations
"""

import psycopg2
import psycopg2.extras
import logging
from pathlib import Path
import sys
import os

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from database_config import get_database_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_database():
    """Installe la base de données INSPECT_IA avec le schéma complet"""
    
    try:
        # Lire le schéma SQL
        schema_file = Path(__file__).parent / "schema_INSPECT_IA.sql"
        if not schema_file.exists():
            logger.error(f"❌ Fichier schéma non trouvé: {schema_file}")
            return False
        
        with open(schema_file, 'r', encoding='utf-8') as f:
            schema_sql = f.read()
        
        # Connexion à PostgreSQL
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        logger.info(f"🔗 Connexion à la base de données: {db_url}")
        
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Exécuter le schéma
        logger.info("📋 Exécution du schéma de base de données...")
        try:
            cursor.execute(schema_sql)
        except Exception as e:
            if "already exists" in str(e):
                logger.info("ℹ️ Tables déjà existantes, continuation...")
            else:
                raise
        
        # Vérifier les tables créées
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """)
        
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"✅ Tables créées: {len(tables)}")
        for table in tables:
            logger.info(f"   - {table}")
        
        # Insérer les données initiales
        logger.info("📊 Insertion des données initiales...")
        
        # Chapitres
        cursor.execute("""
            INSERT INTO chapters (chapter_id, chapter_number, chapter_name, description, specialization, 
                                best_model, model_performance, optimal_threshold, features_count, 
                                data_size, advanced_fraud_detection, business_features_count)
            VALUES 
                ('chap30', 30, 'Produits pharmaceutiques', 'Médicaments et produits pharmaceutiques', 'pharmaceutical',
                 'xgboost', '{"f1": 0.9796, "auc": 0.9995, "precision": 0.9889, "recall": 0.9705}', 0.35, 41, 25334, true, 16),
                ('chap84', 84, 'Machines et appareils mécaniques', 'Machines et équipements industriels', 'mechanical',
                 'xgboost', '{"f1": 0.9888, "auc": 0.9997, "precision": 0.9992, "recall": 0.9834}', 0.22, 54, 15000, true, 18),
                ('chap85', 85, 'Machines et appareils électriques', 'Équipements électriques et électroniques', 'electrical',
                 'lightgbm', '{"f1": 0.9791, "auc": 0.9995, "precision": 0.9985, "recall": 0.9872}', 0.22, 54, 12000, true, 16)
            ON CONFLICT (chapter_id) DO UPDATE SET
                best_model = EXCLUDED.best_model,
                model_performance = EXCLUDED.model_performance,
                optimal_threshold = EXCLUDED.optimal_threshold,
                features_count = EXCLUDED.features_count,
                data_size = EXCLUDED.data_size,
                business_features_count = EXCLUDED.business_features_count
        """)
        
        # Features de base
        basic_features = [
            ('poids_net_kg', 'numerical', 'Poids net en kilogrammes', 'float', True, '0.0'),
            ('nombre_colis', 'numerical', 'Nombre de colis', 'int', True, '1'),
            ('quantite_complement', 'numerical', 'Quantité complémentaire', 'float', True, '0.0'),
            ('taux_droits_percent', 'numerical', 'Taux de droits en pourcentage', 'float', True, '0.0'),
            ('valeur_caf', 'numerical', 'Valeur CAF', 'float', True, '0.0'),
            ('code_sh_complet', 'categorical', 'Code SH complet', 'string', True, ''),
            ('code_pays_origine', 'categorical', 'Code pays d\'origine', 'string', True, ''),
            ('code_pays_provenance', 'categorical', 'Code pays de provenance', 'string', True, ''),
            ('regime_complet', 'categorical', 'Régime complet', 'string', True, ''),
            ('statut_bae', 'categorical', 'Statut BAE', 'string', True, '')
        ]
        
        for feature_name, feature_type, description, data_type, is_required, default_value in basic_features:
            cursor.execute("""
                INSERT INTO features (feature_name, feature_type, feature_category, description, data_type, is_required, default_value)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (feature_name) DO NOTHING
            """, (feature_name, feature_type, 'basic', description, data_type, is_required, default_value))
        
        # Features de détection de fraude avancée
        fraud_features = [
            ('FRAUD_BIENAYME_TCHEBYCHEV', 'fraud_detection', 'Score Bienaymé-Tchebychev', 'float', True, '0.0'),
            ('FRAUD_ANALYSE_MIROIR_TEI', 'fraud_detection', 'Score analyse miroir TEI', 'float', True, '0.0'),
            ('FRAUD_DETECTION_ANOMALIES', 'fraud_detection', 'Score détection d\'anomalies', 'float', True, '0.0'),
            ('FRAUD_CONTROLE_VALEURS_ADMINISTREES', 'fraud_detection', 'Score contrôle valeurs administrées', 'float', True, '0.0'),
            ('FRAUD_SCORE_FINAL', 'fraud_detection', 'Score de fraude final', 'float', True, '0.0'),
            ('FRAUD_DECISION', 'fraud_detection', 'Décision de fraude', 'string', True, 'conforme')
        ]
        
        for feature_name, feature_type, description, data_type, is_required, default_value in fraud_features:
            cursor.execute("""
                INSERT INTO features (feature_name, feature_type, feature_category, description, data_type, is_required, default_value)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (feature_name) DO NOTHING
            """, (feature_name, feature_type, 'fraud_detection', description, data_type, is_required, default_value))
        
        # Features business par chapitre
        business_features_chap30 = [
            'BUSINESS_GLISSEMENT_TARIFAIRE', 'BUSINESS_GLISSEMENT_DESCRIPTION',
            'BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'BUSINESS_VALEUR_UNITAIRE_SUSPECTE', 'BUSINESS_IS_ANTIPALUDEEN'
        ]
        
        business_features_chap84 = [
            'BUSINESS_GLISSEMENT_MACHINE', 'BUSINESS_GLISSEMENT_PAYS_MACHINES',
            'BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'BUSINESS_IS_MACHINE', 'BUSINESS_IS_ELECTRONIQUE'
        ]
        
        business_features_chap85 = [
            'BUSINESS_GLISSEMENT_ELECTRONIQUE', 'BUSINESS_GLISSEMENT_PAYS_ELECTRONIQUES',
            'BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'BUSINESS_POIDS_FAIBLE', 'BUSINESS_IS_ELECTRONIQUE', 'BUSINESS_IS_TELEPHONE'
        ]
        
        # Insérer les features business
        for feature_name in business_features_chap30 + business_features_chap84 + business_features_chap85:
            cursor.execute("""
                INSERT INTO features (feature_name, feature_type, feature_category, description, data_type, is_required, default_value)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (feature_name) DO NOTHING
            """, (feature_name, 'business', 'business', f'Feature business {feature_name}', 'float', False, '0.0'))
        
        # Associer les features aux chapitres
        for chapter_id, features in [('chap30', business_features_chap30), 
                                   ('chap84', business_features_chap84), 
                                   ('chap85', business_features_chap85)]:
            for feature_name in features:
                cursor.execute("""
                    INSERT INTO chapter_features (chapter_id, feature_id, is_used, importance_score)
                    SELECT %s, f.feature_id, true, 0.8
                    FROM features f WHERE f.feature_name = %s
                    ON CONFLICT (chapter_id, feature_id) DO NOTHING
                """, (chapter_id, feature_name))
        
        # Associer toutes les features de base à tous les chapitres
        for chapter_id in ['chap30', 'chap84', 'chap85']:
            for feature_name, _, _, _, _, _ in basic_features + fraud_features:
                cursor.execute("""
                    INSERT INTO chapter_features (chapter_id, feature_id, is_used, importance_score)
                    SELECT %s, f.feature_id, true, 0.9
                    FROM features f WHERE f.feature_name = %s
                    ON CONFLICT (chapter_id, feature_id) DO NOTHING
                """, (chapter_id, feature_name))
        
        cursor.close()
        conn.close()
        
        logger.info("✅ Base de données INSPECT_IA installée avec succès!")
        logger.info("📊 Données initiales insérées:")
        logger.info("   - 3 chapitres (30, 84, 85)")
        logger.info("   - Features de base et de détection de fraude")
        logger.info("   - Features business spécifiques par chapitre")
        logger.info("   - Associations chapitres-features")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur installation base de données: {e}")
        return False

if __name__ == "__main__":
    success = install_database()
    if success:
        print("🎉 Installation terminée avec succès!")
    else:
        print("💥 Échec de l'installation!")
        sys.exit(1)