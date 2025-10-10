"""
Script de vérification des features pour INSPECT_IA
Vérifie que toutes les features nécessaires sont présentes dans la base de données
"""

import psycopg2
import psycopg2.extras
import logging
import sys
from pathlib import Path
import json

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

from database_config import get_database_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_expected_features():
    """Retourne la liste des features attendues par catégorie"""
    
    # Features de base (tous les chapitres)
    basic_features = [
        'poids_net_kg', 'nombre_colis', 'quantite_complement', 'taux_droits_percent',
        'valeur_caf', 'valeur_unitaire_kg', 'ratio_douane_caf', 'code_sh_complet',
        'code_pays_origine', 'code_pays_provenance', 'regime_complet', 'statut_bae',
        'type_regime', 'regime_douanier', 'regime_fiscal', 'code_produit_str',
        'pays_origine_str', 'pays_provenance_str', 'numero_article', 'precision_uemoa'
    ]
    
    # Features de détection de fraude avancée
    fraud_detection_features = [
        'FRAUD_BIENAYME_TCHEBYCHEV', 'FRAUD_ANALYSE_MIROIR_TEI', 'FRAUD_DETECTION_ANOMALIES',
        'FRAUD_CONTROLE_VALEURS_ADMINISTREES', 'FRAUD_SCORE_FINAL', 'FRAUD_DECISION'
    ]
    
    # Features business par chapitre
    business_features = {
        'chap30': [
            'BUSINESS_GLISSEMENT_TARIFAIRE', 'BUSINESS_GLISSEMENT_DESCRIPTION',
            'BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'BUSINESS_VALEUR_UNITAIRE_SUSPECTE', 'BUSINESS_IS_ANTIPALUDEEN'
        ],
        'chap84': [
            'BUSINESS_GLISSEMENT_MACHINE', 'BUSINESS_GLISSEMENT_PAYS_MACHINES',
            'BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'BUSINESS_IS_MACHINE', 'BUSINESS_IS_ELECTRONIQUE'
        ],
        'chap85': [
            'BUSINESS_GLISSEMENT_ELECTRONIQUE', 'BUSINESS_GLISSEMENT_PAYS_ELECTRONIQUES',
            'BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'BUSINESS_POIDS_FAIBLE', 'BUSINESS_IS_ELECTRONIQUE', 'BUSINESS_IS_TELEPHONE'
        ]
    }
    
    return {
        'basic': basic_features,
        'fraud_detection': fraud_detection_features,
        'business': business_features
    }

def verify_features_in_database():
    """Vérifie que toutes les features attendues sont présentes dans la base de données"""
    
    try:
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        expected_features = get_expected_features()
        
        logger.info("🔍 Vérification des features dans la base de données")
        logger.info("=" * 60)
        
        all_good = True
        
        # Vérifier les features de base
        logger.info("\n📊 Features de base:")
        cursor.execute("""
            SELECT feature_name, feature_type, feature_category 
            FROM features 
            WHERE feature_category = 'basic'
            ORDER BY feature_name
        """)
        
        db_basic_features = {row[0]: {'type': row[1], 'category': row[2]} for row in cursor.fetchall()}
        
        for feature in expected_features['basic']:
            if feature in db_basic_features:
                logger.info(f"   ✅ {feature} ({db_basic_features[feature]['type']})")
            else:
                logger.error(f"   ❌ {feature} - MANQUANTE")
                all_good = False
        
        # Vérifier les features de détection de fraude
        logger.info("\n🕵️ Features de détection de fraude:")
        cursor.execute("""
            SELECT feature_name, feature_type, feature_category 
            FROM features 
            WHERE feature_category = 'fraud_detection'
            ORDER BY feature_name
        """)
        
        db_fraud_features = {row[0]: {'type': row[1], 'category': row[2]} for row in cursor.fetchall()}
        
        for feature in expected_features['fraud_detection']:
            if feature in db_fraud_features:
                logger.info(f"   ✅ {feature} ({db_fraud_features[feature]['type']})")
            else:
                logger.error(f"   ❌ {feature} - MANQUANTE")
                all_good = False
        
        # Vérifier les features business par chapitre
        logger.info("\n💼 Features business par chapitre:")
        
        for chapter_id, expected_business_features in expected_features['business'].items():
            logger.info(f"\n   📋 {chapter_id}:")
            
            cursor.execute("""
                SELECT f.feature_name, f.feature_type, f.feature_category
                FROM features f
                JOIN chapter_features cf ON f.feature_id = cf.feature_id
                WHERE cf.chapter_id = %s AND f.feature_category = 'business'
                ORDER BY f.feature_name
            """, (chapter_id,))
            
            db_chapter_business_features = {row[0]: {'type': row[1], 'category': row[2]} for row in cursor.fetchall()}
            
            for feature in expected_business_features:
                if feature in db_chapter_business_features:
                    logger.info(f"      ✅ {feature} ({db_chapter_business_features[feature]['type']})")
                else:
                    logger.error(f"      ❌ {feature} - MANQUANTE")
                    all_good = False
        
        # Vérifier les associations chapitres-features
        logger.info("\n🔗 Associations chapitres-features:")
        
        cursor.execute("""
            SELECT c.chapter_id, c.chapter_name, COUNT(cf.feature_id) as feature_count
            FROM chapters c
            LEFT JOIN chapter_features cf ON c.chapter_id = cf.chapter_id
            GROUP BY c.chapter_id, c.chapter_name
            ORDER BY c.chapter_number
        """)
        
        associations = cursor.fetchall()
        
        for chapter_id, chapter_name, count in associations:
            expected_count = (
                len(expected_features['basic']) + 
                len(expected_features['fraud_detection']) + 
                len(expected_features['business'].get(chapter_id, []))
            )
            
            if count >= expected_count:
                logger.info(f"   ✅ {chapter_id} ({chapter_name}): {count} features")
            else:
                logger.warning(f"   ⚠️ {chapter_id} ({chapter_name}): {count} features (attendu: {expected_count})")
                all_good = False
        
        # Statistiques globales
        logger.info("\n📈 Statistiques globales:")
        
        cursor.execute("SELECT COUNT(*) FROM features")
        total_features = cursor.fetchone()[0]
        logger.info(f"   Total features: {total_features}")
        
        cursor.execute("SELECT COUNT(*) FROM chapter_features")
        total_associations = cursor.fetchone()[0]
        logger.info(f"   Total associations: {total_associations}")
        
        cursor.execute("SELECT COUNT(DISTINCT chapter_id) FROM chapter_features")
        chapters_with_features = cursor.fetchone()[0]
        logger.info(f"   Chapitres avec features: {chapters_with_features}")
        
        cursor.close()
        conn.close()
        
        return all_good
        
    except Exception as e:
        logger.error(f"❌ Erreur vérification features: {e}")
        return False

def verify_feature_consistency():
    """Vérifie la cohérence des features entre les différents modules"""
    
    try:
        logger.info("\n🔍 Vérification de la cohérence des features")
        logger.info("=" * 60)
        
        # Vérifier la cohérence avec les modèles ML
        try:
            from src.chapters.chap30.ml_model_advanced import Chap30MLAdvanced
            from src.chapters.chap84.ml_model_advanced import Chap84MLAdvanced
            from src.chapters.chap85.ml_model_advanced import Chap85MLAdvanced
            
            logger.info("✅ Modules ML importés avec succès")
            
            # Test de récupération des features pour chaque chapitre
            chapters = [
                ('chap30', Chap30MLAdvanced),
                ('chap84', Chap84MLAdvanced),
                ('chap85', Chap85MLAdvanced)
            ]
            
            for chapter_id, ml_class in chapters:
                try:
                    ml_pipeline = ml_class()
                    features = ml_pipeline._get_feature_columns()
                    logger.info(f"   ✅ {chapter_id}: {len(features)} features récupérées")
                except Exception as e:
                    logger.error(f"   ❌ {chapter_id}: Erreur récupération features - {e}")
            
        except Exception as e:
            logger.error(f"❌ Erreur import modules ML: {e}")
            return False
        
        # Vérifier la cohérence avec le système RL
        try:
            from src.shared.advanced_reinforcement_learning import AdvancedRLManager
            
            logger.info("✅ Module RL importé avec succès")
            
            # Test de création d'un manager RL
            rl_manager = AdvancedRLManager("chap30")
            logger.info("   ✅ Manager RL créé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur import module RL: {e}")
            return False
        
        # Vérifier la cohérence avec le système OCR
        try:
            from src.shared.ocr_ingest import FIELD_MAPPING, CSV_TO_ML_MAPPING
            from src.shared.ocr_pipeline import CHAPTER_CONFIGS
            
            logger.info("✅ Modules OCR importés avec succès")
            logger.info(f"   Mapping fields: {len(FIELD_MAPPING)} entrées")
            logger.info(f"   Mapping CSV: {len(CSV_TO_ML_MAPPING)} entrées")
            logger.info(f"   Configurations chapitres: {len(CHAPTER_CONFIGS)} chapitres")
            
        except Exception as e:
            logger.error(f"❌ Erreur import modules OCR: {e}")
            return False
        
        logger.info("✅ Cohérence des features vérifiée")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur vérification cohérence: {e}")
        return False

def generate_feature_report():
    """Génère un rapport détaillé des features"""
    
    try:
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        logger.info("\n📋 Génération du rapport des features")
        logger.info("=" * 60)
        
        # Rapport par catégorie
        cursor.execute("""
            SELECT feature_category, COUNT(*) as count
            FROM features
            GROUP BY feature_category
            ORDER BY feature_category
        """)
        
        categories = cursor.fetchall()
        logger.info("📊 Features par catégorie:")
        for category, count in categories:
            logger.info(f"   {category}: {count} features")
        
        # Rapport par chapitre
        cursor.execute("""
            SELECT c.chapter_id, c.chapter_name, COUNT(cf.feature_id) as feature_count
            FROM chapters c
            LEFT JOIN chapter_features cf ON c.chapter_id = cf.chapter_id
            GROUP BY c.chapter_id, c.chapter_name
            ORDER BY c.chapter_number
        """)
        
        chapters = cursor.fetchall()
        logger.info("\n📊 Features par chapitre:")
        for chapter_id, chapter_name, count in chapters:
            logger.info(f"   {chapter_id} ({chapter_name}): {count} features")
        
        # Features manquantes ou problématiques
        cursor.execute("""
            SELECT f.feature_name, f.feature_type, f.feature_category
            FROM features f
            WHERE f.is_required = true AND f.default_value IS NULL
        """)
        
        problematic_features = cursor.fetchall()
        if problematic_features:
            logger.warning(f"\n⚠️ Features requises sans valeur par défaut: {len(problematic_features)}")
            for feature_name, feature_type, category in problematic_features:
                logger.warning(f"   - {feature_name} ({category})")
        
        cursor.close()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur génération rapport: {e}")
        return False

def run_feature_verification():
    """Exécute la vérification complète des features"""
    
    logger.info("🚀 Démarrage de la vérification des features INSPECT_IA")
    logger.info("=" * 80)
    
    tests = [
        ("Vérification features en base", verify_features_in_database),
        ("Vérification cohérence modules", verify_feature_consistency),
        ("Génération rapport", generate_feature_report)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 {test_name}")
        logger.info("-" * 50)
        
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
    logger.info("📊 RÉSUMÉ DE LA VÉRIFICATION")
    logger.info("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Résultat global: {passed}/{total} vérifications réussies")
    
    if passed == total:
        logger.info("🎉 TOUTES LES VÉRIFICATIONS SONT RÉUSSIES!")
        logger.info("✅ Les features sont correctement configurées")
        return True
    else:
        logger.error(f"💥 {total - passed} vérification(s) ont échoué")
        logger.error("❌ Des corrections sont nécessaires")
        return False

if __name__ == "__main__":
    success = run_feature_verification()
    if not success:
        sys.exit(1)