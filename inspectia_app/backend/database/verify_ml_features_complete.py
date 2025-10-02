"""
Compare les features utilisés par les modèles ML avec ceux en base
"""

import sys
import logging
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ml_features():
    """Récupère toutes les features utilisées par les modèles ML"""
    try:
        from src.chapters.chap30.ml_model_advanced import Chap30MLAdvanced
        from src.chapters.chap84.ml_model_advanced import Chap84MLAdvanced
        from src.chapters.chap85.ml_model_advanced import Chap85MLAdvanced
        
        # Initialiser les modèles
        chap30 = Chap30MLAdvanced()
        chap84 = Chap84MLAdvanced()
        chap85 = Chap85MLAdvanced()
        
        # Récupérer les features
        features30 = chap30._get_feature_columns()
        features84 = chap84._get_feature_columns()
        features85 = chap85._get_feature_columns()
        
        return {
            'chap30': features30,
            'chap84': features84,
            'chap85': features85
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features ML: {e}")
        return {}

def get_database_features():
    """Récupère toutes les features en base de données"""
    try:
        from database_config import get_database_config
        import psycopg2
        
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Récupérer toutes les features
        cursor.execute("SELECT feature_name, feature_category FROM features")
        db_features = cursor.fetchall()
        
        # Récupérer les associations chapitres-features
        cursor.execute("""
            SELECT c.chapter_id, f.feature_name, f.feature_category
            FROM chapters c
            JOIN chapter_features cf ON c.chapter_id = cf.chapter_id
            JOIN features f ON cf.feature_id = f.feature_id
            ORDER BY c.chapter_id, f.feature_category, f.feature_name
        """)
        chapter_features = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            'all_features': db_features,
            'chapter_features': chapter_features
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features base: {e}")
        return {}

def compare_features():
    """Compare les features ML avec ceux en base"""
    
    logger.info("🔍 VÉRIFICATION COMPLÈTE DES FEATURES ML")
    logger.info("=" * 80)
    
    # Récupérer les features ML
    ml_features = get_ml_features()
    if not ml_features:
        logger.error("❌ Impossible de récupérer les features ML")
        return False
    
    # Récupérer les features en base
    db_data = get_database_features()
    if not db_data:
        logger.error("❌ Impossible de récupérer les features en base")
        return False
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires pour faciliter la recherche
    db_feature_names = {name for name, category in db_features}
    
    # Créer un dictionnaire des features par chapitre
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = {}
        
        # Mapper les catégories de la base vers les catégories ML
        ml_category = feature_category
        if feature_category == 'basic':
            # Déterminer si c'est numeric ou categorical basé sur le nom
            if any(keyword in feature_name.upper() for keyword in ['VALEUR', 'POIDS', 'MONTANT', 'TAUX', 'RATIO', 'PRECISION']):
                ml_category = 'numeric'
            elif any(keyword in feature_name.upper() for keyword in ['CODE_', 'PAYS_', 'BUREAU', 'REGIME_', 'NUMERO_DPI']):
                ml_category = 'categorical'
            else:
                ml_category = 'numeric'  # Par défaut pour les autres features numériques
        
        if ml_category not in chapter_db_features[chapter_id]:
            chapter_db_features[chapter_id][ml_category] = set()
        chapter_db_features[chapter_id][ml_category].add(feature_name)
    
    all_good = True
    
    # Vérifier chaque chapitre
    for chapter_id, ml_chapter_features in ml_features.items():
        logger.info(f"\n📋 Vérification {chapter_id}:")
        logger.info("-" * 50)
        
        if chapter_id not in chapter_db_features:
            logger.error(f"❌ Chapitre {chapter_id} non trouvé en base")
            all_good = False
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        # Vérifier chaque catégorie
        for category, ml_features_list in ml_chapter_features.items():
            logger.info(f"\n   📊 Catégorie {category}:")
            
            if category not in db_chapter_features:
                logger.error(f"      ❌ Catégorie {category} manquante en base")
                all_good = False
                continue
            
            db_category_features = db_chapter_features[category]
            missing_features = []
            
            for feature in ml_features_list:
                if feature not in db_category_features:
                    missing_features.append(feature)
                    logger.error(f"      ❌ {feature} - MANQUANTE")
                else:
                    logger.info(f"      ✅ {feature}")
            
            if missing_features:
                logger.error(f"      📊 {len(missing_features)} features manquantes sur {len(ml_features_list)}")
                all_good = False
            else:
                logger.info(f"      📊 Toutes les {len(ml_features_list)} features sont présentes")
    
    # Résumé global
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ DE LA VÉRIFICATION")
    logger.info("=" * 80)
    
    total_ml_features = sum(len(features) for chapter_features in ml_features.values() 
                           for features in chapter_features.values())
    total_db_features = len(db_features)
    
    logger.info(f"📈 Total features ML: {total_ml_features}")
    logger.info(f"📈 Total features en base: {total_db_features}")
    
    if all_good:
        logger.info("🎉 TOUTES LES FEATURES ML SONT PRÉSENTES EN BASE!")
        logger.info("✅ Le système est cohérent")
        return True
    else:
        logger.error("💥 CERTAINES FEATURES ML SONT MANQUANTES EN BASE!")
        logger.error("❌ Des corrections sont nécessaires")
        return False

def get_missing_features_details():
    """Obtient les détails des features manquantes"""
    
    logger.info("\n🔍 DÉTAILS DES FEATURES MANQUANTES")
    logger.info("=" * 80)
    
    ml_features = get_ml_features()
    db_data = get_database_features()
    
    if not ml_features or not db_data:
        return
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires
    db_feature_names = {name for name, category in db_features}
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = set()
        chapter_db_features[chapter_id].add(feature_name)
    
    # Identifier les features manquantes
    all_missing = []
    
    for chapter_id, ml_chapter_features in ml_features.items():
        if chapter_id not in chapter_db_features:
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        for category, ml_features_list in ml_chapter_features.items():
            for feature in ml_features_list:
                if feature not in db_chapter_features:
                    all_missing.append((chapter_id, category, feature))
    
    if all_missing:
        logger.info("📋 Features manquantes par chapitre:")
        for chapter_id, category, feature in all_missing:
            logger.info(f"   - {chapter_id}.{category}: {feature}")
        
        logger.info(f"\n📊 Total features manquantes: {len(all_missing)}")
        
        # Grouper par catégorie
        by_category = {}
        for chapter_id, category, feature in all_missing:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(feature)
        
        logger.info("\n📋 Features manquantes par catégorie:")
        for category, features in by_category.items():
            logger.info(f"   - {category}: {len(features)} features")
            for feature in features:
                logger.info(f"     * {feature}")
    else:
        logger.info("✅ Aucune feature manquante!")

if __name__ == "__main__":
    success = compare_features()
    if not success:
        get_missing_features_details()
        sys.exit(1)


"""

import sys
import logging
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ml_features():
    """Récupère toutes les features utilisées par les modèles ML"""
    try:
        from src.chapters.chap30.ml_model_advanced import Chap30MLAdvanced
        from src.chapters.chap84.ml_model_advanced import Chap84MLAdvanced
        from src.chapters.chap85.ml_model_advanced import Chap85MLAdvanced
        
        # Initialiser les modèles
        chap30 = Chap30MLAdvanced()
        chap84 = Chap84MLAdvanced()
        chap85 = Chap85MLAdvanced()
        
        # Récupérer les features
        features30 = chap30._get_feature_columns()
        features84 = chap84._get_feature_columns()
        features85 = chap85._get_feature_columns()
        
        return {
            'chap30': features30,
            'chap84': features84,
            'chap85': features85
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features ML: {e}")
        return {}

def get_database_features():
    """Récupère toutes les features en base de données"""
    try:
        from database_config import get_database_config
        import psycopg2
        
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Récupérer toutes les features
        cursor.execute("SELECT feature_name, feature_category FROM features")
        db_features = cursor.fetchall()
        
        # Récupérer les associations chapitres-features
        cursor.execute("""
            SELECT c.chapter_id, f.feature_name, f.feature_category
            FROM chapters c
            JOIN chapter_features cf ON c.chapter_id = cf.chapter_id
            JOIN features f ON cf.feature_id = f.feature_id
            ORDER BY c.chapter_id, f.feature_category, f.feature_name
        """)
        chapter_features = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            'all_features': db_features,
            'chapter_features': chapter_features
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features base: {e}")
        return {}

def compare_features():
    """Compare les features ML avec ceux en base"""
    
    logger.info("🔍 VÉRIFICATION COMPLÈTE DES FEATURES ML")
    logger.info("=" * 80)
    
    # Récupérer les features ML
    ml_features = get_ml_features()
    if not ml_features:
        logger.error("❌ Impossible de récupérer les features ML")
        return False
    
    # Récupérer les features en base
    db_data = get_database_features()
    if not db_data:
        logger.error("❌ Impossible de récupérer les features en base")
        return False
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires pour faciliter la recherche
    db_feature_names = {name for name, category in db_features}
    
    # Créer un dictionnaire des features par chapitre
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = {}
        
        # Mapper les catégories de la base vers les catégories ML
        ml_category = feature_category
        if feature_category == 'basic':
            # Déterminer si c'est numeric ou categorical basé sur le nom
            if any(keyword in feature_name.upper() for keyword in ['VALEUR', 'POIDS', 'MONTANT', 'TAUX', 'RATIO', 'PRECISION']):
                ml_category = 'numeric'
            elif any(keyword in feature_name.upper() for keyword in ['CODE_', 'PAYS_', 'BUREAU', 'REGIME_', 'NUMERO_DPI']):
                ml_category = 'categorical'
            else:
                ml_category = 'numeric'  # Par défaut pour les autres features numériques
        
        if ml_category not in chapter_db_features[chapter_id]:
            chapter_db_features[chapter_id][ml_category] = set()
        chapter_db_features[chapter_id][ml_category].add(feature_name)
    
    all_good = True
    
    # Vérifier chaque chapitre
    for chapter_id, ml_chapter_features in ml_features.items():
        logger.info(f"\n📋 Vérification {chapter_id}:")
        logger.info("-" * 50)
        
        if chapter_id not in chapter_db_features:
            logger.error(f"❌ Chapitre {chapter_id} non trouvé en base")
            all_good = False
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        # Vérifier chaque catégorie
        for category, ml_features_list in ml_chapter_features.items():
            logger.info(f"\n   📊 Catégorie {category}:")
            
            if category not in db_chapter_features:
                logger.error(f"      ❌ Catégorie {category} manquante en base")
                all_good = False
                continue
            
            db_category_features = db_chapter_features[category]
            missing_features = []
            
            for feature in ml_features_list:
                if feature not in db_category_features:
                    missing_features.append(feature)
                    logger.error(f"      ❌ {feature} - MANQUANTE")
                else:
                    logger.info(f"      ✅ {feature}")
            
            if missing_features:
                logger.error(f"      📊 {len(missing_features)} features manquantes sur {len(ml_features_list)}")
                all_good = False
            else:
                logger.info(f"      📊 Toutes les {len(ml_features_list)} features sont présentes")
    
    # Résumé global
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ DE LA VÉRIFICATION")
    logger.info("=" * 80)
    
    total_ml_features = sum(len(features) for chapter_features in ml_features.values() 
                           for features in chapter_features.values())
    total_db_features = len(db_features)
    
    logger.info(f"📈 Total features ML: {total_ml_features}")
    logger.info(f"📈 Total features en base: {total_db_features}")
    
    if all_good:
        logger.info("🎉 TOUTES LES FEATURES ML SONT PRÉSENTES EN BASE!")
        logger.info("✅ Le système est cohérent")
        return True
    else:
        logger.error("💥 CERTAINES FEATURES ML SONT MANQUANTES EN BASE!")
        logger.error("❌ Des corrections sont nécessaires")
        return False

def get_missing_features_details():
    """Obtient les détails des features manquantes"""
    
    logger.info("\n🔍 DÉTAILS DES FEATURES MANQUANTES")
    logger.info("=" * 80)
    
    ml_features = get_ml_features()
    db_data = get_database_features()
    
    if not ml_features or not db_data:
        return
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires
    db_feature_names = {name for name, category in db_features}
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = set()
        chapter_db_features[chapter_id].add(feature_name)
    
    # Identifier les features manquantes
    all_missing = []
    
    for chapter_id, ml_chapter_features in ml_features.items():
        if chapter_id not in chapter_db_features:
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        for category, ml_features_list in ml_chapter_features.items():
            for feature in ml_features_list:
                if feature not in db_chapter_features:
                    all_missing.append((chapter_id, category, feature))
    
    if all_missing:
        logger.info("📋 Features manquantes par chapitre:")
        for chapter_id, category, feature in all_missing:
            logger.info(f"   - {chapter_id}.{category}: {feature}")
        
        logger.info(f"\n📊 Total features manquantes: {len(all_missing)}")
        
        # Grouper par catégorie
        by_category = {}
        for chapter_id, category, feature in all_missing:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(feature)
        
        logger.info("\n📋 Features manquantes par catégorie:")
        for category, features in by_category.items():
            logger.info(f"   - {category}: {len(features)} features")
            for feature in features:
                logger.info(f"     * {feature}")
    else:
        logger.info("✅ Aucune feature manquante!")

if __name__ == "__main__":
    success = compare_features()
    if not success:
        get_missing_features_details()
        sys.exit(1)


























"""

import sys
import logging
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ml_features():
    """Récupère toutes les features utilisées par les modèles ML"""
    try:
        from src.chapters.chap30.ml_model_advanced import Chap30MLAdvanced
        from src.chapters.chap84.ml_model_advanced import Chap84MLAdvanced
        from src.chapters.chap85.ml_model_advanced import Chap85MLAdvanced
        
        # Initialiser les modèles
        chap30 = Chap30MLAdvanced()
        chap84 = Chap84MLAdvanced()
        chap85 = Chap85MLAdvanced()
        
        # Récupérer les features
        features30 = chap30._get_feature_columns()
        features84 = chap84._get_feature_columns()
        features85 = chap85._get_feature_columns()
        
        return {
            'chap30': features30,
            'chap84': features84,
            'chap85': features85
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features ML: {e}")
        return {}

def get_database_features():
    """Récupère toutes les features en base de données"""
    try:
        from database_config import get_database_config
        import psycopg2
        
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Récupérer toutes les features
        cursor.execute("SELECT feature_name, feature_category FROM features")
        db_features = cursor.fetchall()
        
        # Récupérer les associations chapitres-features
        cursor.execute("""
            SELECT c.chapter_id, f.feature_name, f.feature_category
            FROM chapters c
            JOIN chapter_features cf ON c.chapter_id = cf.chapter_id
            JOIN features f ON cf.feature_id = f.feature_id
            ORDER BY c.chapter_id, f.feature_category, f.feature_name
        """)
        chapter_features = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            'all_features': db_features,
            'chapter_features': chapter_features
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features base: {e}")
        return {}

def compare_features():
    """Compare les features ML avec ceux en base"""
    
    logger.info("🔍 VÉRIFICATION COMPLÈTE DES FEATURES ML")
    logger.info("=" * 80)
    
    # Récupérer les features ML
    ml_features = get_ml_features()
    if not ml_features:
        logger.error("❌ Impossible de récupérer les features ML")
        return False
    
    # Récupérer les features en base
    db_data = get_database_features()
    if not db_data:
        logger.error("❌ Impossible de récupérer les features en base")
        return False
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires pour faciliter la recherche
    db_feature_names = {name for name, category in db_features}
    
    # Créer un dictionnaire des features par chapitre
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = {}
        
        # Mapper les catégories de la base vers les catégories ML
        ml_category = feature_category
        if feature_category == 'basic':
            # Déterminer si c'est numeric ou categorical basé sur le nom
            if any(keyword in feature_name.upper() for keyword in ['VALEUR', 'POIDS', 'MONTANT', 'TAUX', 'RATIO', 'PRECISION']):
                ml_category = 'numeric'
            elif any(keyword in feature_name.upper() for keyword in ['CODE_', 'PAYS_', 'BUREAU', 'REGIME_', 'NUMERO_DPI']):
                ml_category = 'categorical'
            else:
                ml_category = 'numeric'  # Par défaut pour les autres features numériques
        
        if ml_category not in chapter_db_features[chapter_id]:
            chapter_db_features[chapter_id][ml_category] = set()
        chapter_db_features[chapter_id][ml_category].add(feature_name)
    
    all_good = True
    
    # Vérifier chaque chapitre
    for chapter_id, ml_chapter_features in ml_features.items():
        logger.info(f"\n📋 Vérification {chapter_id}:")
        logger.info("-" * 50)
        
        if chapter_id not in chapter_db_features:
            logger.error(f"❌ Chapitre {chapter_id} non trouvé en base")
            all_good = False
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        # Vérifier chaque catégorie
        for category, ml_features_list in ml_chapter_features.items():
            logger.info(f"\n   📊 Catégorie {category}:")
            
            if category not in db_chapter_features:
                logger.error(f"      ❌ Catégorie {category} manquante en base")
                all_good = False
                continue
            
            db_category_features = db_chapter_features[category]
            missing_features = []
            
            for feature in ml_features_list:
                if feature not in db_category_features:
                    missing_features.append(feature)
                    logger.error(f"      ❌ {feature} - MANQUANTE")
                else:
                    logger.info(f"      ✅ {feature}")
            
            if missing_features:
                logger.error(f"      📊 {len(missing_features)} features manquantes sur {len(ml_features_list)}")
                all_good = False
            else:
                logger.info(f"      📊 Toutes les {len(ml_features_list)} features sont présentes")
    
    # Résumé global
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ DE LA VÉRIFICATION")
    logger.info("=" * 80)
    
    total_ml_features = sum(len(features) for chapter_features in ml_features.values() 
                           for features in chapter_features.values())
    total_db_features = len(db_features)
    
    logger.info(f"📈 Total features ML: {total_ml_features}")
    logger.info(f"📈 Total features en base: {total_db_features}")
    
    if all_good:
        logger.info("🎉 TOUTES LES FEATURES ML SONT PRÉSENTES EN BASE!")
        logger.info("✅ Le système est cohérent")
        return True
    else:
        logger.error("💥 CERTAINES FEATURES ML SONT MANQUANTES EN BASE!")
        logger.error("❌ Des corrections sont nécessaires")
        return False

def get_missing_features_details():
    """Obtient les détails des features manquantes"""
    
    logger.info("\n🔍 DÉTAILS DES FEATURES MANQUANTES")
    logger.info("=" * 80)
    
    ml_features = get_ml_features()
    db_data = get_database_features()
    
    if not ml_features or not db_data:
        return
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires
    db_feature_names = {name for name, category in db_features}
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = set()
        chapter_db_features[chapter_id].add(feature_name)
    
    # Identifier les features manquantes
    all_missing = []
    
    for chapter_id, ml_chapter_features in ml_features.items():
        if chapter_id not in chapter_db_features:
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        for category, ml_features_list in ml_chapter_features.items():
            for feature in ml_features_list:
                if feature not in db_chapter_features:
                    all_missing.append((chapter_id, category, feature))
    
    if all_missing:
        logger.info("📋 Features manquantes par chapitre:")
        for chapter_id, category, feature in all_missing:
            logger.info(f"   - {chapter_id}.{category}: {feature}")
        
        logger.info(f"\n📊 Total features manquantes: {len(all_missing)}")
        
        # Grouper par catégorie
        by_category = {}
        for chapter_id, category, feature in all_missing:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(feature)
        
        logger.info("\n📋 Features manquantes par catégorie:")
        for category, features in by_category.items():
            logger.info(f"   - {category}: {len(features)} features")
            for feature in features:
                logger.info(f"     * {feature}")
    else:
        logger.info("✅ Aucune feature manquante!")

if __name__ == "__main__":
    success = compare_features()
    if not success:
        get_missing_features_details()
        sys.exit(1)


"""

import sys
import logging
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ml_features():
    """Récupère toutes les features utilisées par les modèles ML"""
    try:
        from src.chapters.chap30.ml_model_advanced import Chap30MLAdvanced
        from src.chapters.chap84.ml_model_advanced import Chap84MLAdvanced
        from src.chapters.chap85.ml_model_advanced import Chap85MLAdvanced
        
        # Initialiser les modèles
        chap30 = Chap30MLAdvanced()
        chap84 = Chap84MLAdvanced()
        chap85 = Chap85MLAdvanced()
        
        # Récupérer les features
        features30 = chap30._get_feature_columns()
        features84 = chap84._get_feature_columns()
        features85 = chap85._get_feature_columns()
        
        return {
            'chap30': features30,
            'chap84': features84,
            'chap85': features85
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features ML: {e}")
        return {}

def get_database_features():
    """Récupère toutes les features en base de données"""
    try:
        from database_config import get_database_config
        import psycopg2
        
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Récupérer toutes les features
        cursor.execute("SELECT feature_name, feature_category FROM features")
        db_features = cursor.fetchall()
        
        # Récupérer les associations chapitres-features
        cursor.execute("""
            SELECT c.chapter_id, f.feature_name, f.feature_category
            FROM chapters c
            JOIN chapter_features cf ON c.chapter_id = cf.chapter_id
            JOIN features f ON cf.feature_id = f.feature_id
            ORDER BY c.chapter_id, f.feature_category, f.feature_name
        """)
        chapter_features = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            'all_features': db_features,
            'chapter_features': chapter_features
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features base: {e}")
        return {}

def compare_features():
    """Compare les features ML avec ceux en base"""
    
    logger.info("🔍 VÉRIFICATION COMPLÈTE DES FEATURES ML")
    logger.info("=" * 80)
    
    # Récupérer les features ML
    ml_features = get_ml_features()
    if not ml_features:
        logger.error("❌ Impossible de récupérer les features ML")
        return False
    
    # Récupérer les features en base
    db_data = get_database_features()
    if not db_data:
        logger.error("❌ Impossible de récupérer les features en base")
        return False
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires pour faciliter la recherche
    db_feature_names = {name for name, category in db_features}
    
    # Créer un dictionnaire des features par chapitre
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = {}
        
        # Mapper les catégories de la base vers les catégories ML
        ml_category = feature_category
        if feature_category == 'basic':
            # Déterminer si c'est numeric ou categorical basé sur le nom
            if any(keyword in feature_name.upper() for keyword in ['VALEUR', 'POIDS', 'MONTANT', 'TAUX', 'RATIO', 'PRECISION']):
                ml_category = 'numeric'
            elif any(keyword in feature_name.upper() for keyword in ['CODE_', 'PAYS_', 'BUREAU', 'REGIME_', 'NUMERO_DPI']):
                ml_category = 'categorical'
            else:
                ml_category = 'numeric'  # Par défaut pour les autres features numériques
        
        if ml_category not in chapter_db_features[chapter_id]:
            chapter_db_features[chapter_id][ml_category] = set()
        chapter_db_features[chapter_id][ml_category].add(feature_name)
    
    all_good = True
    
    # Vérifier chaque chapitre
    for chapter_id, ml_chapter_features in ml_features.items():
        logger.info(f"\n📋 Vérification {chapter_id}:")
        logger.info("-" * 50)
        
        if chapter_id not in chapter_db_features:
            logger.error(f"❌ Chapitre {chapter_id} non trouvé en base")
            all_good = False
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        # Vérifier chaque catégorie
        for category, ml_features_list in ml_chapter_features.items():
            logger.info(f"\n   📊 Catégorie {category}:")
            
            if category not in db_chapter_features:
                logger.error(f"      ❌ Catégorie {category} manquante en base")
                all_good = False
                continue
            
            db_category_features = db_chapter_features[category]
            missing_features = []
            
            for feature in ml_features_list:
                if feature not in db_category_features:
                    missing_features.append(feature)
                    logger.error(f"      ❌ {feature} - MANQUANTE")
                else:
                    logger.info(f"      ✅ {feature}")
            
            if missing_features:
                logger.error(f"      📊 {len(missing_features)} features manquantes sur {len(ml_features_list)}")
                all_good = False
            else:
                logger.info(f"      📊 Toutes les {len(ml_features_list)} features sont présentes")
    
    # Résumé global
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ DE LA VÉRIFICATION")
    logger.info("=" * 80)
    
    total_ml_features = sum(len(features) for chapter_features in ml_features.values() 
                           for features in chapter_features.values())
    total_db_features = len(db_features)
    
    logger.info(f"📈 Total features ML: {total_ml_features}")
    logger.info(f"📈 Total features en base: {total_db_features}")
    
    if all_good:
        logger.info("🎉 TOUTES LES FEATURES ML SONT PRÉSENTES EN BASE!")
        logger.info("✅ Le système est cohérent")
        return True
    else:
        logger.error("💥 CERTAINES FEATURES ML SONT MANQUANTES EN BASE!")
        logger.error("❌ Des corrections sont nécessaires")
        return False

def get_missing_features_details():
    """Obtient les détails des features manquantes"""
    
    logger.info("\n🔍 DÉTAILS DES FEATURES MANQUANTES")
    logger.info("=" * 80)
    
    ml_features = get_ml_features()
    db_data = get_database_features()
    
    if not ml_features or not db_data:
        return
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires
    db_feature_names = {name for name, category in db_features}
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = set()
        chapter_db_features[chapter_id].add(feature_name)
    
    # Identifier les features manquantes
    all_missing = []
    
    for chapter_id, ml_chapter_features in ml_features.items():
        if chapter_id not in chapter_db_features:
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        for category, ml_features_list in ml_chapter_features.items():
            for feature in ml_features_list:
                if feature not in db_chapter_features:
                    all_missing.append((chapter_id, category, feature))
    
    if all_missing:
        logger.info("📋 Features manquantes par chapitre:")
        for chapter_id, category, feature in all_missing:
            logger.info(f"   - {chapter_id}.{category}: {feature}")
        
        logger.info(f"\n📊 Total features manquantes: {len(all_missing)}")
        
        # Grouper par catégorie
        by_category = {}
        for chapter_id, category, feature in all_missing:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(feature)
        
        logger.info("\n📋 Features manquantes par catégorie:")
        for category, features in by_category.items():
            logger.info(f"   - {category}: {len(features)} features")
            for feature in features:
                logger.info(f"     * {feature}")
    else:
        logger.info("✅ Aucune feature manquante!")

if __name__ == "__main__":
    success = compare_features()
    if not success:
        get_missing_features_details()
        sys.exit(1)


























"""

import sys
import logging
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ml_features():
    """Récupère toutes les features utilisées par les modèles ML"""
    try:
        from src.chapters.chap30.ml_model_advanced import Chap30MLAdvanced
        from src.chapters.chap84.ml_model_advanced import Chap84MLAdvanced
        from src.chapters.chap85.ml_model_advanced import Chap85MLAdvanced
        
        # Initialiser les modèles
        chap30 = Chap30MLAdvanced()
        chap84 = Chap84MLAdvanced()
        chap85 = Chap85MLAdvanced()
        
        # Récupérer les features
        features30 = chap30._get_feature_columns()
        features84 = chap84._get_feature_columns()
        features85 = chap85._get_feature_columns()
        
        return {
            'chap30': features30,
            'chap84': features84,
            'chap85': features85
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features ML: {e}")
        return {}

def get_database_features():
    """Récupère toutes les features en base de données"""
    try:
        from database_config import get_database_config
        import psycopg2
        
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Récupérer toutes les features
        cursor.execute("SELECT feature_name, feature_category FROM features")
        db_features = cursor.fetchall()
        
        # Récupérer les associations chapitres-features
        cursor.execute("""
            SELECT c.chapter_id, f.feature_name, f.feature_category
            FROM chapters c
            JOIN chapter_features cf ON c.chapter_id = cf.chapter_id
            JOIN features f ON cf.feature_id = f.feature_id
            ORDER BY c.chapter_id, f.feature_category, f.feature_name
        """)
        chapter_features = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            'all_features': db_features,
            'chapter_features': chapter_features
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features base: {e}")
        return {}

def compare_features():
    """Compare les features ML avec ceux en base"""
    
    logger.info("🔍 VÉRIFICATION COMPLÈTE DES FEATURES ML")
    logger.info("=" * 80)
    
    # Récupérer les features ML
    ml_features = get_ml_features()
    if not ml_features:
        logger.error("❌ Impossible de récupérer les features ML")
        return False
    
    # Récupérer les features en base
    db_data = get_database_features()
    if not db_data:
        logger.error("❌ Impossible de récupérer les features en base")
        return False
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires pour faciliter la recherche
    db_feature_names = {name for name, category in db_features}
    
    # Créer un dictionnaire des features par chapitre
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = {}
        
        # Mapper les catégories de la base vers les catégories ML
        ml_category = feature_category
        if feature_category == 'basic':
            # Déterminer si c'est numeric ou categorical basé sur le nom
            if any(keyword in feature_name.upper() for keyword in ['VALEUR', 'POIDS', 'MONTANT', 'TAUX', 'RATIO', 'PRECISION']):
                ml_category = 'numeric'
            elif any(keyword in feature_name.upper() for keyword in ['CODE_', 'PAYS_', 'BUREAU', 'REGIME_', 'NUMERO_DPI']):
                ml_category = 'categorical'
            else:
                ml_category = 'numeric'  # Par défaut pour les autres features numériques
        
        if ml_category not in chapter_db_features[chapter_id]:
            chapter_db_features[chapter_id][ml_category] = set()
        chapter_db_features[chapter_id][ml_category].add(feature_name)
    
    all_good = True
    
    # Vérifier chaque chapitre
    for chapter_id, ml_chapter_features in ml_features.items():
        logger.info(f"\n📋 Vérification {chapter_id}:")
        logger.info("-" * 50)
        
        if chapter_id not in chapter_db_features:
            logger.error(f"❌ Chapitre {chapter_id} non trouvé en base")
            all_good = False
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        # Vérifier chaque catégorie
        for category, ml_features_list in ml_chapter_features.items():
            logger.info(f"\n   📊 Catégorie {category}:")
            
            if category not in db_chapter_features:
                logger.error(f"      ❌ Catégorie {category} manquante en base")
                all_good = False
                continue
            
            db_category_features = db_chapter_features[category]
            missing_features = []
            
            for feature in ml_features_list:
                if feature not in db_category_features:
                    missing_features.append(feature)
                    logger.error(f"      ❌ {feature} - MANQUANTE")
                else:
                    logger.info(f"      ✅ {feature}")
            
            if missing_features:
                logger.error(f"      📊 {len(missing_features)} features manquantes sur {len(ml_features_list)}")
                all_good = False
            else:
                logger.info(f"      📊 Toutes les {len(ml_features_list)} features sont présentes")
    
    # Résumé global
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ DE LA VÉRIFICATION")
    logger.info("=" * 80)
    
    total_ml_features = sum(len(features) for chapter_features in ml_features.values() 
                           for features in chapter_features.values())
    total_db_features = len(db_features)
    
    logger.info(f"📈 Total features ML: {total_ml_features}")
    logger.info(f"📈 Total features en base: {total_db_features}")
    
    if all_good:
        logger.info("🎉 TOUTES LES FEATURES ML SONT PRÉSENTES EN BASE!")
        logger.info("✅ Le système est cohérent")
        return True
    else:
        logger.error("💥 CERTAINES FEATURES ML SONT MANQUANTES EN BASE!")
        logger.error("❌ Des corrections sont nécessaires")
        return False

def get_missing_features_details():
    """Obtient les détails des features manquantes"""
    
    logger.info("\n🔍 DÉTAILS DES FEATURES MANQUANTES")
    logger.info("=" * 80)
    
    ml_features = get_ml_features()
    db_data = get_database_features()
    
    if not ml_features or not db_data:
        return
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires
    db_feature_names = {name for name, category in db_features}
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = set()
        chapter_db_features[chapter_id].add(feature_name)
    
    # Identifier les features manquantes
    all_missing = []
    
    for chapter_id, ml_chapter_features in ml_features.items():
        if chapter_id not in chapter_db_features:
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        for category, ml_features_list in ml_chapter_features.items():
            for feature in ml_features_list:
                if feature not in db_chapter_features:
                    all_missing.append((chapter_id, category, feature))
    
    if all_missing:
        logger.info("📋 Features manquantes par chapitre:")
        for chapter_id, category, feature in all_missing:
            logger.info(f"   - {chapter_id}.{category}: {feature}")
        
        logger.info(f"\n📊 Total features manquantes: {len(all_missing)}")
        
        # Grouper par catégorie
        by_category = {}
        for chapter_id, category, feature in all_missing:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(feature)
        
        logger.info("\n📋 Features manquantes par catégorie:")
        for category, features in by_category.items():
            logger.info(f"   - {category}: {len(features)} features")
            for feature in features:
                logger.info(f"     * {feature}")
    else:
        logger.info("✅ Aucune feature manquante!")

if __name__ == "__main__":
    success = compare_features()
    if not success:
        get_missing_features_details()
        sys.exit(1)


"""

import sys
import logging
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ml_features():
    """Récupère toutes les features utilisées par les modèles ML"""
    try:
        from src.chapters.chap30.ml_model_advanced import Chap30MLAdvanced
        from src.chapters.chap84.ml_model_advanced import Chap84MLAdvanced
        from src.chapters.chap85.ml_model_advanced import Chap85MLAdvanced
        
        # Initialiser les modèles
        chap30 = Chap30MLAdvanced()
        chap84 = Chap84MLAdvanced()
        chap85 = Chap85MLAdvanced()
        
        # Récupérer les features
        features30 = chap30._get_feature_columns()
        features84 = chap84._get_feature_columns()
        features85 = chap85._get_feature_columns()
        
        return {
            'chap30': features30,
            'chap84': features84,
            'chap85': features85
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features ML: {e}")
        return {}

def get_database_features():
    """Récupère toutes les features en base de données"""
    try:
        from database_config import get_database_config
        import psycopg2
        
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Récupérer toutes les features
        cursor.execute("SELECT feature_name, feature_category FROM features")
        db_features = cursor.fetchall()
        
        # Récupérer les associations chapitres-features
        cursor.execute("""
            SELECT c.chapter_id, f.feature_name, f.feature_category
            FROM chapters c
            JOIN chapter_features cf ON c.chapter_id = cf.chapter_id
            JOIN features f ON cf.feature_id = f.feature_id
            ORDER BY c.chapter_id, f.feature_category, f.feature_name
        """)
        chapter_features = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            'all_features': db_features,
            'chapter_features': chapter_features
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features base: {e}")
        return {}

def compare_features():
    """Compare les features ML avec ceux en base"""
    
    logger.info("🔍 VÉRIFICATION COMPLÈTE DES FEATURES ML")
    logger.info("=" * 80)
    
    # Récupérer les features ML
    ml_features = get_ml_features()
    if not ml_features:
        logger.error("❌ Impossible de récupérer les features ML")
        return False
    
    # Récupérer les features en base
    db_data = get_database_features()
    if not db_data:
        logger.error("❌ Impossible de récupérer les features en base")
        return False
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires pour faciliter la recherche
    db_feature_names = {name for name, category in db_features}
    
    # Créer un dictionnaire des features par chapitre
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = {}
        
        # Mapper les catégories de la base vers les catégories ML
        ml_category = feature_category
        if feature_category == 'basic':
            # Déterminer si c'est numeric ou categorical basé sur le nom
            if any(keyword in feature_name.upper() for keyword in ['VALEUR', 'POIDS', 'MONTANT', 'TAUX', 'RATIO', 'PRECISION']):
                ml_category = 'numeric'
            elif any(keyword in feature_name.upper() for keyword in ['CODE_', 'PAYS_', 'BUREAU', 'REGIME_', 'NUMERO_DPI']):
                ml_category = 'categorical'
            else:
                ml_category = 'numeric'  # Par défaut pour les autres features numériques
        
        if ml_category not in chapter_db_features[chapter_id]:
            chapter_db_features[chapter_id][ml_category] = set()
        chapter_db_features[chapter_id][ml_category].add(feature_name)
    
    all_good = True
    
    # Vérifier chaque chapitre
    for chapter_id, ml_chapter_features in ml_features.items():
        logger.info(f"\n📋 Vérification {chapter_id}:")
        logger.info("-" * 50)
        
        if chapter_id not in chapter_db_features:
            logger.error(f"❌ Chapitre {chapter_id} non trouvé en base")
            all_good = False
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        # Vérifier chaque catégorie
        for category, ml_features_list in ml_chapter_features.items():
            logger.info(f"\n   📊 Catégorie {category}:")
            
            if category not in db_chapter_features:
                logger.error(f"      ❌ Catégorie {category} manquante en base")
                all_good = False
                continue
            
            db_category_features = db_chapter_features[category]
            missing_features = []
            
            for feature in ml_features_list:
                if feature not in db_category_features:
                    missing_features.append(feature)
                    logger.error(f"      ❌ {feature} - MANQUANTE")
                else:
                    logger.info(f"      ✅ {feature}")
            
            if missing_features:
                logger.error(f"      📊 {len(missing_features)} features manquantes sur {len(ml_features_list)}")
                all_good = False
            else:
                logger.info(f"      📊 Toutes les {len(ml_features_list)} features sont présentes")
    
    # Résumé global
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ DE LA VÉRIFICATION")
    logger.info("=" * 80)
    
    total_ml_features = sum(len(features) for chapter_features in ml_features.values() 
                           for features in chapter_features.values())
    total_db_features = len(db_features)
    
    logger.info(f"📈 Total features ML: {total_ml_features}")
    logger.info(f"📈 Total features en base: {total_db_features}")
    
    if all_good:
        logger.info("🎉 TOUTES LES FEATURES ML SONT PRÉSENTES EN BASE!")
        logger.info("✅ Le système est cohérent")
        return True
    else:
        logger.error("💥 CERTAINES FEATURES ML SONT MANQUANTES EN BASE!")
        logger.error("❌ Des corrections sont nécessaires")
        return False

def get_missing_features_details():
    """Obtient les détails des features manquantes"""
    
    logger.info("\n🔍 DÉTAILS DES FEATURES MANQUANTES")
    logger.info("=" * 80)
    
    ml_features = get_ml_features()
    db_data = get_database_features()
    
    if not ml_features or not db_data:
        return
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires
    db_feature_names = {name for name, category in db_features}
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = set()
        chapter_db_features[chapter_id].add(feature_name)
    
    # Identifier les features manquantes
    all_missing = []
    
    for chapter_id, ml_chapter_features in ml_features.items():
        if chapter_id not in chapter_db_features:
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        for category, ml_features_list in ml_chapter_features.items():
            for feature in ml_features_list:
                if feature not in db_chapter_features:
                    all_missing.append((chapter_id, category, feature))
    
    if all_missing:
        logger.info("📋 Features manquantes par chapitre:")
        for chapter_id, category, feature in all_missing:
            logger.info(f"   - {chapter_id}.{category}: {feature}")
        
        logger.info(f"\n📊 Total features manquantes: {len(all_missing)}")
        
        # Grouper par catégorie
        by_category = {}
        for chapter_id, category, feature in all_missing:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(feature)
        
        logger.info("\n📋 Features manquantes par catégorie:")
        for category, features in by_category.items():
            logger.info(f"   - {category}: {len(features)} features")
            for feature in features:
                logger.info(f"     * {feature}")
    else:
        logger.info("✅ Aucune feature manquante!")

if __name__ == "__main__":
    success = compare_features()
    if not success:
        get_missing_features_details()
        sys.exit(1)


























"""

import sys
import logging
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ml_features():
    """Récupère toutes les features utilisées par les modèles ML"""
    try:
        from src.chapters.chap30.ml_model_advanced import Chap30MLAdvanced
        from src.chapters.chap84.ml_model_advanced import Chap84MLAdvanced
        from src.chapters.chap85.ml_model_advanced import Chap85MLAdvanced
        
        # Initialiser les modèles
        chap30 = Chap30MLAdvanced()
        chap84 = Chap84MLAdvanced()
        chap85 = Chap85MLAdvanced()
        
        # Récupérer les features
        features30 = chap30._get_feature_columns()
        features84 = chap84._get_feature_columns()
        features85 = chap85._get_feature_columns()
        
        return {
            'chap30': features30,
            'chap84': features84,
            'chap85': features85
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features ML: {e}")
        return {}

def get_database_features():
    """Récupère toutes les features en base de données"""
    try:
        from database_config import get_database_config
        import psycopg2
        
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Récupérer toutes les features
        cursor.execute("SELECT feature_name, feature_category FROM features")
        db_features = cursor.fetchall()
        
        # Récupérer les associations chapitres-features
        cursor.execute("""
            SELECT c.chapter_id, f.feature_name, f.feature_category
            FROM chapters c
            JOIN chapter_features cf ON c.chapter_id = cf.chapter_id
            JOIN features f ON cf.feature_id = f.feature_id
            ORDER BY c.chapter_id, f.feature_category, f.feature_name
        """)
        chapter_features = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            'all_features': db_features,
            'chapter_features': chapter_features
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features base: {e}")
        return {}

def compare_features():
    """Compare les features ML avec ceux en base"""
    
    logger.info("🔍 VÉRIFICATION COMPLÈTE DES FEATURES ML")
    logger.info("=" * 80)
    
    # Récupérer les features ML
    ml_features = get_ml_features()
    if not ml_features:
        logger.error("❌ Impossible de récupérer les features ML")
        return False
    
    # Récupérer les features en base
    db_data = get_database_features()
    if not db_data:
        logger.error("❌ Impossible de récupérer les features en base")
        return False
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires pour faciliter la recherche
    db_feature_names = {name for name, category in db_features}
    
    # Créer un dictionnaire des features par chapitre
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = {}
        
        # Mapper les catégories de la base vers les catégories ML
        ml_category = feature_category
        if feature_category == 'basic':
            # Déterminer si c'est numeric ou categorical basé sur le nom
            if any(keyword in feature_name.upper() for keyword in ['VALEUR', 'POIDS', 'MONTANT', 'TAUX', 'RATIO', 'PRECISION']):
                ml_category = 'numeric'
            elif any(keyword in feature_name.upper() for keyword in ['CODE_', 'PAYS_', 'BUREAU', 'REGIME_', 'NUMERO_DPI']):
                ml_category = 'categorical'
            else:
                ml_category = 'numeric'  # Par défaut pour les autres features numériques
        
        if ml_category not in chapter_db_features[chapter_id]:
            chapter_db_features[chapter_id][ml_category] = set()
        chapter_db_features[chapter_id][ml_category].add(feature_name)
    
    all_good = True
    
    # Vérifier chaque chapitre
    for chapter_id, ml_chapter_features in ml_features.items():
        logger.info(f"\n📋 Vérification {chapter_id}:")
        logger.info("-" * 50)
        
        if chapter_id not in chapter_db_features:
            logger.error(f"❌ Chapitre {chapter_id} non trouvé en base")
            all_good = False
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        # Vérifier chaque catégorie
        for category, ml_features_list in ml_chapter_features.items():
            logger.info(f"\n   📊 Catégorie {category}:")
            
            if category not in db_chapter_features:
                logger.error(f"      ❌ Catégorie {category} manquante en base")
                all_good = False
                continue
            
            db_category_features = db_chapter_features[category]
            missing_features = []
            
            for feature in ml_features_list:
                if feature not in db_category_features:
                    missing_features.append(feature)
                    logger.error(f"      ❌ {feature} - MANQUANTE")
                else:
                    logger.info(f"      ✅ {feature}")
            
            if missing_features:
                logger.error(f"      📊 {len(missing_features)} features manquantes sur {len(ml_features_list)}")
                all_good = False
            else:
                logger.info(f"      📊 Toutes les {len(ml_features_list)} features sont présentes")
    
    # Résumé global
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ DE LA VÉRIFICATION")
    logger.info("=" * 80)
    
    total_ml_features = sum(len(features) for chapter_features in ml_features.values() 
                           for features in chapter_features.values())
    total_db_features = len(db_features)
    
    logger.info(f"📈 Total features ML: {total_ml_features}")
    logger.info(f"📈 Total features en base: {total_db_features}")
    
    if all_good:
        logger.info("🎉 TOUTES LES FEATURES ML SONT PRÉSENTES EN BASE!")
        logger.info("✅ Le système est cohérent")
        return True
    else:
        logger.error("💥 CERTAINES FEATURES ML SONT MANQUANTES EN BASE!")
        logger.error("❌ Des corrections sont nécessaires")
        return False

def get_missing_features_details():
    """Obtient les détails des features manquantes"""
    
    logger.info("\n🔍 DÉTAILS DES FEATURES MANQUANTES")
    logger.info("=" * 80)
    
    ml_features = get_ml_features()
    db_data = get_database_features()
    
    if not ml_features or not db_data:
        return
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires
    db_feature_names = {name for name, category in db_features}
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = set()
        chapter_db_features[chapter_id].add(feature_name)
    
    # Identifier les features manquantes
    all_missing = []
    
    for chapter_id, ml_chapter_features in ml_features.items():
        if chapter_id not in chapter_db_features:
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        for category, ml_features_list in ml_chapter_features.items():
            for feature in ml_features_list:
                if feature not in db_chapter_features:
                    all_missing.append((chapter_id, category, feature))
    
    if all_missing:
        logger.info("📋 Features manquantes par chapitre:")
        for chapter_id, category, feature in all_missing:
            logger.info(f"   - {chapter_id}.{category}: {feature}")
        
        logger.info(f"\n📊 Total features manquantes: {len(all_missing)}")
        
        # Grouper par catégorie
        by_category = {}
        for chapter_id, category, feature in all_missing:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(feature)
        
        logger.info("\n📋 Features manquantes par catégorie:")
        for category, features in by_category.items():
            logger.info(f"   - {category}: {len(features)} features")
            for feature in features:
                logger.info(f"     * {feature}")
    else:
        logger.info("✅ Aucune feature manquante!")

if __name__ == "__main__":
    success = compare_features()
    if not success:
        get_missing_features_details()
        sys.exit(1)


"""

import sys
import logging
from pathlib import Path

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_ml_features():
    """Récupère toutes les features utilisées par les modèles ML"""
    try:
        from src.chapters.chap30.ml_model_advanced import Chap30MLAdvanced
        from src.chapters.chap84.ml_model_advanced import Chap84MLAdvanced
        from src.chapters.chap85.ml_model_advanced import Chap85MLAdvanced
        
        # Initialiser les modèles
        chap30 = Chap30MLAdvanced()
        chap84 = Chap84MLAdvanced()
        chap85 = Chap85MLAdvanced()
        
        # Récupérer les features
        features30 = chap30._get_feature_columns()
        features84 = chap84._get_feature_columns()
        features85 = chap85._get_feature_columns()
        
        return {
            'chap30': features30,
            'chap84': features84,
            'chap85': features85
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features ML: {e}")
        return {}

def get_database_features():
    """Récupère toutes les features en base de données"""
    try:
        from database_config import get_database_config
        import psycopg2
        
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Récupérer toutes les features
        cursor.execute("SELECT feature_name, feature_category FROM features")
        db_features = cursor.fetchall()
        
        # Récupérer les associations chapitres-features
        cursor.execute("""
            SELECT c.chapter_id, f.feature_name, f.feature_category
            FROM chapters c
            JOIN chapter_features cf ON c.chapter_id = cf.chapter_id
            JOIN features f ON cf.feature_id = f.feature_id
            ORDER BY c.chapter_id, f.feature_category, f.feature_name
        """)
        chapter_features = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            'all_features': db_features,
            'chapter_features': chapter_features
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération features base: {e}")
        return {}

def compare_features():
    """Compare les features ML avec ceux en base"""
    
    logger.info("🔍 VÉRIFICATION COMPLÈTE DES FEATURES ML")
    logger.info("=" * 80)
    
    # Récupérer les features ML
    ml_features = get_ml_features()
    if not ml_features:
        logger.error("❌ Impossible de récupérer les features ML")
        return False
    
    # Récupérer les features en base
    db_data = get_database_features()
    if not db_data:
        logger.error("❌ Impossible de récupérer les features en base")
        return False
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires pour faciliter la recherche
    db_feature_names = {name for name, category in db_features}
    
    # Créer un dictionnaire des features par chapitre
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = {}
        
        # Mapper les catégories de la base vers les catégories ML
        ml_category = feature_category
        if feature_category == 'basic':
            # Déterminer si c'est numeric ou categorical basé sur le nom
            if any(keyword in feature_name.upper() for keyword in ['VALEUR', 'POIDS', 'MONTANT', 'TAUX', 'RATIO', 'PRECISION']):
                ml_category = 'numeric'
            elif any(keyword in feature_name.upper() for keyword in ['CODE_', 'PAYS_', 'BUREAU', 'REGIME_', 'NUMERO_DPI']):
                ml_category = 'categorical'
            else:
                ml_category = 'numeric'  # Par défaut pour les autres features numériques
        
        if ml_category not in chapter_db_features[chapter_id]:
            chapter_db_features[chapter_id][ml_category] = set()
        chapter_db_features[chapter_id][ml_category].add(feature_name)
    
    all_good = True
    
    # Vérifier chaque chapitre
    for chapter_id, ml_chapter_features in ml_features.items():
        logger.info(f"\n📋 Vérification {chapter_id}:")
        logger.info("-" * 50)
        
        if chapter_id not in chapter_db_features:
            logger.error(f"❌ Chapitre {chapter_id} non trouvé en base")
            all_good = False
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        # Vérifier chaque catégorie
        for category, ml_features_list in ml_chapter_features.items():
            logger.info(f"\n   📊 Catégorie {category}:")
            
            if category not in db_chapter_features:
                logger.error(f"      ❌ Catégorie {category} manquante en base")
                all_good = False
                continue
            
            db_category_features = db_chapter_features[category]
            missing_features = []
            
            for feature in ml_features_list:
                if feature not in db_category_features:
                    missing_features.append(feature)
                    logger.error(f"      ❌ {feature} - MANQUANTE")
                else:
                    logger.info(f"      ✅ {feature}")
            
            if missing_features:
                logger.error(f"      📊 {len(missing_features)} features manquantes sur {len(ml_features_list)}")
                all_good = False
            else:
                logger.info(f"      📊 Toutes les {len(ml_features_list)} features sont présentes")
    
    # Résumé global
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSUMÉ DE LA VÉRIFICATION")
    logger.info("=" * 80)
    
    total_ml_features = sum(len(features) for chapter_features in ml_features.values() 
                           for features in chapter_features.values())
    total_db_features = len(db_features)
    
    logger.info(f"📈 Total features ML: {total_ml_features}")
    logger.info(f"📈 Total features en base: {total_db_features}")
    
    if all_good:
        logger.info("🎉 TOUTES LES FEATURES ML SONT PRÉSENTES EN BASE!")
        logger.info("✅ Le système est cohérent")
        return True
    else:
        logger.error("💥 CERTAINES FEATURES ML SONT MANQUANTES EN BASE!")
        logger.error("❌ Des corrections sont nécessaires")
        return False

def get_missing_features_details():
    """Obtient les détails des features manquantes"""
    
    logger.info("\n🔍 DÉTAILS DES FEATURES MANQUANTES")
    logger.info("=" * 80)
    
    ml_features = get_ml_features()
    db_data = get_database_features()
    
    if not ml_features or not db_data:
        return
    
    db_features = db_data['all_features']
    chapter_features = db_data['chapter_features']
    
    # Créer des dictionnaires
    db_feature_names = {name for name, category in db_features}
    chapter_db_features = {}
    for chapter_id, feature_name, feature_category in chapter_features:
        if chapter_id not in chapter_db_features:
            chapter_db_features[chapter_id] = set()
        chapter_db_features[chapter_id].add(feature_name)
    
    # Identifier les features manquantes
    all_missing = []
    
    for chapter_id, ml_chapter_features in ml_features.items():
        if chapter_id not in chapter_db_features:
            continue
        
        db_chapter_features = chapter_db_features[chapter_id]
        
        for category, ml_features_list in ml_chapter_features.items():
            for feature in ml_features_list:
                if feature not in db_chapter_features:
                    all_missing.append((chapter_id, category, feature))
    
    if all_missing:
        logger.info("📋 Features manquantes par chapitre:")
        for chapter_id, category, feature in all_missing:
            logger.info(f"   - {chapter_id}.{category}: {feature}")
        
        logger.info(f"\n📊 Total features manquantes: {len(all_missing)}")
        
        # Grouper par catégorie
        by_category = {}
        for chapter_id, category, feature in all_missing:
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(feature)
        
        logger.info("\n📋 Features manquantes par catégorie:")
        for category, features in by_category.items():
            logger.info(f"   - {category}: {len(features)} features")
            for feature in features:
                logger.info(f"     * {feature}")
    else:
        logger.info("✅ Aucune feature manquante!")

if __name__ == "__main__":
    success = compare_features()
    if not success:
        get_missing_features_details()
        sys.exit(1)

























