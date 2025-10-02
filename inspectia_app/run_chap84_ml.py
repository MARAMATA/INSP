#!/usr/bin/env python3
"""
Script pour lancer le pipeline ML du chapitre 84 avec gestion optimisée de la mémoire
"""

import sys
import gc
import logging

# Ajouter le chemin du backend
sys.path.append('backend/src')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Fonction principale"""
    try:
        logger.info("🚀 DÉMARRAGE DU PIPELINE ML CHAPITRE 84")
        logger.info("=" * 60)
        
        # Importer et créer le pipeline ML
        from chapters.chap84.ml_model_advanced import Chap84MLAdvanced
        ml_pipeline = Chap84MLAdvanced()
        
        # Exécuter le pipeline ML complet
        logger.info("📋 Avec TOUTES les données préprocessées (264,494 échantillons)")
        logger.info("🔧 ASTUCES d'optimisation mémoire appliquées")
        
        ml_pipeline.run_complete_ml_pipeline()
        
        logger.info("=" * 60)
        logger.info("✅ PIPELINE ML TERMINÉ AVEC SUCCÈS !")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du pipeline ML: {e}")
        raise
    finally:
        # Nettoyer la mémoire
        gc.collect()
        logger.info("🧹 Mémoire nettoyée")

if __name__ == "__main__":
    main()





Script pour lancer le pipeline ML du chapitre 84 avec gestion optimisée de la mémoire
"""

import sys
import gc
import logging

# Ajouter le chemin du backend
sys.path.append('backend/src')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Fonction principale"""
    try:
        logger.info("🚀 DÉMARRAGE DU PIPELINE ML CHAPITRE 84")
        logger.info("=" * 60)
        
        # Importer et créer le pipeline ML
        from chapters.chap84.ml_model_advanced import Chap84MLAdvanced
        ml_pipeline = Chap84MLAdvanced()
        
        # Exécuter le pipeline ML complet
        logger.info("📋 Avec TOUTES les données préprocessées (264,494 échantillons)")
        logger.info("🔧 ASTUCES d'optimisation mémoire appliquées")
        
        ml_pipeline.run_complete_ml_pipeline()
        
        logger.info("=" * 60)
        logger.info("✅ PIPELINE ML TERMINÉ AVEC SUCCÈS !")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du pipeline ML: {e}")
        raise
    finally:
        # Nettoyer la mémoire
        gc.collect()
        logger.info("🧹 Mémoire nettoyée")

if __name__ == "__main__":
    main()





Script pour lancer le pipeline ML du chapitre 84 avec gestion optimisée de la mémoire
"""

import sys
import gc
import logging

# Ajouter le chemin du backend
sys.path.append('backend/src')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Fonction principale"""
    try:
        logger.info("🚀 DÉMARRAGE DU PIPELINE ML CHAPITRE 84")
        logger.info("=" * 60)
        
        # Importer et créer le pipeline ML
        from chapters.chap84.ml_model_advanced import Chap84MLAdvanced
        ml_pipeline = Chap84MLAdvanced()
        
        # Exécuter le pipeline ML complet
        logger.info("📋 Avec TOUTES les données préprocessées (264,494 échantillons)")
        logger.info("🔧 ASTUCES d'optimisation mémoire appliquées")
        
        ml_pipeline.run_complete_ml_pipeline()
        
        logger.info("=" * 60)
        logger.info("✅ PIPELINE ML TERMINÉ AVEC SUCCÈS !")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du pipeline ML: {e}")
        raise
    finally:
        # Nettoyer la mémoire
        gc.collect()
        logger.info("🧹 Mémoire nettoyée")

if __name__ == "__main__":
    main()





Script pour lancer le pipeline ML du chapitre 84 avec gestion optimisée de la mémoire
"""

import sys
import gc
import logging

# Ajouter le chemin du backend
sys.path.append('backend/src')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Fonction principale"""
    try:
        logger.info("🚀 DÉMARRAGE DU PIPELINE ML CHAPITRE 84")
        logger.info("=" * 60)
        
        # Importer et créer le pipeline ML
        from chapters.chap84.ml_model_advanced import Chap84MLAdvanced
        ml_pipeline = Chap84MLAdvanced()
        
        # Exécuter le pipeline ML complet
        logger.info("📋 Avec TOUTES les données préprocessées (264,494 échantillons)")
        logger.info("🔧 ASTUCES d'optimisation mémoire appliquées")
        
        ml_pipeline.run_complete_ml_pipeline()
        
        logger.info("=" * 60)
        logger.info("✅ PIPELINE ML TERMINÉ AVEC SUCCÈS !")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du pipeline ML: {e}")
        raise
    finally:
        # Nettoyer la mémoire
        gc.collect()
        logger.info("🧹 Mémoire nettoyée")

if __name__ == "__main__":
    main()























