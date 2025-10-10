"""
Script pour tester le stockage des prédictions et décisions RL
"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_predictions_storage():
    """Teste le stockage des prédictions et décisions RL"""
    
    logger.info("🧪 TEST DU STOCKAGE DES PRÉDICTIONS ET RL")
    logger.info("=" * 80)
    
    try:
        from database_config import get_database_config
        import psycopg2
        
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # ========================================
        # 1. TEST DES DÉCLARATIONS (DOIT ÊTRE EN PREMIER)
        # ========================================
        logger.info("\n📊 1. TEST DES DÉCLARATIONS")
        logger.info("-" * 50)
        
        # Créer des déclarations de test
        test_declarations = [
            {
                'declaration_id': '2024/ABJ/12345',
                'chapter_id': 'chap30',
                'file_name': 'test_chap30.csv',
                'file_type': 'csv',
                'source_type': 'api',
                'annee': 2024,
                'bureau': 'ABJ',
                'numero_declaration': '12345',
                'valeur_caf': 50000,
                'poids_net': 100,
                'extraction_status': 'success',
                'validation_status': 'valid'
            },
            {
                'declaration_id': '2024/DKR/67890',
                'chapter_id': 'chap84',
                'file_name': 'test_chap84.csv',
                'file_type': 'csv',
                'source_type': 'api',
                'annee': 2024,
                'bureau': 'DKR',
                'numero_declaration': '67890',
                'valeur_caf': 15000,
                'poids_net': 500,
                'extraction_status': 'success',
                'validation_status': 'valid'
            },
            {
                'declaration_id': '2024/ABJ/54321',
                'chapter_id': 'chap85',
                'file_name': 'test_chap85.csv',
                'file_type': 'csv',
                'source_type': 'api',
                'annee': 2024,
                'bureau': 'ABJ',
                'numero_declaration': '54321',
                'valeur_caf': 25000,
                'poids_net': 50,
                'extraction_status': 'success',
                'validation_status': 'valid'
            }
        ]
        
        # Insérer les déclarations de test
        for decl in test_declarations:
            try:
                cursor.execute("""
                    INSERT INTO declarations (
                        declaration_id, chapter_id, file_name, file_type, source_type,
                        annee, bureau, numero_declaration, valeur_caf, poids_net,
                        extraction_status, validation_status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    decl['declaration_id'], decl['chapter_id'], decl['file_name'],
                    decl['file_type'], decl['source_type'], decl['annee'],
                    decl['bureau'], decl['numero_declaration'], decl['valeur_caf'],
                    decl['poids_net'], decl['extraction_status'], decl['validation_status']
                ))
                logger.info(f"✅ Déclaration insérée: {decl['declaration_id']} ({decl['chapter_id']})")
            except Exception as e:
                logger.error(f"❌ Erreur insertion déclaration {decl['declaration_id']}: {e}")
        
        # ========================================
        # 2. TEST DES PRÉDICTIONS ML
        # ========================================
        logger.info("\n📊 2. TEST DES PRÉDICTIONS ML")
        logger.info("-" * 50)
        
        # Créer des données de test pour les prédictions
        test_predictions = [
            {
                'declaration_id': '2024/ABJ/12345',
                'chapter_id': 'chap30',
                'predicted_fraud': True,
                'fraud_probability': 0.85,
                'confidence_score': 0.92,
                'decision': 'fraude',
                'model_used': 'XGBoost_Chap30_Best',
                'optimal_threshold_used': 0.35,
                'auc_score': 0.9995,
                'f1_score': 0.9796,
                'precision_score': 0.9889,
                'recall_score': 0.9705,
                'ml_integration_used': True,
                'decision_source': 'ml',
                'context_features': {
                    'VALEUR_CAF': 50000,
                    'POIDS_NET': 100,
                    'BUSINESS_GLISSEMENT_TARIFAIRE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.95
                },
                'risk_analysis': {
                    'risk_level': 'high',
                    'fraud_indicators': ['high_value', 'cosmetic_shift', 'statistical_anomaly']
                }
            },
            {
                'declaration_id': '2024/DKR/67890',
                'chapter_id': 'chap84',
                'predicted_fraud': False,
                'fraud_probability': 0.15,
                'confidence_score': 0.88,
                'decision': 'conforme',
                'model_used': 'XGBoost_Chap84_Best',
                'optimal_threshold_used': 0.22,
                'auc_score': 0.9997,
                'f1_score': 0.9888,
                'precision_score': 0.9992,
                'recall_score': 0.9834,
                'ml_integration_used': True,
                'decision_source': 'ml',
                'context_features': {
                    'VALEUR_CAF': 15000,
                    'POIDS_NET': 500,
                    'BUSINESS_GLISSEMENT_MACHINE': False,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.25
                },
                'risk_analysis': {
                    'risk_level': 'low',
                    'fraud_indicators': []
                }
            },
            {
                'declaration_id': '2024/ABJ/54321',
                'chapter_id': 'chap85',
                'predicted_fraud': True,
                'fraud_probability': 0.75,
                'confidence_score': 0.85,
                'decision': 'fraude',
                'model_used': 'LightGBM_Chap85_Best',
                'optimal_threshold_used': 0.22,
                'auc_score': 0.9995,
                'f1_score': 0.9791,
                'precision_score': 0.9985,
                'recall_score': 0.9872,
                'ml_integration_used': True,
                'decision_source': 'ml',
                'context_features': {
                    'VALEUR_CAF': 25000,
                    'POIDS_NET': 50,
                    'BUSINESS_GLISSEMENT_ELECTRONIQUE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.88
                },
                'risk_analysis': {
                    'risk_level': 'high',
                    'fraud_indicators': ['electronic_shift', 'weight_value_mismatch']
                }
            }
        ]
        
        # Insérer les prédictions de test
        for pred in test_predictions:
            try:
                cursor.execute("""
                    INSERT INTO predictions (
                        declaration_id, chapter_id, predicted_fraud, fraud_probability,
                        confidence_score, decision, model_used, optimal_threshold_used,
                        auc_score, f1_score, precision_score, recall_score,
                        ml_integration_used, decision_source, context_features, risk_analysis
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    pred['declaration_id'], pred['chapter_id'], pred['predicted_fraud'],
                    pred['fraud_probability'], pred['confidence_score'], pred['decision'],
                    pred['model_used'], pred['optimal_threshold_used'], pred['auc_score'],
                    pred['f1_score'], pred['precision_score'], pred['recall_score'],
                    pred['ml_integration_used'], pred['decision_source'],
                    json.dumps(pred['context_features']), json.dumps(pred['risk_analysis'])
                ))
                logger.info(f"✅ Prédiction insérée: {pred['declaration_id']} ({pred['chapter_id']})")
            except Exception as e:
                logger.error(f"❌ Erreur insertion prédiction {pred['declaration_id']}: {e}")
        
        # ========================================
        # 3. TEST DES DÉCISIONS RL
        # ========================================
        logger.info("\n📊 3. TEST DES DÉCISIONS RL")
        logger.info("-" * 50)
        
        # Créer des données de test pour les décisions RL
        test_rl_decisions = [
            {
                'chapter_id': 'chap30',
                'declaration_id': '2024/ABJ/12345',
                'ts': datetime.now().isoformat(),
                'context_key': 'high_value_cosmetic_shift',
                'context_json': {
                    'VALEUR_CAF': 50000,
                    'BUSINESS_GLISSEMENT_TARIFAIRE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.95
                },
                'action': 'inspect',
                'model_proba': 0.85,
                'rl_proba': 0.88,
                'exploration': False,
                'decision_source': 'ml_rl_integrated',
                'confidence_score': 0.92,
                'context_complexity': 3,
                'seasonal_factor': 1.0,
                'bureau_risk_score': 0.7,
                'extra_json': {
                    'strategy': 'exploit',
                    'bandit_arm': 'conservative',
                    'epsilon': 0.1
                }
            },
            {
                'chapter_id': 'chap84',
                'declaration_id': '2024/DKR/67890',
                'ts': datetime.now().isoformat(),
                'context_key': 'normal_machine_import',
                'context_json': {
                    'VALEUR_CAF': 15000,
                    'BUSINESS_GLISSEMENT_MACHINE': False,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.25
                },
                'action': 'clear',
                'model_proba': 0.15,
                'rl_proba': 0.12,
                'exploration': True,
                'decision_source': 'advanced_rl',
                'confidence_score': 0.88,
                'context_complexity': 1,
                'seasonal_factor': 1.0,
                'bureau_risk_score': 0.3,
                'extra_json': {
                    'strategy': 'explore',
                    'bandit_arm': 'aggressive',
                    'epsilon': 0.2
                }
            },
            {
                'chapter_id': 'chap85',
                'declaration_id': '2024/ABJ/54321',
                'ts': datetime.now().isoformat(),
                'context_key': 'electronic_device_anomaly',
                'context_json': {
                    'VALEUR_CAF': 25000,
                    'BUSINESS_GLISSEMENT_ELECTRONIQUE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.88
                },
                'action': 'inspect',
                'model_proba': 0.75,
                'rl_proba': 0.78,
                'exploration': False,
                'decision_source': 'ml_rl_integrated',
                'confidence_score': 0.85,
                'context_complexity': 2,
                'seasonal_factor': 1.0,
                'bureau_risk_score': 0.6,
                'extra_json': {
                    'strategy': 'exploit',
                    'bandit_arm': 'balanced',
                    'epsilon': 0.15
                }
            }
        ]
        
        # Insérer les décisions RL de test
        for decision in test_rl_decisions:
            try:
                cursor.execute("""
                    INSERT INTO advanced_decisions (
                        chapter_id, declaration_id, ts, context_key, context_json, action,
                        model_proba, rl_proba, exploration, decision_source, confidence_score,
                        context_complexity, seasonal_factor, bureau_risk_score, extra_json
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    decision['chapter_id'], decision['declaration_id'], decision['ts'],
                    decision['context_key'], json.dumps(decision['context_json']),
                    decision['action'], decision['model_proba'], decision['rl_proba'],
                    decision['exploration'], decision['decision_source'], decision['confidence_score'],
                    decision['context_complexity'], decision['seasonal_factor'],
                    decision['bureau_risk_score'], json.dumps(decision['extra_json'])
                ))
                logger.info(f"✅ Décision RL insérée: {decision['declaration_id']} ({decision['chapter_id']})")
            except Exception as e:
                logger.error(f"❌ Erreur insertion décision RL {decision['declaration_id']}: {e}")
        
        
        conn.commit()
        logger.info("\n✅ Données de test sauvegardées")
        
        # ========================================
        # VÉRIFICATION FINALE
        # ========================================
        logger.info("\n🔍 VÉRIFICATION FINALE")
        logger.info("=" * 50)
        
        # Vérifier les prédictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        logger.info(f"📊 Total prédictions: {total_predictions}")
        
        # Vérifier les décisions RL
        cursor.execute('SELECT COUNT(*) FROM advanced_decisions')
        total_rl_decisions = cursor.fetchone()[0]
        logger.info(f"📊 Total décisions RL: {total_rl_decisions}")
        
        # Vérifier les déclarations
        cursor.execute('SELECT COUNT(*) FROM declarations')
        total_declarations = cursor.fetchone()[0]
        logger.info(f"📊 Total déclarations: {total_declarations}")
        
        # Statistiques par chapitre
        cursor.execute('''
            SELECT chapter_id, COUNT(*) as count
            FROM predictions 
            GROUP BY chapter_id
            ORDER BY chapter_id
        ''')
        pred_stats = cursor.fetchall()
        logger.info("📊 Prédictions par chapitre:")
        for chapter_id, count in pred_stats:
            logger.info(f"  - {chapter_id}: {count}")
        
        cursor.execute('''
            SELECT chapter_id, COUNT(*) as count
            FROM advanced_decisions 
            GROUP BY chapter_id
            ORDER BY chapter_id
        ''')
        rl_stats = cursor.fetchall()
        logger.info("📊 Décisions RL par chapitre:")
        for chapter_id, count in rl_stats:
            logger.info(f"  - {chapter_id}: {count}")
        
        if total_predictions > 0 and total_rl_decisions > 0 and total_declarations > 0:
            logger.info("\n🎉 STOCKAGE DES PRÉDICTIONS ET RL VALIDÉ!")
            logger.info("✅ Les prédictions ML sont bien stockées")
            logger.info("✅ Les décisions RL sont bien stockées")
            logger.info("✅ Les déclarations sont bien stockées")
            logger.info("✅ Le système de stockage fonctionne correctement")
            return True
        else:
            logger.error("\n💥 PROBLÈME DE STOCKAGE!")
            logger.error("❌ Des données n'ont pas été stockées correctement")
            return False
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Erreur test stockage: {e}")
        return False

if __name__ == "__main__":
    success = test_predictions_storage()
    if not success:
        sys.exit(1)



























"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_predictions_storage():
    """Teste le stockage des prédictions et décisions RL"""
    
    logger.info("🧪 TEST DU STOCKAGE DES PRÉDICTIONS ET RL")
    logger.info("=" * 80)
    
    try:
        from database_config import get_database_config
        import psycopg2
        
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # ========================================
        # 1. TEST DES DÉCLARATIONS (DOIT ÊTRE EN PREMIER)
        # ========================================
        logger.info("\n📊 1. TEST DES DÉCLARATIONS")
        logger.info("-" * 50)
        
        # Créer des déclarations de test
        test_declarations = [
            {
                'declaration_id': '2024/ABJ/12345',
                'chapter_id': 'chap30',
                'file_name': 'test_chap30.csv',
                'file_type': 'csv',
                'source_type': 'api',
                'annee': 2024,
                'bureau': 'ABJ',
                'numero_declaration': '12345',
                'valeur_caf': 50000,
                'poids_net': 100,
                'extraction_status': 'success',
                'validation_status': 'valid'
            },
            {
                'declaration_id': '2024/DKR/67890',
                'chapter_id': 'chap84',
                'file_name': 'test_chap84.csv',
                'file_type': 'csv',
                'source_type': 'api',
                'annee': 2024,
                'bureau': 'DKR',
                'numero_declaration': '67890',
                'valeur_caf': 15000,
                'poids_net': 500,
                'extraction_status': 'success',
                'validation_status': 'valid'
            },
            {
                'declaration_id': '2024/ABJ/54321',
                'chapter_id': 'chap85',
                'file_name': 'test_chap85.csv',
                'file_type': 'csv',
                'source_type': 'api',
                'annee': 2024,
                'bureau': 'ABJ',
                'numero_declaration': '54321',
                'valeur_caf': 25000,
                'poids_net': 50,
                'extraction_status': 'success',
                'validation_status': 'valid'
            }
        ]
        
        # Insérer les déclarations de test
        for decl in test_declarations:
            try:
                cursor.execute("""
                    INSERT INTO declarations (
                        declaration_id, chapter_id, file_name, file_type, source_type,
                        annee, bureau, numero_declaration, valeur_caf, poids_net,
                        extraction_status, validation_status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    decl['declaration_id'], decl['chapter_id'], decl['file_name'],
                    decl['file_type'], decl['source_type'], decl['annee'],
                    decl['bureau'], decl['numero_declaration'], decl['valeur_caf'],
                    decl['poids_net'], decl['extraction_status'], decl['validation_status']
                ))
                logger.info(f"✅ Déclaration insérée: {decl['declaration_id']} ({decl['chapter_id']})")
            except Exception as e:
                logger.error(f"❌ Erreur insertion déclaration {decl['declaration_id']}: {e}")
        
        # ========================================
        # 2. TEST DES PRÉDICTIONS ML
        # ========================================
        logger.info("\n📊 2. TEST DES PRÉDICTIONS ML")
        logger.info("-" * 50)
        
        # Créer des données de test pour les prédictions
        test_predictions = [
            {
                'declaration_id': '2024/ABJ/12345',
                'chapter_id': 'chap30',
                'predicted_fraud': True,
                'fraud_probability': 0.85,
                'confidence_score': 0.92,
                'decision': 'fraude',
                'model_used': 'XGBoost_Chap30_Best',
                'optimal_threshold_used': 0.35,
                'auc_score': 0.9995,
                'f1_score': 0.9796,
                'precision_score': 0.9889,
                'recall_score': 0.9705,
                'ml_integration_used': True,
                'decision_source': 'ml',
                'context_features': {
                    'VALEUR_CAF': 50000,
                    'POIDS_NET': 100,
                    'BUSINESS_GLISSEMENT_TARIFAIRE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.95
                },
                'risk_analysis': {
                    'risk_level': 'high',
                    'fraud_indicators': ['high_value', 'cosmetic_shift', 'statistical_anomaly']
                }
            },
            {
                'declaration_id': '2024/DKR/67890',
                'chapter_id': 'chap84',
                'predicted_fraud': False,
                'fraud_probability': 0.15,
                'confidence_score': 0.88,
                'decision': 'conforme',
                'model_used': 'XGBoost_Chap84_Best',
                'optimal_threshold_used': 0.22,
                'auc_score': 0.9997,
                'f1_score': 0.9888,
                'precision_score': 0.9992,
                'recall_score': 0.9834,
                'ml_integration_used': True,
                'decision_source': 'ml',
                'context_features': {
                    'VALEUR_CAF': 15000,
                    'POIDS_NET': 500,
                    'BUSINESS_GLISSEMENT_MACHINE': False,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.25
                },
                'risk_analysis': {
                    'risk_level': 'low',
                    'fraud_indicators': []
                }
            },
            {
                'declaration_id': '2024/ABJ/54321',
                'chapter_id': 'chap85',
                'predicted_fraud': True,
                'fraud_probability': 0.75,
                'confidence_score': 0.85,
                'decision': 'fraude',
                'model_used': 'LightGBM_Chap85_Best',
                'optimal_threshold_used': 0.22,
                'auc_score': 0.9995,
                'f1_score': 0.9791,
                'precision_score': 0.9985,
                'recall_score': 0.9872,
                'ml_integration_used': True,
                'decision_source': 'ml',
                'context_features': {
                    'VALEUR_CAF': 25000,
                    'POIDS_NET': 50,
                    'BUSINESS_GLISSEMENT_ELECTRONIQUE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.88
                },
                'risk_analysis': {
                    'risk_level': 'high',
                    'fraud_indicators': ['electronic_shift', 'weight_value_mismatch']
                }
            }
        ]
        
        # Insérer les prédictions de test
        for pred in test_predictions:
            try:
                cursor.execute("""
                    INSERT INTO predictions (
                        declaration_id, chapter_id, predicted_fraud, fraud_probability,
                        confidence_score, decision, model_used, optimal_threshold_used,
                        auc_score, f1_score, precision_score, recall_score,
                        ml_integration_used, decision_source, context_features, risk_analysis
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    pred['declaration_id'], pred['chapter_id'], pred['predicted_fraud'],
                    pred['fraud_probability'], pred['confidence_score'], pred['decision'],
                    pred['model_used'], pred['optimal_threshold_used'], pred['auc_score'],
                    pred['f1_score'], pred['precision_score'], pred['recall_score'],
                    pred['ml_integration_used'], pred['decision_source'],
                    json.dumps(pred['context_features']), json.dumps(pred['risk_analysis'])
                ))
                logger.info(f"✅ Prédiction insérée: {pred['declaration_id']} ({pred['chapter_id']})")
            except Exception as e:
                logger.error(f"❌ Erreur insertion prédiction {pred['declaration_id']}: {e}")
        
        # ========================================
        # 3. TEST DES DÉCISIONS RL
        # ========================================
        logger.info("\n📊 3. TEST DES DÉCISIONS RL")
        logger.info("-" * 50)
        
        # Créer des données de test pour les décisions RL
        test_rl_decisions = [
            {
                'chapter_id': 'chap30',
                'declaration_id': '2024/ABJ/12345',
                'ts': datetime.now().isoformat(),
                'context_key': 'high_value_cosmetic_shift',
                'context_json': {
                    'VALEUR_CAF': 50000,
                    'BUSINESS_GLISSEMENT_TARIFAIRE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.95
                },
                'action': 'inspect',
                'model_proba': 0.85,
                'rl_proba': 0.88,
                'exploration': False,
                'decision_source': 'ml_rl_integrated',
                'confidence_score': 0.92,
                'context_complexity': 3,
                'seasonal_factor': 1.0,
                'bureau_risk_score': 0.7,
                'extra_json': {
                    'strategy': 'exploit',
                    'bandit_arm': 'conservative',
                    'epsilon': 0.1
                }
            },
            {
                'chapter_id': 'chap84',
                'declaration_id': '2024/DKR/67890',
                'ts': datetime.now().isoformat(),
                'context_key': 'normal_machine_import',
                'context_json': {
                    'VALEUR_CAF': 15000,
                    'BUSINESS_GLISSEMENT_MACHINE': False,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.25
                },
                'action': 'clear',
                'model_proba': 0.15,
                'rl_proba': 0.12,
                'exploration': True,
                'decision_source': 'advanced_rl',
                'confidence_score': 0.88,
                'context_complexity': 1,
                'seasonal_factor': 1.0,
                'bureau_risk_score': 0.3,
                'extra_json': {
                    'strategy': 'explore',
                    'bandit_arm': 'aggressive',
                    'epsilon': 0.2
                }
            },
            {
                'chapter_id': 'chap85',
                'declaration_id': '2024/ABJ/54321',
                'ts': datetime.now().isoformat(),
                'context_key': 'electronic_device_anomaly',
                'context_json': {
                    'VALEUR_CAF': 25000,
                    'BUSINESS_GLISSEMENT_ELECTRONIQUE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.88
                },
                'action': 'inspect',
                'model_proba': 0.75,
                'rl_proba': 0.78,
                'exploration': False,
                'decision_source': 'ml_rl_integrated',
                'confidence_score': 0.85,
                'context_complexity': 2,
                'seasonal_factor': 1.0,
                'bureau_risk_score': 0.6,
                'extra_json': {
                    'strategy': 'exploit',
                    'bandit_arm': 'balanced',
                    'epsilon': 0.15
                }
            }
        ]
        
        # Insérer les décisions RL de test
        for decision in test_rl_decisions:
            try:
                cursor.execute("""
                    INSERT INTO advanced_decisions (
                        chapter_id, declaration_id, ts, context_key, context_json, action,
                        model_proba, rl_proba, exploration, decision_source, confidence_score,
                        context_complexity, seasonal_factor, bureau_risk_score, extra_json
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    decision['chapter_id'], decision['declaration_id'], decision['ts'],
                    decision['context_key'], json.dumps(decision['context_json']),
                    decision['action'], decision['model_proba'], decision['rl_proba'],
                    decision['exploration'], decision['decision_source'], decision['confidence_score'],
                    decision['context_complexity'], decision['seasonal_factor'],
                    decision['bureau_risk_score'], json.dumps(decision['extra_json'])
                ))
                logger.info(f"✅ Décision RL insérée: {decision['declaration_id']} ({decision['chapter_id']})")
            except Exception as e:
                logger.error(f"❌ Erreur insertion décision RL {decision['declaration_id']}: {e}")
        
        
        conn.commit()
        logger.info("\n✅ Données de test sauvegardées")
        
        # ========================================
        # VÉRIFICATION FINALE
        # ========================================
        logger.info("\n🔍 VÉRIFICATION FINALE")
        logger.info("=" * 50)
        
        # Vérifier les prédictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        logger.info(f"📊 Total prédictions: {total_predictions}")
        
        # Vérifier les décisions RL
        cursor.execute('SELECT COUNT(*) FROM advanced_decisions')
        total_rl_decisions = cursor.fetchone()[0]
        logger.info(f"📊 Total décisions RL: {total_rl_decisions}")
        
        # Vérifier les déclarations
        cursor.execute('SELECT COUNT(*) FROM declarations')
        total_declarations = cursor.fetchone()[0]
        logger.info(f"📊 Total déclarations: {total_declarations}")
        
        # Statistiques par chapitre
        cursor.execute('''
            SELECT chapter_id, COUNT(*) as count
            FROM predictions 
            GROUP BY chapter_id
            ORDER BY chapter_id
        ''')
        pred_stats = cursor.fetchall()
        logger.info("📊 Prédictions par chapitre:")
        for chapter_id, count in pred_stats:
            logger.info(f"  - {chapter_id}: {count}")
        
        cursor.execute('''
            SELECT chapter_id, COUNT(*) as count
            FROM advanced_decisions 
            GROUP BY chapter_id
            ORDER BY chapter_id
        ''')
        rl_stats = cursor.fetchall()
        logger.info("📊 Décisions RL par chapitre:")
        for chapter_id, count in rl_stats:
            logger.info(f"  - {chapter_id}: {count}")
        
        if total_predictions > 0 and total_rl_decisions > 0 and total_declarations > 0:
            logger.info("\n🎉 STOCKAGE DES PRÉDICTIONS ET RL VALIDÉ!")
            logger.info("✅ Les prédictions ML sont bien stockées")
            logger.info("✅ Les décisions RL sont bien stockées")
            logger.info("✅ Les déclarations sont bien stockées")
            logger.info("✅ Le système de stockage fonctionne correctement")
            return True
        else:
            logger.error("\n💥 PROBLÈME DE STOCKAGE!")
            logger.error("❌ Des données n'ont pas été stockées correctement")
            return False
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Erreur test stockage: {e}")
        return False

if __name__ == "__main__":
    success = test_predictions_storage()
    if not success:
        sys.exit(1)



























"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_predictions_storage():
    """Teste le stockage des prédictions et décisions RL"""
    
    logger.info("🧪 TEST DU STOCKAGE DES PRÉDICTIONS ET RL")
    logger.info("=" * 80)
    
    try:
        from database_config import get_database_config
        import psycopg2
        
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # ========================================
        # 1. TEST DES DÉCLARATIONS (DOIT ÊTRE EN PREMIER)
        # ========================================
        logger.info("\n📊 1. TEST DES DÉCLARATIONS")
        logger.info("-" * 50)
        
        # Créer des déclarations de test
        test_declarations = [
            {
                'declaration_id': '2024/ABJ/12345',
                'chapter_id': 'chap30',
                'file_name': 'test_chap30.csv',
                'file_type': 'csv',
                'source_type': 'api',
                'annee': 2024,
                'bureau': 'ABJ',
                'numero_declaration': '12345',
                'valeur_caf': 50000,
                'poids_net': 100,
                'extraction_status': 'success',
                'validation_status': 'valid'
            },
            {
                'declaration_id': '2024/DKR/67890',
                'chapter_id': 'chap84',
                'file_name': 'test_chap84.csv',
                'file_type': 'csv',
                'source_type': 'api',
                'annee': 2024,
                'bureau': 'DKR',
                'numero_declaration': '67890',
                'valeur_caf': 15000,
                'poids_net': 500,
                'extraction_status': 'success',
                'validation_status': 'valid'
            },
            {
                'declaration_id': '2024/ABJ/54321',
                'chapter_id': 'chap85',
                'file_name': 'test_chap85.csv',
                'file_type': 'csv',
                'source_type': 'api',
                'annee': 2024,
                'bureau': 'ABJ',
                'numero_declaration': '54321',
                'valeur_caf': 25000,
                'poids_net': 50,
                'extraction_status': 'success',
                'validation_status': 'valid'
            }
        ]
        
        # Insérer les déclarations de test
        for decl in test_declarations:
            try:
                cursor.execute("""
                    INSERT INTO declarations (
                        declaration_id, chapter_id, file_name, file_type, source_type,
                        annee, bureau, numero_declaration, valeur_caf, poids_net,
                        extraction_status, validation_status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    decl['declaration_id'], decl['chapter_id'], decl['file_name'],
                    decl['file_type'], decl['source_type'], decl['annee'],
                    decl['bureau'], decl['numero_declaration'], decl['valeur_caf'],
                    decl['poids_net'], decl['extraction_status'], decl['validation_status']
                ))
                logger.info(f"✅ Déclaration insérée: {decl['declaration_id']} ({decl['chapter_id']})")
            except Exception as e:
                logger.error(f"❌ Erreur insertion déclaration {decl['declaration_id']}: {e}")
        
        # ========================================
        # 2. TEST DES PRÉDICTIONS ML
        # ========================================
        logger.info("\n📊 2. TEST DES PRÉDICTIONS ML")
        logger.info("-" * 50)
        
        # Créer des données de test pour les prédictions
        test_predictions = [
            {
                'declaration_id': '2024/ABJ/12345',
                'chapter_id': 'chap30',
                'predicted_fraud': True,
                'fraud_probability': 0.85,
                'confidence_score': 0.92,
                'decision': 'fraude',
                'model_used': 'XGBoost_Chap30_Best',
                'optimal_threshold_used': 0.35,
                'auc_score': 0.9995,
                'f1_score': 0.9796,
                'precision_score': 0.9889,
                'recall_score': 0.9705,
                'ml_integration_used': True,
                'decision_source': 'ml',
                'context_features': {
                    'VALEUR_CAF': 50000,
                    'POIDS_NET': 100,
                    'BUSINESS_GLISSEMENT_TARIFAIRE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.95
                },
                'risk_analysis': {
                    'risk_level': 'high',
                    'fraud_indicators': ['high_value', 'cosmetic_shift', 'statistical_anomaly']
                }
            },
            {
                'declaration_id': '2024/DKR/67890',
                'chapter_id': 'chap84',
                'predicted_fraud': False,
                'fraud_probability': 0.15,
                'confidence_score': 0.88,
                'decision': 'conforme',
                'model_used': 'XGBoost_Chap84_Best',
                'optimal_threshold_used': 0.22,
                'auc_score': 0.9997,
                'f1_score': 0.9888,
                'precision_score': 0.9992,
                'recall_score': 0.9834,
                'ml_integration_used': True,
                'decision_source': 'ml',
                'context_features': {
                    'VALEUR_CAF': 15000,
                    'POIDS_NET': 500,
                    'BUSINESS_GLISSEMENT_MACHINE': False,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.25
                },
                'risk_analysis': {
                    'risk_level': 'low',
                    'fraud_indicators': []
                }
            },
            {
                'declaration_id': '2024/ABJ/54321',
                'chapter_id': 'chap85',
                'predicted_fraud': True,
                'fraud_probability': 0.75,
                'confidence_score': 0.85,
                'decision': 'fraude',
                'model_used': 'LightGBM_Chap85_Best',
                'optimal_threshold_used': 0.22,
                'auc_score': 0.9995,
                'f1_score': 0.9791,
                'precision_score': 0.9985,
                'recall_score': 0.9872,
                'ml_integration_used': True,
                'decision_source': 'ml',
                'context_features': {
                    'VALEUR_CAF': 25000,
                    'POIDS_NET': 50,
                    'BUSINESS_GLISSEMENT_ELECTRONIQUE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.88
                },
                'risk_analysis': {
                    'risk_level': 'high',
                    'fraud_indicators': ['electronic_shift', 'weight_value_mismatch']
                }
            }
        ]
        
        # Insérer les prédictions de test
        for pred in test_predictions:
            try:
                cursor.execute("""
                    INSERT INTO predictions (
                        declaration_id, chapter_id, predicted_fraud, fraud_probability,
                        confidence_score, decision, model_used, optimal_threshold_used,
                        auc_score, f1_score, precision_score, recall_score,
                        ml_integration_used, decision_source, context_features, risk_analysis
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    pred['declaration_id'], pred['chapter_id'], pred['predicted_fraud'],
                    pred['fraud_probability'], pred['confidence_score'], pred['decision'],
                    pred['model_used'], pred['optimal_threshold_used'], pred['auc_score'],
                    pred['f1_score'], pred['precision_score'], pred['recall_score'],
                    pred['ml_integration_used'], pred['decision_source'],
                    json.dumps(pred['context_features']), json.dumps(pred['risk_analysis'])
                ))
                logger.info(f"✅ Prédiction insérée: {pred['declaration_id']} ({pred['chapter_id']})")
            except Exception as e:
                logger.error(f"❌ Erreur insertion prédiction {pred['declaration_id']}: {e}")
        
        # ========================================
        # 3. TEST DES DÉCISIONS RL
        # ========================================
        logger.info("\n📊 3. TEST DES DÉCISIONS RL")
        logger.info("-" * 50)
        
        # Créer des données de test pour les décisions RL
        test_rl_decisions = [
            {
                'chapter_id': 'chap30',
                'declaration_id': '2024/ABJ/12345',
                'ts': datetime.now().isoformat(),
                'context_key': 'high_value_cosmetic_shift',
                'context_json': {
                    'VALEUR_CAF': 50000,
                    'BUSINESS_GLISSEMENT_TARIFAIRE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.95
                },
                'action': 'inspect',
                'model_proba': 0.85,
                'rl_proba': 0.88,
                'exploration': False,
                'decision_source': 'ml_rl_integrated',
                'confidence_score': 0.92,
                'context_complexity': 3,
                'seasonal_factor': 1.0,
                'bureau_risk_score': 0.7,
                'extra_json': {
                    'strategy': 'exploit',
                    'bandit_arm': 'conservative',
                    'epsilon': 0.1
                }
            },
            {
                'chapter_id': 'chap84',
                'declaration_id': '2024/DKR/67890',
                'ts': datetime.now().isoformat(),
                'context_key': 'normal_machine_import',
                'context_json': {
                    'VALEUR_CAF': 15000,
                    'BUSINESS_GLISSEMENT_MACHINE': False,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.25
                },
                'action': 'clear',
                'model_proba': 0.15,
                'rl_proba': 0.12,
                'exploration': True,
                'decision_source': 'advanced_rl',
                'confidence_score': 0.88,
                'context_complexity': 1,
                'seasonal_factor': 1.0,
                'bureau_risk_score': 0.3,
                'extra_json': {
                    'strategy': 'explore',
                    'bandit_arm': 'aggressive',
                    'epsilon': 0.2
                }
            },
            {
                'chapter_id': 'chap85',
                'declaration_id': '2024/ABJ/54321',
                'ts': datetime.now().isoformat(),
                'context_key': 'electronic_device_anomaly',
                'context_json': {
                    'VALEUR_CAF': 25000,
                    'BUSINESS_GLISSEMENT_ELECTRONIQUE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.88
                },
                'action': 'inspect',
                'model_proba': 0.75,
                'rl_proba': 0.78,
                'exploration': False,
                'decision_source': 'ml_rl_integrated',
                'confidence_score': 0.85,
                'context_complexity': 2,
                'seasonal_factor': 1.0,
                'bureau_risk_score': 0.6,
                'extra_json': {
                    'strategy': 'exploit',
                    'bandit_arm': 'balanced',
                    'epsilon': 0.15
                }
            }
        ]
        
        # Insérer les décisions RL de test
        for decision in test_rl_decisions:
            try:
                cursor.execute("""
                    INSERT INTO advanced_decisions (
                        chapter_id, declaration_id, ts, context_key, context_json, action,
                        model_proba, rl_proba, exploration, decision_source, confidence_score,
                        context_complexity, seasonal_factor, bureau_risk_score, extra_json
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    decision['chapter_id'], decision['declaration_id'], decision['ts'],
                    decision['context_key'], json.dumps(decision['context_json']),
                    decision['action'], decision['model_proba'], decision['rl_proba'],
                    decision['exploration'], decision['decision_source'], decision['confidence_score'],
                    decision['context_complexity'], decision['seasonal_factor'],
                    decision['bureau_risk_score'], json.dumps(decision['extra_json'])
                ))
                logger.info(f"✅ Décision RL insérée: {decision['declaration_id']} ({decision['chapter_id']})")
            except Exception as e:
                logger.error(f"❌ Erreur insertion décision RL {decision['declaration_id']}: {e}")
        
        
        conn.commit()
        logger.info("\n✅ Données de test sauvegardées")
        
        # ========================================
        # VÉRIFICATION FINALE
        # ========================================
        logger.info("\n🔍 VÉRIFICATION FINALE")
        logger.info("=" * 50)
        
        # Vérifier les prédictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        logger.info(f"📊 Total prédictions: {total_predictions}")
        
        # Vérifier les décisions RL
        cursor.execute('SELECT COUNT(*) FROM advanced_decisions')
        total_rl_decisions = cursor.fetchone()[0]
        logger.info(f"📊 Total décisions RL: {total_rl_decisions}")
        
        # Vérifier les déclarations
        cursor.execute('SELECT COUNT(*) FROM declarations')
        total_declarations = cursor.fetchone()[0]
        logger.info(f"📊 Total déclarations: {total_declarations}")
        
        # Statistiques par chapitre
        cursor.execute('''
            SELECT chapter_id, COUNT(*) as count
            FROM predictions 
            GROUP BY chapter_id
            ORDER BY chapter_id
        ''')
        pred_stats = cursor.fetchall()
        logger.info("📊 Prédictions par chapitre:")
        for chapter_id, count in pred_stats:
            logger.info(f"  - {chapter_id}: {count}")
        
        cursor.execute('''
            SELECT chapter_id, COUNT(*) as count
            FROM advanced_decisions 
            GROUP BY chapter_id
            ORDER BY chapter_id
        ''')
        rl_stats = cursor.fetchall()
        logger.info("📊 Décisions RL par chapitre:")
        for chapter_id, count in rl_stats:
            logger.info(f"  - {chapter_id}: {count}")
        
        if total_predictions > 0 and total_rl_decisions > 0 and total_declarations > 0:
            logger.info("\n🎉 STOCKAGE DES PRÉDICTIONS ET RL VALIDÉ!")
            logger.info("✅ Les prédictions ML sont bien stockées")
            logger.info("✅ Les décisions RL sont bien stockées")
            logger.info("✅ Les déclarations sont bien stockées")
            logger.info("✅ Le système de stockage fonctionne correctement")
            return True
        else:
            logger.error("\n💥 PROBLÈME DE STOCKAGE!")
            logger.error("❌ Des données n'ont pas été stockées correctement")
            return False
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Erreur test stockage: {e}")
        return False

if __name__ == "__main__":
    success = test_predictions_storage()
    if not success:
        sys.exit(1)



























"""

import sys
import logging
from pathlib import Path
import json
from datetime import datetime

# Ajouter le chemin du projet
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_predictions_storage():
    """Teste le stockage des prédictions et décisions RL"""
    
    logger.info("🧪 TEST DU STOCKAGE DES PRÉDICTIONS ET RL")
    logger.info("=" * 80)
    
    try:
        from database_config import get_database_config
        import psycopg2
        
        db_config = get_database_config()
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # ========================================
        # 1. TEST DES DÉCLARATIONS (DOIT ÊTRE EN PREMIER)
        # ========================================
        logger.info("\n📊 1. TEST DES DÉCLARATIONS")
        logger.info("-" * 50)
        
        # Créer des déclarations de test
        test_declarations = [
            {
                'declaration_id': '2024/ABJ/12345',
                'chapter_id': 'chap30',
                'file_name': 'test_chap30.csv',
                'file_type': 'csv',
                'source_type': 'api',
                'annee': 2024,
                'bureau': 'ABJ',
                'numero_declaration': '12345',
                'valeur_caf': 50000,
                'poids_net': 100,
                'extraction_status': 'success',
                'validation_status': 'valid'
            },
            {
                'declaration_id': '2024/DKR/67890',
                'chapter_id': 'chap84',
                'file_name': 'test_chap84.csv',
                'file_type': 'csv',
                'source_type': 'api',
                'annee': 2024,
                'bureau': 'DKR',
                'numero_declaration': '67890',
                'valeur_caf': 15000,
                'poids_net': 500,
                'extraction_status': 'success',
                'validation_status': 'valid'
            },
            {
                'declaration_id': '2024/ABJ/54321',
                'chapter_id': 'chap85',
                'file_name': 'test_chap85.csv',
                'file_type': 'csv',
                'source_type': 'api',
                'annee': 2024,
                'bureau': 'ABJ',
                'numero_declaration': '54321',
                'valeur_caf': 25000,
                'poids_net': 50,
                'extraction_status': 'success',
                'validation_status': 'valid'
            }
        ]
        
        # Insérer les déclarations de test
        for decl in test_declarations:
            try:
                cursor.execute("""
                    INSERT INTO declarations (
                        declaration_id, chapter_id, file_name, file_type, source_type,
                        annee, bureau, numero_declaration, valeur_caf, poids_net,
                        extraction_status, validation_status
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    decl['declaration_id'], decl['chapter_id'], decl['file_name'],
                    decl['file_type'], decl['source_type'], decl['annee'],
                    decl['bureau'], decl['numero_declaration'], decl['valeur_caf'],
                    decl['poids_net'], decl['extraction_status'], decl['validation_status']
                ))
                logger.info(f"✅ Déclaration insérée: {decl['declaration_id']} ({decl['chapter_id']})")
            except Exception as e:
                logger.error(f"❌ Erreur insertion déclaration {decl['declaration_id']}: {e}")
        
        # ========================================
        # 2. TEST DES PRÉDICTIONS ML
        # ========================================
        logger.info("\n📊 2. TEST DES PRÉDICTIONS ML")
        logger.info("-" * 50)
        
        # Créer des données de test pour les prédictions
        test_predictions = [
            {
                'declaration_id': '2024/ABJ/12345',
                'chapter_id': 'chap30',
                'predicted_fraud': True,
                'fraud_probability': 0.85,
                'confidence_score': 0.92,
                'decision': 'fraude',
                'model_used': 'XGBoost_Chap30_Best',
                'optimal_threshold_used': 0.35,
                'auc_score': 0.9995,
                'f1_score': 0.9796,
                'precision_score': 0.9889,
                'recall_score': 0.9705,
                'ml_integration_used': True,
                'decision_source': 'ml',
                'context_features': {
                    'VALEUR_CAF': 50000,
                    'POIDS_NET': 100,
                    'BUSINESS_GLISSEMENT_TARIFAIRE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.95
                },
                'risk_analysis': {
                    'risk_level': 'high',
                    'fraud_indicators': ['high_value', 'cosmetic_shift', 'statistical_anomaly']
                }
            },
            {
                'declaration_id': '2024/DKR/67890',
                'chapter_id': 'chap84',
                'predicted_fraud': False,
                'fraud_probability': 0.15,
                'confidence_score': 0.88,
                'decision': 'conforme',
                'model_used': 'XGBoost_Chap84_Best',
                'optimal_threshold_used': 0.22,
                'auc_score': 0.9997,
                'f1_score': 0.9888,
                'precision_score': 0.9992,
                'recall_score': 0.9834,
                'ml_integration_used': True,
                'decision_source': 'ml',
                'context_features': {
                    'VALEUR_CAF': 15000,
                    'POIDS_NET': 500,
                    'BUSINESS_GLISSEMENT_MACHINE': False,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.25
                },
                'risk_analysis': {
                    'risk_level': 'low',
                    'fraud_indicators': []
                }
            },
            {
                'declaration_id': '2024/ABJ/54321',
                'chapter_id': 'chap85',
                'predicted_fraud': True,
                'fraud_probability': 0.75,
                'confidence_score': 0.85,
                'decision': 'fraude',
                'model_used': 'LightGBM_Chap85_Best',
                'optimal_threshold_used': 0.22,
                'auc_score': 0.9995,
                'f1_score': 0.9791,
                'precision_score': 0.9985,
                'recall_score': 0.9872,
                'ml_integration_used': True,
                'decision_source': 'ml',
                'context_features': {
                    'VALEUR_CAF': 25000,
                    'POIDS_NET': 50,
                    'BUSINESS_GLISSEMENT_ELECTRONIQUE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.88
                },
                'risk_analysis': {
                    'risk_level': 'high',
                    'fraud_indicators': ['electronic_shift', 'weight_value_mismatch']
                }
            }
        ]
        
        # Insérer les prédictions de test
        for pred in test_predictions:
            try:
                cursor.execute("""
                    INSERT INTO predictions (
                        declaration_id, chapter_id, predicted_fraud, fraud_probability,
                        confidence_score, decision, model_used, optimal_threshold_used,
                        auc_score, f1_score, precision_score, recall_score,
                        ml_integration_used, decision_source, context_features, risk_analysis
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    pred['declaration_id'], pred['chapter_id'], pred['predicted_fraud'],
                    pred['fraud_probability'], pred['confidence_score'], pred['decision'],
                    pred['model_used'], pred['optimal_threshold_used'], pred['auc_score'],
                    pred['f1_score'], pred['precision_score'], pred['recall_score'],
                    pred['ml_integration_used'], pred['decision_source'],
                    json.dumps(pred['context_features']), json.dumps(pred['risk_analysis'])
                ))
                logger.info(f"✅ Prédiction insérée: {pred['declaration_id']} ({pred['chapter_id']})")
            except Exception as e:
                logger.error(f"❌ Erreur insertion prédiction {pred['declaration_id']}: {e}")
        
        # ========================================
        # 3. TEST DES DÉCISIONS RL
        # ========================================
        logger.info("\n📊 3. TEST DES DÉCISIONS RL")
        logger.info("-" * 50)
        
        # Créer des données de test pour les décisions RL
        test_rl_decisions = [
            {
                'chapter_id': 'chap30',
                'declaration_id': '2024/ABJ/12345',
                'ts': datetime.now().isoformat(),
                'context_key': 'high_value_cosmetic_shift',
                'context_json': {
                    'VALEUR_CAF': 50000,
                    'BUSINESS_GLISSEMENT_TARIFAIRE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.95
                },
                'action': 'inspect',
                'model_proba': 0.85,
                'rl_proba': 0.88,
                'exploration': False,
                'decision_source': 'ml_rl_integrated',
                'confidence_score': 0.92,
                'context_complexity': 3,
                'seasonal_factor': 1.0,
                'bureau_risk_score': 0.7,
                'extra_json': {
                    'strategy': 'exploit',
                    'bandit_arm': 'conservative',
                    'epsilon': 0.1
                }
            },
            {
                'chapter_id': 'chap84',
                'declaration_id': '2024/DKR/67890',
                'ts': datetime.now().isoformat(),
                'context_key': 'normal_machine_import',
                'context_json': {
                    'VALEUR_CAF': 15000,
                    'BUSINESS_GLISSEMENT_MACHINE': False,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.25
                },
                'action': 'clear',
                'model_proba': 0.15,
                'rl_proba': 0.12,
                'exploration': True,
                'decision_source': 'advanced_rl',
                'confidence_score': 0.88,
                'context_complexity': 1,
                'seasonal_factor': 1.0,
                'bureau_risk_score': 0.3,
                'extra_json': {
                    'strategy': 'explore',
                    'bandit_arm': 'aggressive',
                    'epsilon': 0.2
                }
            },
            {
                'chapter_id': 'chap85',
                'declaration_id': '2024/ABJ/54321',
                'ts': datetime.now().isoformat(),
                'context_key': 'electronic_device_anomaly',
                'context_json': {
                    'VALEUR_CAF': 25000,
                    'BUSINESS_GLISSEMENT_ELECTRONIQUE': True,
                    'BIENAYME_CHEBYCHEV_SCORE': 0.88
                },
                'action': 'inspect',
                'model_proba': 0.75,
                'rl_proba': 0.78,
                'exploration': False,
                'decision_source': 'ml_rl_integrated',
                'confidence_score': 0.85,
                'context_complexity': 2,
                'seasonal_factor': 1.0,
                'bureau_risk_score': 0.6,
                'extra_json': {
                    'strategy': 'exploit',
                    'bandit_arm': 'balanced',
                    'epsilon': 0.15
                }
            }
        ]
        
        # Insérer les décisions RL de test
        for decision in test_rl_decisions:
            try:
                cursor.execute("""
                    INSERT INTO advanced_decisions (
                        chapter_id, declaration_id, ts, context_key, context_json, action,
                        model_proba, rl_proba, exploration, decision_source, confidence_score,
                        context_complexity, seasonal_factor, bureau_risk_score, extra_json
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                """, (
                    decision['chapter_id'], decision['declaration_id'], decision['ts'],
                    decision['context_key'], json.dumps(decision['context_json']),
                    decision['action'], decision['model_proba'], decision['rl_proba'],
                    decision['exploration'], decision['decision_source'], decision['confidence_score'],
                    decision['context_complexity'], decision['seasonal_factor'],
                    decision['bureau_risk_score'], json.dumps(decision['extra_json'])
                ))
                logger.info(f"✅ Décision RL insérée: {decision['declaration_id']} ({decision['chapter_id']})")
            except Exception as e:
                logger.error(f"❌ Erreur insertion décision RL {decision['declaration_id']}: {e}")
        
        
        conn.commit()
        logger.info("\n✅ Données de test sauvegardées")
        
        # ========================================
        # VÉRIFICATION FINALE
        # ========================================
        logger.info("\n🔍 VÉRIFICATION FINALE")
        logger.info("=" * 50)
        
        # Vérifier les prédictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        logger.info(f"📊 Total prédictions: {total_predictions}")
        
        # Vérifier les décisions RL
        cursor.execute('SELECT COUNT(*) FROM advanced_decisions')
        total_rl_decisions = cursor.fetchone()[0]
        logger.info(f"📊 Total décisions RL: {total_rl_decisions}")
        
        # Vérifier les déclarations
        cursor.execute('SELECT COUNT(*) FROM declarations')
        total_declarations = cursor.fetchone()[0]
        logger.info(f"📊 Total déclarations: {total_declarations}")
        
        # Statistiques par chapitre
        cursor.execute('''
            SELECT chapter_id, COUNT(*) as count
            FROM predictions 
            GROUP BY chapter_id
            ORDER BY chapter_id
        ''')
        pred_stats = cursor.fetchall()
        logger.info("📊 Prédictions par chapitre:")
        for chapter_id, count in pred_stats:
            logger.info(f"  - {chapter_id}: {count}")
        
        cursor.execute('''
            SELECT chapter_id, COUNT(*) as count
            FROM advanced_decisions 
            GROUP BY chapter_id
            ORDER BY chapter_id
        ''')
        rl_stats = cursor.fetchall()
        logger.info("📊 Décisions RL par chapitre:")
        for chapter_id, count in rl_stats:
            logger.info(f"  - {chapter_id}: {count}")
        
        if total_predictions > 0 and total_rl_decisions > 0 and total_declarations > 0:
            logger.info("\n🎉 STOCKAGE DES PRÉDICTIONS ET RL VALIDÉ!")
            logger.info("✅ Les prédictions ML sont bien stockées")
            logger.info("✅ Les décisions RL sont bien stockées")
            logger.info("✅ Les déclarations sont bien stockées")
            logger.info("✅ Le système de stockage fonctionne correctement")
            return True
        else:
            logger.error("\n💥 PROBLÈME DE STOCKAGE!")
            logger.error("❌ Des données n'ont pas été stockées correctement")
            return False
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Erreur test stockage: {e}")
        return False

if __name__ == "__main__":
    success = test_predictions_storage()
    if not success:
        sys.exit(1)


























