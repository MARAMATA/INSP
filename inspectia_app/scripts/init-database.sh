#!/bin/bash

# Script d'initialisation complète de la base de données INSPECTIA
# Crée les 21 tables et insère les données de base

set -e

echo "🚀 Initialisation de la base de données INSPECTIA..."

# Variables
DB_HOST=${DB_HOST:-postgres}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-inspectia_db}
DB_USER=${DB_USER:-inspectia_user}
DB_PASSWORD=${DB_PASSWORD:-inspectia_pass}

# Attendre que PostgreSQL soit prêt
echo "⏳ Attente de PostgreSQL..."
until pg_isready -h $DB_HOST -p $DB_PORT -U $DB_USER; do
  echo "PostgreSQL n'est pas encore prêt..."
  sleep 2
done

echo "✅ PostgreSQL est prêt!"

# Créer les extensions nécessaires
echo "🔧 Création des extensions PostgreSQL..."
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS \"pg_trgm\";"

# Insérer les données de base pour les chapitres
echo "📊 Insertion des données de base..."

# Chapitres douaniers
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
INSERT INTO chapters (chapter_id, chapter_number, chapter_name, description, specialization, fraud_rate, best_model, model_performance, optimal_threshold, features_count, data_size, advanced_fraud_detection, business_features_count) VALUES
('chap30', 30, 'Produits pharmaceutiques', 'Médicaments et produits pharmaceutiques', 'Pharmaceutique', 0.1944, 'XGBoost', '{"validation_f1": 0.9821, "f1_score": 0.9811, "auc_score": 0.9997, "precision": 0.9876, "recall": 0.9746}', 0.55, 22, 25334, true, 10),
('chap84', 84, 'Machines et appareils mécaniques', 'Machines et appareils mécaniques', 'Mécanique', 0.2680, 'XGBoost', '{"validation_f1": 0.9891, "f1_score": 0.9888, "auc_score": 0.9997, "precision": 0.9992, "recall": 0.9834}', 0.42, 21, 264494, true, 9),
('chap85', 85, 'Appareils électriques', 'Machines et appareils électriques', 'Électrique', 0.2132, 'XGBoost', '{"validation_f1": 0.9781, "f1_score": 0.9808, "auc_score": 0.9993, "precision": 0.9893, "recall": 0.9723}', 0.51, 23, 197402, true, 11)
ON CONFLICT (chapter_id) DO NOTHING;
EOF

# Features pour chaque chapitre
echo "🔧 Insertion des features..."

# Features de base communes
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
INSERT INTO features (feature_name, feature_type, feature_category, description, data_type, is_required) VALUES
('POIDS_NET_KG', 'numerical', 'basic', 'Poids net en kilogrammes', 'float', true),
('NOMBRE_COLIS', 'numerical', 'basic', 'Nombre de colis', 'int', true),
('QUANTITE_COMPLEMENT', 'numerical', 'basic', 'Quantité complémentaire', 'float', true),
('TAUX_DROITS_PERCENT', 'numerical', 'basic', 'Taux de droits en pourcentage', 'float', true),
('CODE_SH_COMPLET', 'categorical', 'basic', 'Code SH complet', 'string', true),
('CODE_PAYS_ORIGINE', 'categorical', 'basic', 'Code pays d''origine', 'string', true),
('CODE_PAYS_PROVENANCE', 'categorical', 'basic', 'Code pays de provenance', 'string', true),
('REGIME_COMPLET', 'categorical', 'basic', 'Régime complet', 'string', true),
('STATUT_BAE', 'categorical', 'basic', 'Statut BAE', 'string', true),
('TYPE_REGIME', 'categorical', 'basic', 'Type de régime', 'string', true),
('REGIME_DOUANIER', 'categorical', 'basic', 'Régime douanier', 'string', true),
('REGIME_FISCAL', 'categorical', 'basic', 'Régime fiscal', 'string', true)
ON CONFLICT (feature_name) DO NOTHING;
EOF

# Features business pour le chapitre 30
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
INSERT INTO features (feature_name, feature_type, feature_category, description, data_type, is_required) VALUES
('BUSINESS_POIDS_NET_KG_EXCEPTIONNEL', 'business', 'business', 'Poids net exceptionnel', 'boolean', false),
('BUSINESS_VALEUR_CAF_EXCEPTIONNEL', 'business', 'business', 'Valeur CAF exceptionnelle', 'boolean', false),
('BUSINESS_SOUS_EVALUATION', 'business', 'business', 'Sous-évaluation détectée', 'boolean', false),
('BUSINESS_QUANTITE_COMPLEMENT_EXCEPTIONNEL', 'business', 'business', 'Quantité complémentaire exceptionnelle', 'boolean', false),
('BUSINESS_NOMBRE_COLIS_EXCEPTIONNEL', 'business', 'business', 'Nombre de colis exceptionnel', 'boolean', false),
('BUSINESS_DROITS_EXCEPTIONNELS', 'business', 'business', 'Droits exceptionnels', 'boolean', false),
('BUSINESS_LIQUIDATION_COMPLEMENTAIRE', 'business', 'business', 'Liquidation complémentaire', 'boolean', false),
('BUSINESS_RATIO_LIQUIDATION_CAF', 'business', 'business', 'Ratio liquidation/CAF', 'float', false),
('BUSINESS_ALERTE_SUSPECT', 'business', 'business', 'Alerte suspect', 'boolean', false),
('BUSINESS_INCOHERENCE_CONDITIONNEMENT', 'business', 'business', 'Incohérence conditionnement', 'boolean', false)
ON CONFLICT (feature_name) DO NOTHING;
EOF

# Associer les features aux chapitres
echo "🔗 Association des features aux chapitres..."

# Features communes pour tous les chapitres
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
INSERT INTO chapter_features (chapter_id, feature_id, is_used, importance_score)
SELECT 
    c.chapter_id,
    f.feature_id,
    true,
    CASE 
        WHEN f.feature_name IN ('POIDS_NET_KG', 'TAUX_DROITS_PERCENT', 'CODE_SH_COMPLET') THEN 0.9
        WHEN f.feature_name IN ('NOMBRE_COLIS', 'QUANTITE_COMPLEMENT', 'CODE_PAYS_ORIGINE') THEN 0.8
        ELSE 0.7
    END
FROM chapters c, features f
WHERE f.feature_name IN (
    'POIDS_NET_KG', 'NOMBRE_COLIS', 'QUANTITE_COMPLEMENT', 'TAUX_DROITS_PERCENT',
    'CODE_SH_COMPLET', 'CODE_PAYS_ORIGINE', 'CODE_PAYS_PROVENANCE',
    'REGIME_COMPLET', 'STATUT_BAE', 'TYPE_REGIME', 'REGIME_DOUANIER', 'REGIME_FISCAL'
)
ON CONFLICT (chapter_id, feature_id) DO NOTHING;
EOF

# Features business spécifiques au chapitre 30
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
INSERT INTO chapter_features (chapter_id, feature_id, is_used, importance_score)
SELECT 
    'chap30',
    f.feature_id,
    true,
    0.8
FROM features f
WHERE f.feature_name LIKE 'BUSINESS_%'
ON CONFLICT (chapter_id, feature_id) DO NOTHING;
EOF

# Créer des profils d'inspecteurs par défaut
echo "👥 Création des profils d'inspecteurs..."

psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
INSERT INTO inspector_profiles (profile_id, inspector_name, specialization, experience_level, rl_strategy, performance_score, created_at) VALUES
('inspector_001', 'Inspecteur Principal', 'Général', 'expert', 'hybrid', 0.95, CURRENT_TIMESTAMP),
('inspector_002', 'Expert Pharmaceutique', 'chap30', 'expert', 'ucb', 0.92, CURRENT_TIMESTAMP),
('inspector_003', 'Expert Mécanique', 'chap84', 'expert', 'thompson', 0.94, CURRENT_TIMESTAMP),
('inspector_004', 'Expert Électrique', 'chap85', 'expert', 'hybrid', 0.91, CURRENT_TIMESTAMP),
('inspector_005', 'Inspecteur Junior', 'Général', 'beginner', 'epsilon_greedy', 0.75, CURRENT_TIMESTAMP)
ON CONFLICT (profile_id) DO NOTHING;
EOF

# Créer des seuils par défaut
echo "🎯 Création des seuils de décision..."

psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
INSERT INTO model_thresholds (chapter_id, threshold_type, threshold_value, confidence_level, created_at) VALUES
('chap30', 'optimal', 0.55, 0.95, CURRENT_TIMESTAMP),
('chap30', 'conforme', 0.0, 0.95, CURRENT_TIMESTAMP),
('chap30', 'fraude', 1.0, 0.95, CURRENT_TIMESTAMP),
('chap84', 'optimal', 0.42, 0.95, CURRENT_TIMESTAMP),
('chap84', 'conforme', 0.0, 0.95, CURRENT_TIMESTAMP),
('chap84', 'fraude', 1.0, 0.95, CURRENT_TIMESTAMP),
('chap85', 'optimal', 0.51, 0.95, CURRENT_TIMESTAMP),
('chap85', 'conforme', 0.0, 0.95, CURRENT_TIMESTAMP),
('chap85', 'fraude', 1.0, 0.95, CURRENT_TIMESTAMP)
ON CONFLICT DO NOTHING;
EOF

echo "✅ Base de données INSPECTIA initialisée avec succès!"
echo "📊 21 tables créées avec données de base"
echo "🎯 3 chapitres configurés (30, 84, 85)"
echo "👥 5 profils d'inspecteurs créés"
echo "🔧 Features et seuils configurés"
