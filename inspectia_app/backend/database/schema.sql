-- =============================================================================
-- SCHEMA POSTGRESQL COMPLET POUR INSPECT_IA
-- Système de détection de fraude douanière avec ML-RL avancé
-- Version 2.0 - Nouveau système avec toutes les features avancées
-- =============================================================================

-- Extension pour UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- 1. TABLES DE CONFIGURATION ET MÉTADONNÉES
-- =============================================================================

-- Table des chapitres douaniers avec nouvelles métriques
CREATE TABLE chapters (
    chapter_id VARCHAR(10) PRIMARY KEY,
    chapter_number INTEGER NOT NULL UNIQUE,
    chapter_name VARCHAR(255) NOT NULL,
    description TEXT,
    specialization VARCHAR(100),
    fraud_rate DECIMAL(5,4) DEFAULT 0.0,
    best_model VARCHAR(50), -- 'xgboost', 'lightgbm', 'catboost'
    model_performance JSONB, -- {f1, auc, precision, recall}
    optimal_threshold DECIMAL(8,6), -- Seuil optimal calculé
    features_count INTEGER, -- Nombre de features utilisées
    data_size INTEGER, -- Taille du dataset d'entraînement
    advanced_fraud_detection BOOLEAN DEFAULT TRUE,
    business_features_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des modèles ML avec nouvelles métriques
CREATE TABLE models (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chapter_id VARCHAR(10) REFERENCES chapters(chapter_id),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'xgboost', 'catboost', 'lightgbm', 'randomforest', 'logisticregression'
    version VARCHAR(20) DEFAULT '2.0.0',
    training_date TIMESTAMP,
    performance_metrics JSONB, -- {f1, auc, precision, recall, optimal_threshold}
    feature_list JSONB, -- Liste complète des features
    hyperparameters JSONB,
    train_val_test_split JSONB, -- {train_size, val_size, test_size}
    shap_analysis JSONB, -- Résultats d'analyse SHAP
    model_path TEXT,
    is_best_model BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des features avec nouvelles catégories
CREATE TABLE features (
    feature_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    feature_name VARCHAR(100) NOT NULL UNIQUE,
    feature_type VARCHAR(50) NOT NULL, -- 'numerical', 'categorical', 'business', 'fraud_detection'
    feature_category VARCHAR(50), -- 'basic', 'business', 'fraud_detection', 'advanced'
    description TEXT,
    data_type VARCHAR(20), -- 'float', 'int', 'string', 'boolean'
    is_required BOOLEAN DEFAULT FALSE,
    default_value TEXT,
    validation_rules JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des associations chapitres-features
CREATE TABLE chapter_features (
    chapter_id VARCHAR(10) REFERENCES chapters(chapter_id),
    feature_id UUID REFERENCES features(feature_id),
    is_used BOOLEAN DEFAULT TRUE,
    importance_score DECIMAL(8,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (chapter_id, feature_id)
);

-- =============================================================================
-- 2. TABLES DE DONNÉES ET PRÉDICTIONS
-- =============================================================================

-- Table des déclarations douanières avec nouvelles colonnes
CREATE TABLE declarations (
    declaration_id VARCHAR(100) PRIMARY KEY,
    chapter_id VARCHAR(10) REFERENCES chapters(chapter_id),
    file_name VARCHAR(255),
    file_type VARCHAR(20), -- 'csv', 'pdf', 'image'
    source_type VARCHAR(50), -- 'ocr', 'manual', 'api'
    
    -- Données de base de la déclaration
    poids_net_kg DECIMAL(15,3),
    nombre_colis INTEGER,
    quantite_complement DECIMAL(15,3),
    taux_droits_percent DECIMAL(8,4),
    valeur_caf DECIMAL(15,2),
    valeur_unitaire_kg DECIMAL(15,6),
    ratio_douane_caf DECIMAL(8,6),
    
    -- Codes et classifications
    code_sh_complet VARCHAR(20),
    code_pays_origine VARCHAR(10),
    code_pays_provenance VARCHAR(10),
    regime_complet VARCHAR(50),
    statut_bae VARCHAR(50),
    type_regime VARCHAR(50),
    regime_douanier VARCHAR(50),
    regime_fiscal VARCHAR(50),
    
    -- Nouvelles colonnes pour le système avancé
    code_produit_str VARCHAR(100),
    pays_origine_str VARCHAR(100),
    pays_provenance_str VARCHAR(100),
    numero_article INTEGER,
    precision_uemoa INTEGER,
    
    -- Métadonnées
    extraction_status VARCHAR(50) DEFAULT 'success',
    validation_status VARCHAR(50) DEFAULT 'valid',
    processing_notes TEXT,
    raw_data JSONB,
    ocr_confidence DECIMAL(8,6),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des prédictions ML avec nouvelles colonnes
CREATE TABLE predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    declaration_id VARCHAR(100) REFERENCES declarations(declaration_id),
    model_id UUID REFERENCES models(model_id),
    chapter_id VARCHAR(10) REFERENCES chapters(chapter_id),
    
    -- Résultats de prédiction
    predicted_fraud BOOLEAN NOT NULL,
    fraud_probability DECIMAL(8,6) NOT NULL,
    confidence_score DECIMAL(8,6),
    decision VARCHAR(20), -- 'conforme', 'zone_grise', 'fraude'
    
    -- Nouvelles métadonnées de prédiction
    model_used VARCHAR(50), -- Nom du modèle utilisé
    optimal_threshold_used DECIMAL(8,6), -- Seuil optimal utilisé
    auc_score DECIMAL(8,6), -- AUC du modèle
    f1_score DECIMAL(8,6), -- F1 du modèle
    precision_score DECIMAL(8,6), -- Precision du modèle
    recall_score DECIMAL(8,6), -- Recall du modèle
    
    -- Métadonnées de prédiction
    ml_integration_used BOOLEAN DEFAULT FALSE,
    decision_source VARCHAR(50), -- 'ml', 'rl', 'hybrid', 'expert'
    context_features JSONB,
    risk_analysis JSONB,
    
    -- Seuils utilisés
    ml_threshold DECIMAL(8,6),
    rl_threshold DECIMAL(8,6),
    decision_threshold DECIMAL(8,6),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des features extraites avec nouvelles colonnes
CREATE TABLE declaration_features (
    feature_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    declaration_id VARCHAR(100) REFERENCES declarations(declaration_id),
    chapter_id VARCHAR(10) REFERENCES chapters(chapter_id),
    
    -- Features de base
    poids_net_kg DECIMAL(15,3),
    nombre_colis INTEGER,
    quantite_complement DECIMAL(15,3),
    taux_droits_percent DECIMAL(8,4),
    valeur_caf DECIMAL(15,2),
    valeur_unitaire_kg DECIMAL(15,6),
    ratio_douane_caf DECIMAL(8,6),
    
    -- Features catégorielles
    code_sh_complet VARCHAR(20),
    code_pays_origine VARCHAR(10),
    code_pays_provenance VARCHAR(10),
    regime_complet VARCHAR(50),
    statut_bae VARCHAR(50),
    type_regime VARCHAR(50),
    regime_douanier VARCHAR(50),
    regime_fiscal VARCHAR(50),
    
    -- Nouvelles features string
    code_produit_str VARCHAR(100),
    pays_origine_str VARCHAR(100),
    pays_provenance_str VARCHAR(100),
    numero_article INTEGER,
    precision_uemoa INTEGER,
    
    -- Features business spécifiques par chapitre
    -- Chapitre 30 (Pharmaceutique)
    business_glissement_cosmetique BOOLEAN DEFAULT FALSE,
    business_glissement_pays_cosmetiques BOOLEAN DEFAULT FALSE,
    business_glissement_ratio_suspect DECIMAL(8,6),
    business_is_medicament BOOLEAN DEFAULT FALSE,
    business_is_antipaludeen BOOLEAN DEFAULT FALSE,
    
    -- Chapitre 84 (Mécanique)
    business_glissement_machine BOOLEAN DEFAULT FALSE,
    business_glissement_pays_machines BOOLEAN DEFAULT FALSE,
    business_is_machine BOOLEAN DEFAULT FALSE,
    business_is_electronique BOOLEAN DEFAULT FALSE,
    
    -- Chapitre 85 (Électronique)
    business_glissement_electronique BOOLEAN DEFAULT FALSE,
    business_glissement_pays_electroniques BOOLEAN DEFAULT FALSE,
    business_poids_faible BOOLEAN DEFAULT FALSE,
    business_is_telephone BOOLEAN DEFAULT FALSE,
    
    -- Features communes
    business_glissement_ratio_suspect DECIMAL(8,6),
    business_risk_pays_high BOOLEAN DEFAULT FALSE,
    business_origine_diff_provenance BOOLEAN DEFAULT FALSE,
    business_valeur_elevee BOOLEAN DEFAULT FALSE,
    
    -- Features de détection de fraude avancée
    bienayme_chebychev_score DECIMAL(8,6),
    bienayme_chebychev_anomaly BOOLEAN DEFAULT FALSE,
    tei_calcule DECIMAL(8,6),
    mirror_tei_score DECIMAL(8,6),
    mirror_tei_deviation DECIMAL(8,6),
    mirror_tei_anomaly BOOLEAN DEFAULT FALSE,
    spectral_cluster_score DECIMAL(8,6),
    spectral_cluster_anomaly BOOLEAN DEFAULT FALSE,
    hierarchical_cluster_score DECIMAL(8,6),
    hierarchical_cluster_anomaly BOOLEAN DEFAULT FALSE,
    admin_values_score DECIMAL(8,6),
    admin_values_deviation DECIMAL(8,6),
    admin_values_anomaly BOOLEAN DEFAULT FALSE,
    composite_fraud_score DECIMAL(8,6),
    ratio_poids_valeur DECIMAL(8,6),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 3. TABLES SYSTÈME RL AVANCÉ
-- =============================================================================

-- Table des décisions RL avec nouvelles colonnes
CREATE TABLE rl_decisions (
    decision_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chapter_id VARCHAR(10) REFERENCES chapters(chapter_id),
    declaration_id VARCHAR(100) REFERENCES declarations(declaration_id),
    
    -- Décision RL
    action_taken VARCHAR(50) NOT NULL, -- 'inspect', 'pass', 'flag'
    confidence_level DECIMAL(8,6),
    context_complexity DECIMAL(8,6),
    bureau_risk_score DECIMAL(8,6),
    
    -- Nouvelles métadonnées RL
    bandit_arm_selected VARCHAR(50),
    epsilon_value DECIMAL(8,6),
    exploration_exploitation VARCHAR(20), -- 'exploration', 'exploitation'
    reward_received DECIMAL(8,6),
    regret DECIMAL(8,6),
    
    -- Contexte de décision
    context_features JSONB,
    inspector_profile_id VARCHAR(100),
    decision_reasoning TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des profils d'inspecteurs avec nouvelles colonnes
CREATE TABLE inspector_profiles (
    profile_id VARCHAR(100) PRIMARY KEY,
    chapter_id VARCHAR(10) REFERENCES chapters(chapter_id),
    inspector_name VARCHAR(255),
    experience_level VARCHAR(50), -- 'junior', 'senior', 'expert'
    
    -- Nouvelles métriques de performance
    total_decisions INTEGER DEFAULT 0,
    correct_decisions INTEGER DEFAULT 0,
    accuracy_rate DECIMAL(8,6),
    fraud_detection_rate DECIMAL(8,6),
    false_positive_rate DECIMAL(8,6),
    
    -- Préférences et spécialisations
    preferred_chapters JSONB,
    specialization_areas JSONB,
    risk_tolerance DECIMAL(8,6),
    
    -- Métadonnées
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table de l'historique des feedbacks avec nouvelles colonnes
CREATE TABLE feedback_history (
    feedback_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chapter_id VARCHAR(10) REFERENCES chapters(chapter_id),
    declaration_id VARCHAR(100) REFERENCES declarations(declaration_id),
    prediction_id UUID REFERENCES predictions(prediction_id),
    rl_decision_id UUID REFERENCES rl_decisions(decision_id),
    
    -- Feedback
    feedback_type VARCHAR(50) NOT NULL, -- 'correct', 'incorrect', 'partial'
    feedback_value DECIMAL(8,6), -- Valeur de récompense
    feedback_source VARCHAR(50), -- 'inspector', 'system', 'expert'
    
    -- Nouvelles métadonnées de feedback
    feedback_confidence DECIMAL(8,6),
    feedback_reasoning TEXT,
    inspector_id VARCHAR(100),
    validation_method VARCHAR(50),
    
    -- Métadonnées
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 4. TABLES D'ANALYSE ET MÉTRIQUES
-- =============================================================================

-- Table des résultats d'analyse avec nouvelles colonnes
CREATE TABLE analysis_results (
    analysis_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chapter_id VARCHAR(10) REFERENCES chapters(chapter_id),
    analysis_type VARCHAR(50) NOT NULL, -- 'fraud_detection', 'business_analysis', 'risk_assessment'
    
    -- Résultats d'analyse
    analysis_data JSONB,
    confidence_score DECIMAL(8,6),
    risk_level VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    
    -- Nouvelles métadonnées d'analyse
    analysis_method VARCHAR(50),
    features_used JSONB,
    model_version VARCHAR(20),
    processing_time_ms INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des seuils de modèles avec nouvelles colonnes
CREATE TABLE model_thresholds (
    threshold_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chapter_id VARCHAR(10) REFERENCES chapters(chapter_id),
    model_id UUID REFERENCES models(model_id),
    
    -- Seuils
    optimal_threshold DECIMAL(8,6) NOT NULL,
    conforme_threshold DECIMAL(8,6),
    fraude_threshold DECIMAL(8,6),
    zone_grise_min DECIMAL(8,6),
    zone_grise_max DECIMAL(8,6),
    
    -- Nouvelles métadonnées de seuils
    threshold_method VARCHAR(50), -- 'precision_recall', 'f1_optimization', 'roc_optimization'
    validation_metrics JSONB,
    threshold_confidence DECIMAL(8,6),
    
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des métriques de performance avec nouvelles colonnes
CREATE TABLE performance_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chapter_id VARCHAR(10) REFERENCES chapters(chapter_id),
    model_id UUID REFERENCES models(model_id),
    
    -- Métriques de base
    f1_score DECIMAL(8,6),
    auc_score DECIMAL(8,6),
    precision_score DECIMAL(8,6),
    recall_score DECIMAL(8,6),
    accuracy DECIMAL(8,6),
    
    -- Nouvelles métriques avancées
    specificity DECIMAL(8,6),
    sensitivity DECIMAL(8,6),
    matthews_correlation DECIMAL(8,6),
    cohen_kappa DECIMAL(8,6),
    
    -- Métriques de validation
    train_f1 DECIMAL(8,6),
    val_f1 DECIMAL(8,6),
    test_f1 DECIMAL(8,6),
    train_auc DECIMAL(8,6),
    val_auc DECIMAL(8,6),
    test_auc DECIMAL(8,6),
    
    -- Métadonnées
    evaluation_date TIMESTAMP,
    dataset_size INTEGER,
    evaluation_method VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table des logs système avec nouvelles colonnes
CREATE TABLE system_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chapter_id VARCHAR(10) REFERENCES chapters(chapter_id),
    
    -- Log
    log_level VARCHAR(20) NOT NULL, -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    log_message TEXT NOT NULL,
    log_source VARCHAR(100), -- Module qui a généré le log
    
    -- Nouvelles métadonnées de log
    execution_time_ms INTEGER,
    memory_usage_mb DECIMAL(10,2),
    cpu_usage_percent DECIMAL(5,2),
    error_code VARCHAR(50),
    stack_trace TEXT,
    
    -- Contexte
    user_id VARCHAR(100),
    session_id VARCHAR(100),
    request_id VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- 5. INDEX POUR OPTIMISATION DES PERFORMANCES
-- =============================================================================

-- Index sur les clés étrangères
CREATE INDEX idx_declarations_chapter_id ON declarations(chapter_id);
CREATE INDEX idx_predictions_declaration_id ON predictions(declaration_id);
CREATE INDEX idx_predictions_model_id ON predictions(model_id);
CREATE INDEX idx_predictions_chapter_id ON predictions(chapter_id);
CREATE INDEX idx_declaration_features_declaration_id ON declaration_features(declaration_id);
CREATE INDEX idx_rl_decisions_chapter_id ON rl_decisions(chapter_id);
CREATE INDEX idx_feedback_history_chapter_id ON feedback_history(chapter_id);

-- Index sur les colonnes de recherche fréquente
CREATE INDEX idx_declarations_created_at ON declarations(created_at);
CREATE INDEX idx_predictions_created_at ON predictions(created_at);
CREATE INDEX idx_declarations_file_type ON declarations(file_type);
CREATE INDEX idx_predictions_decision ON predictions(decision);
CREATE INDEX idx_predictions_fraud_probability ON predictions(fraud_probability);

-- Index sur les nouvelles colonnes importantes
CREATE INDEX idx_declarations_code_sh ON declarations(code_sh_complet);
CREATE INDEX idx_declarations_pays_origine ON declarations(code_pays_origine);
CREATE INDEX idx_declaration_features_fraud_scores ON declaration_features(composite_fraud_score);
CREATE INDEX idx_rl_decisions_action ON rl_decisions(action_taken);
CREATE INDEX idx_performance_metrics_evaluation_date ON performance_metrics(evaluation_date);

-- Index composites pour les requêtes complexes
CREATE INDEX idx_predictions_chapter_decision ON predictions(chapter_id, decision);
CREATE INDEX idx_declarations_chapter_created ON declarations(chapter_id, created_at);
CREATE INDEX idx_feedback_history_chapter_type ON feedback_history(chapter_id, feedback_type);

-- =============================================================================
-- 6. VUES UTILITAIRES
-- =============================================================================

-- Vue combinée déclarations-prédictions
CREATE VIEW declarations_with_predictions AS
SELECT 
    d.*,
    p.prediction_id,
    p.predicted_fraud,
    p.fraud_probability,
    p.decision,
    p.model_used,
    p.optimal_threshold_used,
    p.auc_score,
    p.f1_score,
    p.precision_score,
    p.recall_score,
    p.created_at as prediction_created_at
FROM declarations d
LEFT JOIN predictions p ON d.declaration_id = p.declaration_id;

-- Vue des statistiques par chapitre
CREATE VIEW chapter_statistics AS
SELECT 
    c.chapter_id,
    c.chapter_name,
    c.best_model,
    c.fraud_rate,
    c.features_count,
    COUNT(d.declaration_id) as total_declarations,
    COUNT(p.prediction_id) as total_predictions,
    AVG(p.fraud_probability) as avg_fraud_probability,
    COUNT(CASE WHEN p.decision = 'fraude' THEN 1 END) as fraud_predictions,
    COUNT(CASE WHEN p.decision = 'conforme' THEN 1 END) as conform_predictions,
    COUNT(CASE WHEN p.decision = 'zone_grise' THEN 1 END) as gray_zone_predictions
FROM chapters c
LEFT JOIN declarations d ON c.chapter_id = d.chapter_id
LEFT JOIN predictions p ON d.declaration_id = p.declaration_id
GROUP BY c.chapter_id, c.chapter_name, c.best_model, c.fraud_rate, c.features_count;

-- Vue des performances des modèles
CREATE VIEW model_performance_summary AS
SELECT 
    m.model_id,
    m.model_name,
    m.model_type,
    c.chapter_name,
    pm.f1_score,
    pm.auc_score,
    pm.precision_score,
    pm.recall_score,
    pm.accuracy,
    pm.evaluation_date,
    m.is_best_model
FROM models m
JOIN chapters c ON m.chapter_id = c.chapter_id
LEFT JOIN performance_metrics pm ON m.model_id = pm.model_id
WHERE m.is_active = TRUE;

-- =============================================================================
-- 7. TRIGGERS POUR MISE À JOUR AUTOMATIQUE
-- =============================================================================

-- Fonction pour mettre à jour updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers pour updated_at
CREATE TRIGGER update_chapters_updated_at BEFORE UPDATE ON chapters
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_declarations_updated_at BEFORE UPDATE ON declarations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_inspector_profiles_updated_at BEFORE UPDATE ON inspector_profiles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_model_thresholds_updated_at BEFORE UPDATE ON model_thresholds
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- 8. DONNÉES INITIALES
-- =============================================================================

-- Insertion des chapitres avec nouvelles métriques
INSERT INTO chapters (chapter_id, chapter_number, chapter_name, description, specialization, fraud_rate, best_model, model_performance, optimal_threshold, features_count, data_size, business_features_count) VALUES
('chap30', 30, 'Produits pharmaceutiques', 'Produits pharmaceutiques et cosmétiques', 'Pharmaceutique', 0.15, 'XGBoost', '{"f1": 0.9811, "auc": 0.9997, "precision": 0.9987, "recall": 0.9746}', 0.23, 52, 50000, 5),
('chap84', 84, 'Machines et appareils mécaniques', 'Machines et équipements mécaniques', 'Mécanique', 0.12, 'XGBoost', '{"f1": 0.9888, "auc": 0.9997, "precision": 0.9992, "recall": 0.9834}', 0.22, 54, 75000, 5),
('chap85', 85, 'Machines et appareils électriques', 'Appareils électriques et électroniques', 'Électronique', 0.18, 'LightGBM', '{"f1": 0.9791, "auc": 0.9995, "precision": 0.9985, "recall": 0.9872}', 0.22, 54, 60000, 6);

-- Insertion des features de base
INSERT INTO features (feature_name, feature_type, feature_category, description, data_type, is_required) VALUES
-- Features de base
('POIDS_NET_KG', 'numerical', 'basic', 'Poids net en kilogrammes', 'float', TRUE),
('NOMBRE_COLIS', 'numerical', 'basic', 'Nombre de colis', 'int', TRUE),
('QUANTITE_COMPLEMENT', 'numerical', 'basic', 'Quantité complémentaire', 'float', TRUE),
('TAUX_DROITS_PERCENT', 'numerical', 'basic', 'Taux de droits en pourcentage', 'float', TRUE),
('VALEUR_CAF', 'numerical', 'basic', 'Valeur CAF', 'float', TRUE),
('VALEUR_UNITAIRE_KG', 'numerical', 'basic', 'Valeur unitaire par kg', 'float', TRUE),
('RATIO_DOUANE_CAF', 'numerical', 'basic', 'Ratio douane/CAF', 'float', TRUE),

-- Features catégorielles
('CODE_SH_COMPLET', 'categorical', 'basic', 'Code SH complet', 'string', TRUE),
('CODE_PAYS_ORIGINE', 'categorical', 'basic', 'Code pays d''origine', 'string', TRUE),
('CODE_PAYS_PROVENANCE', 'categorical', 'basic', 'Code pays de provenance', 'string', TRUE),
('REGIME_COMPLET', 'categorical', 'basic', 'Régime complet', 'string', TRUE),
('STATUT_BAE', 'categorical', 'basic', 'Statut BAE', 'string', TRUE),
('TYPE_REGIME', 'categorical', 'basic', 'Type de régime', 'string', TRUE),
('REGIME_DOUANIER', 'categorical', 'basic', 'Régime douanier', 'string', TRUE),
('REGIME_FISCAL', 'categorical', 'basic', 'Régime fiscal', 'string', TRUE),

-- Features string
('CODE_PRODUIT_STR', 'categorical', 'basic', 'Code produit en string', 'string', TRUE),
('PAYS_ORIGINE_STR', 'categorical', 'basic', 'Pays d''origine en string', 'string', TRUE),
('PAYS_PROVENANCE_STR', 'categorical', 'basic', 'Pays de provenance en string', 'string', TRUE),
('NUMERO_ARTICLE', 'numerical', 'basic', 'Numéro d''article', 'int', TRUE),
('PRECISION_UEMOA', 'numerical', 'basic', 'Précision UEMOA', 'int', TRUE);

-- Insertion des features business pour le chapitre 30
INSERT INTO features (feature_name, feature_type, feature_category, description, data_type, is_required) VALUES
('BUSINESS_GLISSEMENT_COSMETIQUE', 'business', 'business', 'Glissement tarifaire cosmétique', 'boolean', FALSE),
('BUSINESS_GLISSEMENT_PAYS_COSMETIQUES', 'business', 'business', 'Glissement pays cosmétiques', 'boolean', FALSE),
('BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'business', 'business', 'Ratio suspect de glissement', 'float', FALSE),
('BUSINESS_IS_MEDICAMENT', 'business', 'business', 'Est un médicament', 'boolean', FALSE),
('BUSINESS_IS_ANTIPALUDEEN', 'business', 'business', 'Est un antipaluéen', 'boolean', FALSE);

-- Insertion des features business pour le chapitre 84
INSERT INTO features (feature_name, feature_type, feature_category, description, data_type, is_required) VALUES
('BUSINESS_GLISSEMENT_MACHINE', 'business', 'business', 'Glissement tarifaire machine', 'boolean', FALSE),
('BUSINESS_GLISSEMENT_PAYS_MACHINES', 'business', 'business', 'Glissement pays machines', 'boolean', FALSE),
('BUSINESS_IS_MACHINE', 'business', 'business', 'Est une machine', 'boolean', FALSE),
('BUSINESS_IS_ELECTRONIQUE', 'business', 'business', 'Est électronique', 'boolean', FALSE);

-- Insertion des features business pour le chapitre 85
INSERT INTO features (feature_name, feature_type, feature_category, description, data_type, is_required) VALUES
('BUSINESS_GLISSEMENT_ELECTRONIQUE', 'business', 'business', 'Glissement tarifaire électronique', 'boolean', FALSE),
('BUSINESS_GLISSEMENT_PAYS_ELECTRONIQUES', 'business', 'business', 'Glissement pays électroniques', 'boolean', FALSE),
('BUSINESS_POIDS_FAIBLE', 'business', 'business', 'Poids faible suspect', 'boolean', FALSE),
('BUSINESS_IS_TELEPHONE', 'business', 'business', 'Est un téléphone', 'boolean', FALSE);

-- Insertion des features de détection de fraude avancée
INSERT INTO features (feature_name, feature_type, feature_category, description, data_type, is_required) VALUES
('BIENAYME_CHEBYCHEV_SCORE', 'fraud_detection', 'fraud_detection', 'Score Bienaymé-Tchebychev', 'float', FALSE),
('BIENAYME_CHEBYCHEV_ANOMALY', 'fraud_detection', 'fraud_detection', 'Anomalie Bienaymé-Tchebychev', 'boolean', FALSE),
('TEI_CALCULE', 'fraud_detection', 'fraud_detection', 'TEI calculé', 'float', FALSE),
('MIRROR_TEI_SCORE', 'fraud_detection', 'fraud_detection', 'Score analyse miroir TEI', 'float', FALSE),
('MIRROR_TEI_DEVIATION', 'fraud_detection', 'fraud_detection', 'Déviation analyse miroir TEI', 'float', FALSE),
('MIRROR_TEI_ANOMALY', 'fraud_detection', 'fraud_detection', 'Anomalie analyse miroir TEI', 'boolean', FALSE),
('SPECTRAL_CLUSTER_SCORE', 'fraud_detection', 'fraud_detection', 'Score clustering spectral', 'float', FALSE),
('SPECTRAL_CLUSTER_ANOMALY', 'fraud_detection', 'fraud_detection', 'Anomalie clustering spectral', 'boolean', FALSE),
('HIERARCHICAL_CLUSTER_SCORE', 'fraud_detection', 'fraud_detection', 'Score clustering hiérarchique', 'float', FALSE),
('HIERARCHICAL_CLUSTER_ANOMALY', 'fraud_detection', 'fraud_detection', 'Anomalie clustering hiérarchique', 'boolean', FALSE),
('ADMIN_VALUES_SCORE', 'fraud_detection', 'fraud_detection', 'Score valeurs administrées', 'float', FALSE),
('ADMIN_VALUES_DEVIATION', 'fraud_detection', 'fraud_detection', 'Déviation valeurs administrées', 'float', FALSE),
('ADMIN_VALUES_ANOMALY', 'fraud_detection', 'fraud_detection', 'Anomalie valeurs administrées', 'boolean', FALSE),
('COMPOSITE_FRAUD_SCORE', 'fraud_detection', 'fraud_detection', 'Score composite de fraude', 'float', FALSE),
('RATIO_POIDS_VALEUR', 'fraud_detection', 'fraud_detection', 'Ratio poids/valeur', 'float', FALSE);

-- Insertion des features communes
INSERT INTO features (feature_name, feature_type, feature_category, description, data_type, is_required) VALUES
('BUSINESS_RISK_PAYS_HIGH', 'business', 'business', 'Pays à haut risque', 'boolean', FALSE),
('BUSINESS_ORIGINE_DIFF_PROVENANCE', 'business', 'business', 'Origine différente de provenance', 'boolean', FALSE),
('BUSINESS_VALEUR_ELEVEE', 'business', 'business', 'Valeur élevée suspecte', 'boolean', FALSE);

-- =============================================================================
-- 9. COMMENTAIRES FINAUX
-- =============================================================================

COMMENT ON DATABASE INSPECT_IA IS 'Base de données pour le système de détection de fraude douanière INSPECT_IA avec ML-RL avancé';
COMMENT ON TABLE chapters IS 'Chapitres douaniers avec métriques de performance des modèles ML';
COMMENT ON TABLE models IS 'Modèles ML avec métriques de performance et seuils optimaux';
COMMENT ON TABLE declarations IS 'Déclarations douanières avec toutes les features de base et avancées';
COMMENT ON TABLE predictions IS 'Prédictions ML avec métadonnées complètes et seuils utilisés';
COMMENT ON TABLE declaration_features IS 'Features extraites par déclaration incluant business et fraud detection';
COMMENT ON TABLE rl_decisions IS 'Décisions du système RL avec métadonnées de bandit et contexte';
COMMENT ON TABLE inspector_profiles IS 'Profils d''inspecteurs avec métriques de performance';
COMMENT ON TABLE feedback_history IS 'Historique des feedbacks avec validation et métadonnées';
COMMENT ON TABLE analysis_results IS 'Résultats d''analyse avec méthodes et métriques';
COMMENT ON TABLE model_thresholds IS 'Seuils de modèles avec méthodes de calcul et validation';
COMMENT ON TABLE performance_metrics IS 'Métriques de performance avec validation train/val/test';
COMMENT ON TABLE system_logs IS 'Logs système avec métriques de performance et contexte';
