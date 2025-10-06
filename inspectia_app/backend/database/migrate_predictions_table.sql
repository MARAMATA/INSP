-- =============================================================================
-- MIGRATION: Ajouter les colonnes manquantes à la table predictions
-- =============================================================================

-- Ajouter les colonnes manquantes à la table predictions
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS model_version VARCHAR(20);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS processing_timestamp TIMESTAMP;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS threshold_used DECIMAL(8,6);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS feature_importance JSONB;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS explanation TEXT;

-- Afficher la structure de la table après migration
\d predictions;
