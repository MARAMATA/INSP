"""
Modèles SQLAlchemy pour INSPECT_IA
Système de détection de fraude douanière avec ML-RL avancé
Définition des tables et relations pour la base de données PostgreSQL
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, DateTime, 
    ForeignKey, JSON, DECIMAL, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class Chapter(Base):
    """Table des chapitres douaniers"""
    __tablename__ = 'chapters'
    
    chapter_id = Column(String(10), primary_key=True)
    chapter_number = Column(Integer, unique=True, nullable=False)
    chapter_name = Column(String(255), nullable=False)
    description = Column(Text)
    specialization = Column(String(100))
    fraud_rate = Column(DECIMAL(5, 4), default=0.0)
    best_model = Column(String(50))  # 'xgboost', 'lightgbm', 'catboost'
    model_performance = Column(JSON)  # {f1, auc, precision, recall}
    optimal_threshold = Column(DECIMAL(8, 6))  # Seuil optimal calculé
    features_count = Column(Integer)  # Nombre de features utilisées
    data_size = Column(Integer)  # Taille du dataset d'entraînement
    advanced_fraud_detection = Column(Boolean, default=True)
    business_features_count = Column(Integer)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relations
    models = relationship("Model", back_populates="chapter")
    declarations = relationship("Declaration", back_populates="chapter")
    predictions = relationship("Prediction", back_populates="chapter")
    rl_decisions = relationship("RLDecision", back_populates="chapter")
    feedback_history = relationship("FeedbackHistory", back_populates="chapter")
    analysis_results = relationship("AnalysisResult", back_populates="chapter")
    model_thresholds = relationship("ModelThreshold", back_populates="chapter")
    performance_metrics = relationship("PerformanceMetric", back_populates="chapter")
    system_logs = relationship("SystemLog", back_populates="chapter")
    chapter_features = relationship("ChapterFeature", back_populates="chapter")

class Model(Base):
    """Table des modèles ML"""
    __tablename__ = 'models'
    
    model_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # 'xgboost', 'catboost', 'lightgbm', 'randomforest', 'logisticregression'
    version = Column(String(20), default='2.0.0')
    training_date = Column(DateTime)
    performance_metrics = Column(JSON)  # {f1, auc, precision, recall, optimal_threshold}
    feature_list = Column(JSON)  # Liste complète des features
    hyperparameters = Column(JSON)
    train_val_test_split = Column(JSON)  # {train_size, val_size, test_size}
    shap_analysis = Column(JSON)  # Résultats d'analyse SHAP
    model_path = Column(Text)
    is_best_model = Column(Boolean, default=False)  # Indique si c'est le meilleur modèle
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relations
    chapter = relationship("Chapter", back_populates="models")
    predictions = relationship("Prediction", back_populates="model")
    model_thresholds = relationship("ModelThreshold", back_populates="model")
    performance_metrics = relationship("PerformanceMetric", back_populates="model")

class Feature(Base):
    """Table des features"""
    __tablename__ = 'features'
    
    feature_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    feature_name = Column(String(100), unique=True, nullable=False)
    feature_type = Column(String(50), nullable=False)  # 'numerical', 'categorical', 'business', 'fraud_detection'
    feature_category = Column(String(50))  # 'basic', 'business', 'fraud_detection', 'advanced'
    description = Column(Text)
    data_type = Column(String(20))  # 'float', 'int', 'string', 'boolean'
    is_required = Column(Boolean, default=False)
    default_value = Column(Text)
    validation_rules = Column(JSON)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    chapter_features = relationship("ChapterFeature", back_populates="feature")
    declaration_features = relationship("DeclarationFeature", back_populates="feature")

class ChapterFeature(Base):
    """Table d'association chapitres-features"""
    __tablename__ = 'chapter_features'
    
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), primary_key=True)
    feature_id = Column(UUID(as_uuid=True), ForeignKey('features.feature_id'), primary_key=True)
    is_required = Column(Boolean, default=True)
    feature_order = Column(Integer)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    chapter = relationship("Chapter", back_populates="chapter_features")
    feature = relationship("Feature", back_populates="chapter_features")

class Declaration(Base):
    """Table des déclarations douanières"""
    __tablename__ = 'declarations'
    
    declaration_id = Column(String(100), primary_key=True)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    file_name = Column(String(255))
    file_type = Column(String(20))  # 'csv', 'pdf', 'image'
    source_type = Column(String(50))  # 'ocr', 'manual', 'api'
    
    # Données de base
    poids_net_kg = Column(DECIMAL(15, 3))
    nombre_colis = Column(Integer)
    quantite_complement = Column(DECIMAL(15, 3))
    taux_droits_percent = Column(DECIMAL(8, 4))
    valeur_caf = Column(DECIMAL(15, 2))
    valeur_unitaire_kg = Column(DECIMAL(15, 6))
    ratio_douane_caf = Column(DECIMAL(8, 6))
    
    # Codes et classifications
    code_sh_complet = Column(String(20))
    code_pays_origine = Column(String(10))
    code_pays_provenance = Column(String(10))
    regime_complet = Column(String(50))
    statut_bae = Column(String(50))
    type_regime = Column(String(50))
    regime_douanier = Column(String(50))
    regime_fiscal = Column(String(50))
    
    # Nouvelles colonnes pour le système avancé
    code_produit_str = Column(String(100))
    pays_origine_str = Column(String(100))
    pays_provenance_str = Column(String(100))
    numero_article = Column(Integer)
    precision_uemoa = Column(Integer)
    
    # Métadonnées
    extraction_status = Column(String(50), default='success')
    validation_status = Column(String(50), default='valid')
    processing_notes = Column(Text)
    raw_data = Column(JSON)
    ocr_confidence = Column(DECIMAL(8, 6))
    
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relations
    chapter = relationship("Chapter", back_populates="declarations")
    predictions = relationship("Prediction", back_populates="declaration")
    rl_decisions = relationship("RLDecision", back_populates="declaration")
    feedback_history = relationship("FeedbackHistory", back_populates="declaration")
    analysis_results = relationship("AnalysisResult", back_populates="declaration")
    declaration_features = relationship("DeclarationFeature", back_populates="declaration")

class Prediction(Base):
    """Table des prédictions ML"""
    __tablename__ = 'predictions'
    
    prediction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    declaration_id = Column(String(100), ForeignKey('declarations.declaration_id'), nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.model_id'), nullable=False)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    
    # Résultats de prédiction
    predicted_fraud = Column(Boolean, nullable=False)
    fraud_probability = Column(DECIMAL(8, 6), nullable=False)
    confidence_score = Column(DECIMAL(8, 6))
    decision = Column(String(20))  # 'conforme', 'zone_grise', 'fraude'
    
    # Nouvelles métadonnées de prédiction
    model_used = Column(String(50))  # Nom du modèle utilisé
    optimal_threshold_used = Column(DECIMAL(8, 6))  # Seuil optimal utilisé
    auc_score = Column(DECIMAL(8, 6))  # AUC du modèle
    f1_score = Column(DECIMAL(8, 6))  # F1 du modèle
    precision_score = Column(DECIMAL(8, 6))  # Precision du modèle
    recall_score = Column(DECIMAL(8, 6))  # Recall du modèle
    
    # Métadonnées
    ml_integration_used = Column(Boolean, default=False)
    decision_source = Column(String(50))  # 'ml', 'rl', 'hybrid', 'expert'
    context_features = Column(JSON)
    risk_analysis = Column(JSON)
    
    # Seuils utilisés
    ml_threshold = Column(DECIMAL(8, 6))
    rl_threshold = Column(DECIMAL(8, 6))
    decision_threshold = Column(DECIMAL(8, 6))
    
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    declaration = relationship("Declaration", back_populates="predictions")
    model = relationship("Model", back_populates="predictions")
    chapter = relationship("Chapter", back_populates="predictions")

class DeclarationFeature(Base):
    """Table des features extraites pour chaque déclaration"""
    __tablename__ = 'declaration_features'
    
    feature_id = Column(UUID(as_uuid=True), ForeignKey('features.feature_id'), primary_key=True)
    declaration_id = Column(String(100), ForeignKey('declarations.declaration_id'), primary_key=True)
    feature_value = Column(Text, nullable=False)
    feature_numeric_value = Column(DECIMAL(20, 6))
    is_activated = Column(Boolean, default=False)  # Pour les features business
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    feature = relationship("Feature", back_populates="declaration_features")
    declaration = relationship("Declaration", back_populates="declaration_features")

class RLDecision(Base):
    """Table des décisions RL"""
    __tablename__ = 'rl_decisions'
    
    decision_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    declaration_id = Column(String(100), ForeignKey('declarations.declaration_id'), nullable=False)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    
    # Contexte
    context_key = Column(String(255))
    context_json = Column(JSON)
    
    # Décision RL
    action = Column(String(50), nullable=False)
    model_probability = Column(DECIMAL(8, 6))
    rl_probability = Column(DECIMAL(8, 6), nullable=False)
    exploration_used = Column(Boolean, default=False)
    decision_source = Column(String(50))
    
    # Métriques
    confidence_score = Column(DECIMAL(8, 6), default=0.5)
    context_complexity = Column(Integer, default=1)
    seasonal_factor = Column(DECIMAL(8, 6), default=1.0)
    bureau_risk_score = Column(DECIMAL(8, 6), default=0.5)
    
    # Métadonnées
    extra_json = Column(JSON)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    declaration = relationship("Declaration", back_populates="rl_decisions")
    chapter = relationship("Chapter", back_populates="rl_decisions")

class InspectorProfile(Base):
    """Table des profils d'inspecteurs"""
    __tablename__ = 'inspector_profiles'
    
    inspector_id = Column(String(100), primary_key=True)
    name = Column(String(255))
    expertise_level = Column(String(20), default='standard')  # 'junior', 'standard', 'senior', 'expert'
    specialization = Column(JSON)  # Array des chapitres spécialisés
    total_reviews = Column(Integer, default=0)
    accuracy_rate = Column(DECIMAL(5, 4), default=0.0)
    avg_confidence = Column(DECIMAL(5, 4), default=0.5)
    avg_review_time = Column(DECIMAL(8, 2), default=300.0)
    performance_trend = Column(String(20), default='stable')
    created_at = Column(DateTime, default=func.current_timestamp())
    last_active = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    feedback_history = relationship("FeedbackHistory", back_populates="inspector")

class FeedbackHistory(Base):
    """Table des feedbacks"""
    __tablename__ = 'feedback_history'
    
    feedback_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    declaration_id = Column(String(100), ForeignKey('declarations.declaration_id'), nullable=False)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    inspector_id = Column(String(100), ForeignKey('inspector_profiles.inspector_id'), nullable=False)
    
    # Décisions
    inspector_decision = Column(Boolean, nullable=False)
    inspector_confidence = Column(DECIMAL(5, 4), nullable=False)
    predicted_fraud = Column(Boolean, nullable=False)
    predicted_probability = Column(DECIMAL(8, 6), nullable=False)
    predicted_action = Column(String(50))
    
    # Contexte
    context_key = Column(String(255))
    context_json = Column(JSON)
    exploration_used = Column(Boolean)
    notes = Column(Text)
    
    # Métriques de qualité
    feedback_quality_score = Column(DECIMAL(5, 4), default=0.8)
    inspector_expertise_level = Column(String(20), default='standard')
    review_time_seconds = Column(Integer)
    feedback_category = Column(String(20), default='regular')  # 'regular', 'urgent', 'review', 'audit'
    similar_cases_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    declaration = relationship("Declaration", back_populates="feedback_history")
    chapter = relationship("Chapter", back_populates="feedback_history")
    inspector = relationship("InspectorProfile", back_populates="feedback_history")

class AnalysisResult(Base):
    """Table des résultats d'analyse détaillée"""
    __tablename__ = 'analysis_results'
    
    analysis_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    declaration_id = Column(String(100), ForeignKey('declarations.declaration_id'), nullable=False)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    
    # Analyse de risque
    risk_score = Column(DECIMAL(8, 6))
    risk_level = Column(String(20))  # 'low', 'medium', 'high', 'critical'
    
    # Features business
    activated_business_features = Column(JSON)
    business_analysis = Column(JSON)
    
    # Incohérences
    inconsistencies = Column(JSON)
    inconsistency_count = Column(Integer, default=0)
    
    # Métadonnées
    analysis_type = Column(String(50), default='standard')
    analysis_confidence = Column(DECIMAL(8, 6))
    processing_time_ms = Column(Integer)
    
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    declaration = relationship("Declaration", back_populates="analysis_results")
    chapter = relationship("Chapter", back_populates="analysis_results")

class ModelThreshold(Base):
    """Table des seuils et configurations"""
    __tablename__ = 'model_thresholds'
    
    threshold_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.model_id'), nullable=False)
    
    # Seuils de décision
    conforme_threshold = Column(DECIMAL(8, 6))
    fraude_threshold = Column(DECIMAL(8, 6))
    optimal_threshold = Column(DECIMAL(8, 6))
    ml_threshold = Column(DECIMAL(8, 6))
    decision_threshold = Column(DECIMAL(8, 6))
    rl_threshold = Column(DECIMAL(8, 6))
    
    # Qualité de calibration
    calibration_quality = Column(String(20))  # 'EXCELLENT', 'GOOD', 'FAIR', 'POOR'
    brier_score = Column(DECIMAL(8, 6))
    ece = Column(DECIMAL(8, 6))  # Expected Calibration Error
    bss = Column(DECIMAL(8, 6))  # Brier Skill Score
    
    # Performance
    performance_metrics = Column(JSON)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relations
    chapter = relationship("Chapter", back_populates="model_thresholds")
    model = relationship("Model", back_populates="model_thresholds")

class PerformanceMetric(Base):
    """Table des métriques de performance"""
    __tablename__ = 'performance_metrics'
    
    metric_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.model_id'), nullable=False)
    metric_type = Column(String(50), nullable=False)  # 'accuracy', 'precision', 'recall', 'f1_score', 'auc'
    metric_value = Column(DECIMAL(10, 6), nullable=False)
    metric_period = Column(String(20))  # 'daily', 'weekly', 'monthly'
    calculation_date = Column(DateTime)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    chapter = relationship("Chapter", back_populates="performance_metrics")
    model = relationship("Model", back_populates="performance_metrics")

class SystemLog(Base):
    """Table des logs système"""
    __tablename__ = 'system_logs'
    
    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    log_level = Column(String(20), nullable=False)  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component = Column(String(100))  # 'ocr_pipeline', 'ml_model', 'rl_system', 'api'
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'))
    message = Column(Text, nullable=False)
    details = Column(JSON)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)
    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)

    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)


"""
Modèles SQLAlchemy pour InspectIA
Définition des tables et relations pour la base de données PostgreSQL
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, Text, DateTime, 
    ForeignKey, JSON, DECIMAL, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class Chapter(Base):
    """Table des chapitres douaniers"""
    __tablename__ = 'chapters'
    
    chapter_id = Column(String(10), primary_key=True)
    chapter_number = Column(Integer, unique=True, nullable=False)
    chapter_name = Column(String(255), nullable=False)
    description = Column(Text)
    specialization = Column(String(100))
    fraud_rate = Column(DECIMAL(5, 4), default=0.0)
    best_model = Column(String(50))
    model_performance = Column(JSON)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relations
    models = relationship("Model", back_populates="chapter")
    declarations = relationship("Declaration", back_populates="chapter")
    predictions = relationship("Prediction", back_populates="chapter")
    rl_decisions = relationship("RLDecision", back_populates="chapter")
    feedback_history = relationship("FeedbackHistory", back_populates="chapter")
    analysis_results = relationship("AnalysisResult", back_populates="chapter")
    model_thresholds = relationship("ModelThreshold", back_populates="chapter")
    performance_metrics = relationship("PerformanceMetric", back_populates="chapter")
    system_logs = relationship("SystemLog", back_populates="chapter")
    chapter_features = relationship("ChapterFeature", back_populates="chapter")

class Model(Base):
    """Table des modèles ML"""
    __tablename__ = 'models'
    
    model_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    model_name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)  # 'xgboost', 'catboost', etc.
    version = Column(String(20), default='1.0.0')
    training_date = Column(DateTime)
    performance_metrics = Column(JSON)
    feature_list = Column(JSON)
    hyperparameters = Column(JSON)
    calibration_info = Column(JSON)
    model_path = Column(Text)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relations
    chapter = relationship("Chapter", back_populates="models")
    predictions = relationship("Prediction", back_populates="model")
    model_thresholds = relationship("ModelThreshold", back_populates="model")
    performance_metrics = relationship("PerformanceMetric", back_populates="model")

class Feature(Base):
    """Table des features"""
    __tablename__ = 'features'
    
    feature_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    feature_name = Column(String(100), unique=True, nullable=False)
    feature_type = Column(String(50), nullable=False)  # 'numerical', 'categorical', 'business'
    description = Column(Text)
    data_type = Column(String(20))  # 'float', 'int', 'string', 'boolean'
    is_business_feature = Column(Boolean, default=False)
    calculation_formula = Column(Text)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    chapter_features = relationship("ChapterFeature", back_populates="feature")
    declaration_features = relationship("DeclarationFeature", back_populates="feature")

class ChapterFeature(Base):
    """Table d'association chapitres-features"""
    __tablename__ = 'chapter_features'
    
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), primary_key=True)
    feature_id = Column(UUID(as_uuid=True), ForeignKey('features.feature_id'), primary_key=True)
    is_required = Column(Boolean, default=True)
    feature_order = Column(Integer)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    chapter = relationship("Chapter", back_populates="chapter_features")
    feature = relationship("Feature", back_populates="chapter_features")

class Declaration(Base):
    """Table des déclarations douanières"""
    __tablename__ = 'declarations'
    
    declaration_id = Column(String(100), primary_key=True)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    file_name = Column(String(255))
    file_type = Column(String(20))  # 'csv', 'pdf', 'image'
    source_type = Column(String(50))  # 'ocr', 'manual', 'api'
    
    # Données de base
    poids_net_kg = Column(DECIMAL(15, 3))
    nombre_colis = Column(Integer)
    quantite_complement = Column(DECIMAL(15, 3))
    taux_droits_percent = Column(DECIMAL(8, 4))
    valeur_caf = Column(DECIMAL(15, 2))
    
    # Codes et classifications
    code_sh_complet = Column(String(20))
    code_pays_origine = Column(String(10))
    code_pays_provenance = Column(String(10))
    regime_complet = Column(String(50))
    statut_bae = Column(String(50))
    type_regime = Column(String(50))
    regime_douanier = Column(String(50))
    regime_fiscal = Column(String(50))
    
    # Métadonnées
    extraction_status = Column(String(50), default='success')
    validation_status = Column(String(50), default='valid')
    processing_notes = Column(Text)
    raw_data = Column(JSON)
    
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relations
    chapter = relationship("Chapter", back_populates="declarations")
    predictions = relationship("Prediction", back_populates="declaration")
    rl_decisions = relationship("RLDecision", back_populates="declaration")
    feedback_history = relationship("FeedbackHistory", back_populates="declaration")
    analysis_results = relationship("AnalysisResult", back_populates="declaration")
    declaration_features = relationship("DeclarationFeature", back_populates="declaration")

class Prediction(Base):
    """Table des prédictions ML"""
    __tablename__ = 'predictions'
    
    prediction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    declaration_id = Column(String(100), ForeignKey('declarations.declaration_id'), nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.model_id'), nullable=False)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    
    # Résultats de prédiction
    predicted_fraud = Column(Boolean, nullable=False)
    fraud_probability = Column(DECIMAL(8, 6), nullable=False)
    confidence_score = Column(DECIMAL(8, 6))
    decision = Column(String(20))  # 'conforme', 'zone_grise', 'fraude'
    
    # Métadonnées
    ml_integration_used = Column(Boolean, default=False)
    decision_source = Column(String(50))  # 'ml', 'rl', 'hybrid', 'expert'
    context_features = Column(JSON)
    risk_analysis = Column(JSON)
    
    # Seuils utilisés
    ml_threshold = Column(DECIMAL(8, 6))
    rl_threshold = Column(DECIMAL(8, 6))
    decision_threshold = Column(DECIMAL(8, 6))
    
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    declaration = relationship("Declaration", back_populates="predictions")
    model = relationship("Model", back_populates="predictions")
    chapter = relationship("Chapter", back_populates="predictions")

class DeclarationFeature(Base):
    """Table des features extraites pour chaque déclaration"""
    __tablename__ = 'declaration_features'
    
    feature_id = Column(UUID(as_uuid=True), ForeignKey('features.feature_id'), primary_key=True)
    declaration_id = Column(String(100), ForeignKey('declarations.declaration_id'), primary_key=True)
    feature_value = Column(Text, nullable=False)
    feature_numeric_value = Column(DECIMAL(20, 6))
    is_activated = Column(Boolean, default=False)  # Pour les features business
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    feature = relationship("Feature", back_populates="declaration_features")
    declaration = relationship("Declaration", back_populates="declaration_features")

class RLDecision(Base):
    """Table des décisions RL"""
    __tablename__ = 'rl_decisions'
    
    decision_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    declaration_id = Column(String(100), ForeignKey('declarations.declaration_id'), nullable=False)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    
    # Contexte
    context_key = Column(String(255))
    context_json = Column(JSON)
    
    # Décision RL
    action = Column(String(50), nullable=False)
    model_probability = Column(DECIMAL(8, 6))
    rl_probability = Column(DECIMAL(8, 6), nullable=False)
    exploration_used = Column(Boolean, default=False)
    decision_source = Column(String(50))
    
    # Métriques
    confidence_score = Column(DECIMAL(8, 6), default=0.5)
    context_complexity = Column(Integer, default=1)
    seasonal_factor = Column(DECIMAL(8, 6), default=1.0)
    bureau_risk_score = Column(DECIMAL(8, 6), default=0.5)
    
    # Métadonnées
    extra_json = Column(JSON)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    declaration = relationship("Declaration", back_populates="rl_decisions")
    chapter = relationship("Chapter", back_populates="rl_decisions")

class InspectorProfile(Base):
    """Table des profils d'inspecteurs"""
    __tablename__ = 'inspector_profiles'
    
    inspector_id = Column(String(100), primary_key=True)
    name = Column(String(255))
    expertise_level = Column(String(20), default='standard')  # 'junior', 'standard', 'senior', 'expert'
    specialization = Column(JSON)  # Array des chapitres spécialisés
    total_reviews = Column(Integer, default=0)
    accuracy_rate = Column(DECIMAL(5, 4), default=0.0)
    avg_confidence = Column(DECIMAL(5, 4), default=0.5)
    avg_review_time = Column(DECIMAL(8, 2), default=300.0)
    performance_trend = Column(String(20), default='stable')
    created_at = Column(DateTime, default=func.current_timestamp())
    last_active = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    feedback_history = relationship("FeedbackHistory", back_populates="inspector")

class FeedbackHistory(Base):
    """Table des feedbacks"""
    __tablename__ = 'feedback_history'
    
    feedback_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    declaration_id = Column(String(100), ForeignKey('declarations.declaration_id'), nullable=False)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    inspector_id = Column(String(100), ForeignKey('inspector_profiles.inspector_id'), nullable=False)
    
    # Décisions
    inspector_decision = Column(Boolean, nullable=False)
    inspector_confidence = Column(DECIMAL(5, 4), nullable=False)
    predicted_fraud = Column(Boolean, nullable=False)
    predicted_probability = Column(DECIMAL(8, 6), nullable=False)
    predicted_action = Column(String(50))
    
    # Contexte
    context_key = Column(String(255))
    context_json = Column(JSON)
    exploration_used = Column(Boolean)
    notes = Column(Text)
    
    # Métriques de qualité
    feedback_quality_score = Column(DECIMAL(5, 4), default=0.8)
    inspector_expertise_level = Column(String(20), default='standard')
    review_time_seconds = Column(Integer)
    feedback_category = Column(String(20), default='regular')  # 'regular', 'urgent', 'review', 'audit'
    similar_cases_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    declaration = relationship("Declaration", back_populates="feedback_history")
    chapter = relationship("Chapter", back_populates="feedback_history")
    inspector = relationship("InspectorProfile", back_populates="feedback_history")

class AnalysisResult(Base):
    """Table des résultats d'analyse détaillée"""
    __tablename__ = 'analysis_results'
    
    analysis_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    declaration_id = Column(String(100), ForeignKey('declarations.declaration_id'), nullable=False)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    
    # Analyse de risque
    risk_score = Column(DECIMAL(8, 6))
    risk_level = Column(String(20))  # 'low', 'medium', 'high', 'critical'
    
    # Features business
    activated_business_features = Column(JSON)
    business_analysis = Column(JSON)
    
    # Incohérences
    inconsistencies = Column(JSON)
    inconsistency_count = Column(Integer, default=0)
    
    # Métadonnées
    analysis_type = Column(String(50), default='standard')
    analysis_confidence = Column(DECIMAL(8, 6))
    processing_time_ms = Column(Integer)
    
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    declaration = relationship("Declaration", back_populates="analysis_results")
    chapter = relationship("Chapter", back_populates="analysis_results")

class ModelThreshold(Base):
    """Table des seuils et configurations"""
    __tablename__ = 'model_thresholds'
    
    threshold_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.model_id'), nullable=False)
    
    # Seuils de décision
    conforme_threshold = Column(DECIMAL(8, 6))
    fraude_threshold = Column(DECIMAL(8, 6))
    optimal_threshold = Column(DECIMAL(8, 6))
    ml_threshold = Column(DECIMAL(8, 6))
    decision_threshold = Column(DECIMAL(8, 6))
    rl_threshold = Column(DECIMAL(8, 6))
    
    # Qualité de calibration
    calibration_quality = Column(String(20))  # 'EXCELLENT', 'GOOD', 'FAIR', 'POOR'
    brier_score = Column(DECIMAL(8, 6))
    ece = Column(DECIMAL(8, 6))  # Expected Calibration Error
    bss = Column(DECIMAL(8, 6))  # Brier Skill Score
    
    # Performance
    performance_metrics = Column(JSON)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.current_timestamp())
    updated_at = Column(DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp())
    
    # Relations
    chapter = relationship("Chapter", back_populates="model_thresholds")
    model = relationship("Model", back_populates="model_thresholds")

class PerformanceMetric(Base):
    """Table des métriques de performance"""
    __tablename__ = 'performance_metrics'
    
    metric_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'), nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey('models.model_id'), nullable=False)
    metric_type = Column(String(50), nullable=False)  # 'accuracy', 'precision', 'recall', 'f1_score', 'auc'
    metric_value = Column(DECIMAL(10, 6), nullable=False)
    metric_period = Column(String(20))  # 'daily', 'weekly', 'monthly'
    calculation_date = Column(DateTime)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    chapter = relationship("Chapter", back_populates="performance_metrics")
    model = relationship("Model", back_populates="performance_metrics")

class SystemLog(Base):
    """Table des logs système"""
    __tablename__ = 'system_logs'
    
    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    log_level = Column(String(20), nullable=False)  # 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    component = Column(String(100))  # 'ocr_pipeline', 'ml_model', 'rl_system', 'api'
    chapter_id = Column(String(10), ForeignKey('chapters.chapter_id'))
    message = Column(Text, nullable=False)
    details = Column(JSON)
    created_at = Column(DateTime, default=func.current_timestamp())
    
    # Relations
    chapter = relationship("Chapter", back_populates="system_logs")

# Index pour optimiser les performances
Index('idx_declarations_chapter_id', Declaration.chapter_id)
Index('idx_declarations_created_at', Declaration.created_at)
Index('idx_declarations_code_sh', Declaration.code_sh_complet)
Index('idx_declarations_pays_origine', Declaration.code_pays_origine)

Index('idx_predictions_declaration_id', Prediction.declaration_id)
Index('idx_predictions_chapter_id', Prediction.chapter_id)
Index('idx_predictions_created_at', Prediction.created_at)
Index('idx_predictions_fraud_probability', Prediction.fraud_probability)

Index('idx_rl_decisions_declaration_id', RLDecision.declaration_id)
Index('idx_rl_decisions_chapter_id', RLDecision.chapter_id)
Index('idx_rl_decisions_created_at', RLDecision.created_at)

Index('idx_feedback_history_declaration_id', FeedbackHistory.declaration_id)
Index('idx_feedback_history_inspector_id', FeedbackHistory.inspector_id)
Index('idx_feedback_history_created_at', FeedbackHistory.created_at)

Index('idx_analysis_results_declaration_id', AnalysisResult.declaration_id)
Index('idx_analysis_results_chapter_id', AnalysisResult.chapter_id)

Index('idx_declaration_features_declaration_id', DeclarationFeature.declaration_id)
Index('idx_declaration_features_feature_id', DeclarationFeature.feature_id)
Index('idx_declaration_features_activated', DeclarationFeature.is_activated)
