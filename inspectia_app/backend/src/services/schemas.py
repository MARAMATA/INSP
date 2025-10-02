# -*- coding: utf-8 -*-
"""
Schémas Pydantic de l'API.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class AnalyzeResponse(BaseModel):
    chapter: str
    rows_articles: int
    rows_declarations: int
    alerts_articles: int
    outputs: Dict[str, str]


class ArticleIn(BaseModel):
    article_key: Optional[str] = None
    hs_code: Optional[str] = None
    libelle: Optional[str] = None
    qte: Optional[float] = None
    unite: Optional[str] = None
    poids_net: Optional[float] = None
    poids_brut: Optional[float] = None
    valeur_ligne: Optional[float] = None
    pays_origine: Optional[str] = None
    marque: Optional[str] = None
    modele: Optional[str] = None


class DeclarationIn(BaseModel):
    decl_key: Optional[str] = None
    annee: Optional[int] = None
    bureau: Optional[str] = None
    numero: Optional[int] = None
    importateur: Optional[str] = None
    fournisseur: Optional[str] = None
    pays_origine: Optional[str] = None
    pays_provenance: Optional[str] = None
    incoterm: Optional[str] = None
    mode_paiement: Optional[str] = None
    regime: Optional[str] = None
    monnaie: Optional[str] = None
    valeur_decl: Optional[float] = None
    date_enreg: Optional[str] = None
    canal: Optional[str] = None
    nbre_colis: Optional[int] = None
    facture_total: Optional[float] = None
    articles: List[ArticleIn] = Field(default_factory=list)


class PredictRequest(BaseModel):
    chapter: str
    declaration: DeclarationIn


class AlertOut(BaseModel):
    code: str
    severity: str
    message: str
    legal_ref: Optional[str] = None
    field_refs: Optional[List[str]] = None
    confidence: Optional[float] = None
    category: Optional[str] = None
    risk_score: Optional[float] = None
    article_key: Optional[str] = None


class MLPredictionOut(BaseModel):
    prediction: int
    probability: float
    confidence: str
    features_used: List[str]
    feature_values: Optional[Dict[str, float]] = None
    error: Optional[str] = None


class ModelInfoOut(BaseModel):
    available: bool
    chapter: Optional[str] = None
    model_type: Optional[str] = None
    features_count: Optional[int] = None
    features: Optional[List[str]] = None
    metrics: Optional[Dict] = None
    error: Optional[str] = None


class PredictResponse(BaseModel):
    decl_key: str
    chapter: str
    risk_score: float
    risk_class: str
    alerts: List[AlertOut] = Field(default_factory=list)
    pv_path: Optional[str] = None
    ml_prediction: Optional[MLPredictionOut] = None


class UploadResponse(BaseModel):
    document_id: str
    path: str
