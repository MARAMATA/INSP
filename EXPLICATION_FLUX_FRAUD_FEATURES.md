# 🔍 EXPLICATION: QUI UTILISE LES FRAUD FEATURES ?

## 📊 FLUX COMPLET

### 1️⃣ PENDANT L'ENTRAÎNEMENT (une fois)

```
DONNÉES BRUTES (CSV)
    ↓
preprocess.py
    ↓
AdvancedFraudDetection.run_complete_analysis()
    ├── Calcule BIENAYME_CHEBYCHEV_SCORE pour TOUTES les déclarations
    ├── Calcule MIRROR_TEI_SCORE pour TOUTES les déclarations
    ├── Calcule ADMIN_VALUES_SCORE pour TOUTES les déclarations
    └── Sauvegarde fraud_detection_stats.json
    ↓
CHAP30_PROCESSED_ADVANCED.csv
    (contient TOUTES les fraud features avec valeurs NON-ZÉRO)
    ↓
ml_model_advanced.py
    ├── Charge CHAP30_PROCESSED_ADVANCED.csv
    ├── Entraîne les modèles ML avec TOUTES les features (y compris fraud features)
    └── Sauvegarde catboost_model.pkl
```

### 2️⃣ PENDANT LA PRÉDICTION (temps réel)

```
NOUVELLE DÉCLARATION (CSV/PDF/Image)
    ↓
OCR_INGEST.process_declaration_file()
    ├── Extrait les données brutes
    ├── Appelle create_advanced_context_from_ocr_data()
    │   └── Appelle _create_advanced_fraud_scores()
    │       ├── Charge fraud_detection_stats.json
    │       ├── Calcule BIENAYME_CHEBYCHEV_SCORE
    │       ├── Calcule MIRROR_TEI_SCORE
    │       ├── Calcule ADMIN_VALUES_SCORE
    │       └── Retourne les scores
    ↓
CONTEXTE AVANCÉ (avec fraud features NON-ZÉRO)
    ↓
OCR_PIPELINE.predict_fraud()
    ├── Reçoit le contexte avec fraud features
    ├── Crée un DataFrame avec TOUTES les features
    └── Appelle le modèle ML (catboost_model.pkl)
        └── Le modèle utilise les fraud features pour prédire
    ↓
PRÉDICTION FINALE
```

## 🔑 QUI FAIT QUOI ?

### OCR_INGEST
**RÔLE**: Préparer les données et calculer les fraud features
- ✅ `process_declaration_file()` : Point d'entrée
- ✅ `create_advanced_context_from_ocr_data()` : Crée le contexte
- ✅ `_create_advanced_fraud_scores()` : **CALCULE LES FRAUD FEATURES** en utilisant fraud_detection_stats.json

### OCR_PIPELINE  
**RÔLE**: Utiliser le contexte pour faire la prédiction ML
- ✅ `predict_fraud()` : Reçoit le contexte AVEC fraud features déjà calculées
- ✅ Crée un DataFrame avec toutes les features
- ✅ Appelle le modèle ML qui utilise ces features

## 📝 EXEMPLE CONCRET

### Nouvelle déclaration arrive: 30049000_FR, VALEUR_CAF=1,000,000

1. **OCR_INGEST** calcule:
   ```python
   # Charge fraud_detection_stats.json
   stats = {...}  # 152 couples pour chap30
   
   # Récupère stats pour 30049000_FR
   po_stats = stats['product_origin_stats']['30049000_FR']
   # mean = 950000, std = 480000
   
   # Calcule le score
   BIENAYME_CHEBYCHEV_SCORE = |1000000 - 950000| / 480000 = 0.104
   MIRROR_TEI_SCORE = |15.0 - 14.5| / 6.0 = 0.083
   ADMIN_VALUES_SCORE = |1000000 - 850000| / 750000 = 0.200
   ```

2. **OCR_PIPELINE** utilise:
   ```python
   context = {
       'VALEUR_CAF': 1000000,
       'BIENAYME_CHEBYCHEV_SCORE': 0.104,  ← Déjà calculé !
       'MIRROR_TEI_SCORE': 0.083,           ← Déjà calculé !
       'ADMIN_VALUES_SCORE': 0.200,         ← Déjà calculé !
       ...
   }
   
   # Crée DataFrame et appelle le modèle ML
   prediction = model.predict_proba(context)
   ```

## 🎯 RÉPONSE À VOTRE QUESTION

**OCR_INGEST** calcule les fraud features
**OCR_PIPELINE** les utilise pour la prédiction

Les deux travaillent ensemble :
- OCR_INGEST = Préparation des données
- OCR_PIPELINE = Utilisation des données pour prédire

---
**Date**: 2025-01-09
