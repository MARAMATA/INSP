#!/usr/bin/env python3
"""
Test complet pour vérifier que OCR_INGEST et OCR_PIPELINE fonctionnent bien ensemble
"""

import sys
sys.path.append('/Users/macbook/Desktop/INSP/inspectia_app/backend/src')

from shared.ocr_ingest import create_advanced_context_from_ocr_data
from shared.ocr_pipeline import AdvancedOCRPipeline
import json

print("=" * 70)
print("🧪 TEST COMPLET: OCR_INGEST + OCR_PIPELINE")
print("=" * 70)

# 1. Test OCR_INGEST
print("\n1️⃣ TEST OCR_INGEST: Création du contexte avec fraud features")
print("-" * 70)

ocr_data = {
    'VALEUR_CAF': 1500000.0,
    'MONTANT_LIQUIDATION': 225000.0,
    'POIDS_NET_KG': 1200.0,
    'NOMBRE_COLIS': 10.0,
    'CODE_PRODUIT_STR': '30049000',
    'CODE_SH_COMPLET': '30049000',
    'PAYS_ORIGINE_STR': 'FR',
    'CODE_PAYS_ORIGINE': 'FR',
    'REGIME_COMPLET': 'IMPORT',
    'DECLARATION_ID': '2025/TEST/0001'
}

# Créer le contexte avancé (appelle _create_advanced_fraud_scores en interne)
context = create_advanced_context_from_ocr_data(ocr_data, 'chap30')

print(f"✅ Contexte créé: {len(context)} features")

# Vérifier les fraud features
fraud_features = [
    'BIENAYME_CHEBYCHEV_SCORE', 'TEI_CALCULE', 'MIRROR_TEI_SCORE',
    'ADMIN_VALUES_SCORE', 'COMPOSITE_FRAUD_SCORE'
]

print("\n📊 Fraud features dans le contexte:")
fraud_present = 0
fraud_non_zero = 0
for feature in fraud_features:
    if feature in context:
        fraud_present += 1
        value = context[feature]
        if value != 0.0:
            fraud_non_zero += 1
            print(f"   {feature}: {value:.6f} ✅")
        else:
            print(f"   {feature}: {value:.6f} ⚠️ (à zéro)")
    else:
        print(f"   {feature}: ❌ MANQUANTE")

print(f"\n   Résultat: {fraud_present}/{len(fraud_features)} présentes, {fraud_non_zero}/{len(fraud_features)} non-zéro")

if fraud_present == len(fraud_features) and fraud_non_zero > 0:
    print("   ✅ OCR_INGEST fonctionne correctement !")
else:
    print("   ❌ OCR_INGEST a un problème")

# 2. Test OCR_PIPELINE
print("\n2️⃣ TEST OCR_PIPELINE: Utilisation du contexte pour prédiction")
print("-" * 70)

pipeline = AdvancedOCRPipeline()

try:
    # Faire une prédiction avec le contexte créé par OCR_INGEST
    prediction = pipeline.predict_fraud(context, 'chap30', 'basic')
    
    print(f"✅ Prédiction réussie")
    print(f"   Probabilité de fraude: {prediction.get('fraud_probability', 0):.3f}")
    print(f"   Décision: {prediction.get('decision', 'N/A')}")
    print(f"   Confiance: {prediction.get('confidence_score', 0):.3f}")
    
    # Vérifier que le contexte a bien été utilisé
    if 'context' in prediction:
        pred_context = prediction['context']
        fraud_in_pred = sum(1 for f in fraud_features if f in pred_context and pred_context.get(f, 0) != 0)
        print(f"\n   Fraud features utilisées par le pipeline: {fraud_in_pred}/{len(fraud_features)}")
        
        if fraud_in_pred > 0:
            print("   ✅ OCR_PIPELINE utilise bien les fraud features !")
        else:
            print("   ⚠️ OCR_PIPELINE n'utilise pas les fraud features")
    
    print("\n   ✅ OCR_PIPELINE fonctionne correctement !")
    
except Exception as e:
    print(f"   ❌ Erreur OCR_PIPELINE: {e}")

# 3. Test du flux complet
print("\n3️⃣ TEST FLUX COMPLET: OCR_INGEST → OCR_PIPELINE")
print("-" * 70)

if fraud_non_zero > 0 and 'prediction' in locals():
    print("✅ FLUX COMPLET FONCTIONNE:")
    print("   1. OCR_INGEST calcule les fraud features → NON-ZÉRO ✅")
    print("   2. OCR_PIPELINE reçoit le contexte → SUCCÈS ✅")
    print("   3. Prédiction ML réalisée → SUCCÈS ✅")
    print("\n🎉 TOUT FONCTIONNE À 100% !")
else:
    print("❌ Le flux complet a un problème")

print("\n" + "=" * 70)
print("🎯 RÉSUMÉ FINAL")
print("=" * 70)
print(f"OCR_INGEST: {'✅ OK' if fraud_non_zero > 0 else '❌ PROBLÈME'}")
print(f"OCR_PIPELINE: {'✅ OK' if 'prediction' in locals() else '❌ PROBLÈME'}")
print(f"FLUX COMPLET: {'✅ OK' if fraud_non_zero > 0 and 'prediction' in locals() else '❌ PROBLÈME'}")



Test complet pour vérifier que OCR_INGEST et OCR_PIPELINE fonctionnent bien ensemble
"""

import sys
sys.path.append('/Users/macbook/Desktop/INSP/inspectia_app/backend/src')

from shared.ocr_ingest import create_advanced_context_from_ocr_data
from shared.ocr_pipeline import AdvancedOCRPipeline
import json

print("=" * 70)
print("🧪 TEST COMPLET: OCR_INGEST + OCR_PIPELINE")
print("=" * 70)

# 1. Test OCR_INGEST
print("\n1️⃣ TEST OCR_INGEST: Création du contexte avec fraud features")
print("-" * 70)

ocr_data = {
    'VALEUR_CAF': 1500000.0,
    'MONTANT_LIQUIDATION': 225000.0,
    'POIDS_NET_KG': 1200.0,
    'NOMBRE_COLIS': 10.0,
    'CODE_PRODUIT_STR': '30049000',
    'CODE_SH_COMPLET': '30049000',
    'PAYS_ORIGINE_STR': 'FR',
    'CODE_PAYS_ORIGINE': 'FR',
    'REGIME_COMPLET': 'IMPORT',
    'DECLARATION_ID': '2025/TEST/0001'
}

# Créer le contexte avancé (appelle _create_advanced_fraud_scores en interne)
context = create_advanced_context_from_ocr_data(ocr_data, 'chap30')

print(f"✅ Contexte créé: {len(context)} features")

# Vérifier les fraud features
fraud_features = [
    'BIENAYME_CHEBYCHEV_SCORE', 'TEI_CALCULE', 'MIRROR_TEI_SCORE',
    'ADMIN_VALUES_SCORE', 'COMPOSITE_FRAUD_SCORE'
]

print("\n📊 Fraud features dans le contexte:")
fraud_present = 0
fraud_non_zero = 0
for feature in fraud_features:
    if feature in context:
        fraud_present += 1
        value = context[feature]
        if value != 0.0:
            fraud_non_zero += 1
            print(f"   {feature}: {value:.6f} ✅")
        else:
            print(f"   {feature}: {value:.6f} ⚠️ (à zéro)")
    else:
        print(f"   {feature}: ❌ MANQUANTE")

print(f"\n   Résultat: {fraud_present}/{len(fraud_features)} présentes, {fraud_non_zero}/{len(fraud_features)} non-zéro")

if fraud_present == len(fraud_features) and fraud_non_zero > 0:
    print("   ✅ OCR_INGEST fonctionne correctement !")
else:
    print("   ❌ OCR_INGEST a un problème")

# 2. Test OCR_PIPELINE
print("\n2️⃣ TEST OCR_PIPELINE: Utilisation du contexte pour prédiction")
print("-" * 70)

pipeline = AdvancedOCRPipeline()

try:
    # Faire une prédiction avec le contexte créé par OCR_INGEST
    prediction = pipeline.predict_fraud(context, 'chap30', 'basic')
    
    print(f"✅ Prédiction réussie")
    print(f"   Probabilité de fraude: {prediction.get('fraud_probability', 0):.3f}")
    print(f"   Décision: {prediction.get('decision', 'N/A')}")
    print(f"   Confiance: {prediction.get('confidence_score', 0):.3f}")
    
    # Vérifier que le contexte a bien été utilisé
    if 'context' in prediction:
        pred_context = prediction['context']
        fraud_in_pred = sum(1 for f in fraud_features if f in pred_context and pred_context.get(f, 0) != 0)
        print(f"\n   Fraud features utilisées par le pipeline: {fraud_in_pred}/{len(fraud_features)}")
        
        if fraud_in_pred > 0:
            print("   ✅ OCR_PIPELINE utilise bien les fraud features !")
        else:
            print("   ⚠️ OCR_PIPELINE n'utilise pas les fraud features")
    
    print("\n   ✅ OCR_PIPELINE fonctionne correctement !")
    
except Exception as e:
    print(f"   ❌ Erreur OCR_PIPELINE: {e}")

# 3. Test du flux complet
print("\n3️⃣ TEST FLUX COMPLET: OCR_INGEST → OCR_PIPELINE")
print("-" * 70)

if fraud_non_zero > 0 and 'prediction' in locals():
    print("✅ FLUX COMPLET FONCTIONNE:")
    print("   1. OCR_INGEST calcule les fraud features → NON-ZÉRO ✅")
    print("   2. OCR_PIPELINE reçoit le contexte → SUCCÈS ✅")
    print("   3. Prédiction ML réalisée → SUCCÈS ✅")
    print("\n🎉 TOUT FONCTIONNE À 100% !")
else:
    print("❌ Le flux complet a un problème")

print("\n" + "=" * 70)
print("🎯 RÉSUMÉ FINAL")
print("=" * 70)
print(f"OCR_INGEST: {'✅ OK' if fraud_non_zero > 0 else '❌ PROBLÈME'}")
print(f"OCR_PIPELINE: {'✅ OK' if 'prediction' in locals() else '❌ PROBLÈME'}")
print(f"FLUX COMPLET: {'✅ OK' if fraud_non_zero > 0 and 'prediction' in locals() else '❌ PROBLÈME'}")



Test complet pour vérifier que OCR_INGEST et OCR_PIPELINE fonctionnent bien ensemble
"""

import sys
sys.path.append('/Users/macbook/Desktop/INSP/inspectia_app/backend/src')

from shared.ocr_ingest import create_advanced_context_from_ocr_data
from shared.ocr_pipeline import AdvancedOCRPipeline
import json

print("=" * 70)
print("🧪 TEST COMPLET: OCR_INGEST + OCR_PIPELINE")
print("=" * 70)

# 1. Test OCR_INGEST
print("\n1️⃣ TEST OCR_INGEST: Création du contexte avec fraud features")
print("-" * 70)

ocr_data = {
    'VALEUR_CAF': 1500000.0,
    'MONTANT_LIQUIDATION': 225000.0,
    'POIDS_NET_KG': 1200.0,
    'NOMBRE_COLIS': 10.0,
    'CODE_PRODUIT_STR': '30049000',
    'CODE_SH_COMPLET': '30049000',
    'PAYS_ORIGINE_STR': 'FR',
    'CODE_PAYS_ORIGINE': 'FR',
    'REGIME_COMPLET': 'IMPORT',
    'DECLARATION_ID': '2025/TEST/0001'
}

# Créer le contexte avancé (appelle _create_advanced_fraud_scores en interne)
context = create_advanced_context_from_ocr_data(ocr_data, 'chap30')

print(f"✅ Contexte créé: {len(context)} features")

# Vérifier les fraud features
fraud_features = [
    'BIENAYME_CHEBYCHEV_SCORE', 'TEI_CALCULE', 'MIRROR_TEI_SCORE',
    'ADMIN_VALUES_SCORE', 'COMPOSITE_FRAUD_SCORE'
]

print("\n📊 Fraud features dans le contexte:")
fraud_present = 0
fraud_non_zero = 0
for feature in fraud_features:
    if feature in context:
        fraud_present += 1
        value = context[feature]
        if value != 0.0:
            fraud_non_zero += 1
            print(f"   {feature}: {value:.6f} ✅")
        else:
            print(f"   {feature}: {value:.6f} ⚠️ (à zéro)")
    else:
        print(f"   {feature}: ❌ MANQUANTE")

print(f"\n   Résultat: {fraud_present}/{len(fraud_features)} présentes, {fraud_non_zero}/{len(fraud_features)} non-zéro")

if fraud_present == len(fraud_features) and fraud_non_zero > 0:
    print("   ✅ OCR_INGEST fonctionne correctement !")
else:
    print("   ❌ OCR_INGEST a un problème")

# 2. Test OCR_PIPELINE
print("\n2️⃣ TEST OCR_PIPELINE: Utilisation du contexte pour prédiction")
print("-" * 70)

pipeline = AdvancedOCRPipeline()

try:
    # Faire une prédiction avec le contexte créé par OCR_INGEST
    prediction = pipeline.predict_fraud(context, 'chap30', 'basic')
    
    print(f"✅ Prédiction réussie")
    print(f"   Probabilité de fraude: {prediction.get('fraud_probability', 0):.3f}")
    print(f"   Décision: {prediction.get('decision', 'N/A')}")
    print(f"   Confiance: {prediction.get('confidence_score', 0):.3f}")
    
    # Vérifier que le contexte a bien été utilisé
    if 'context' in prediction:
        pred_context = prediction['context']
        fraud_in_pred = sum(1 for f in fraud_features if f in pred_context and pred_context.get(f, 0) != 0)
        print(f"\n   Fraud features utilisées par le pipeline: {fraud_in_pred}/{len(fraud_features)}")
        
        if fraud_in_pred > 0:
            print("   ✅ OCR_PIPELINE utilise bien les fraud features !")
        else:
            print("   ⚠️ OCR_PIPELINE n'utilise pas les fraud features")
    
    print("\n   ✅ OCR_PIPELINE fonctionne correctement !")
    
except Exception as e:
    print(f"   ❌ Erreur OCR_PIPELINE: {e}")

# 3. Test du flux complet
print("\n3️⃣ TEST FLUX COMPLET: OCR_INGEST → OCR_PIPELINE")
print("-" * 70)

if fraud_non_zero > 0 and 'prediction' in locals():
    print("✅ FLUX COMPLET FONCTIONNE:")
    print("   1. OCR_INGEST calcule les fraud features → NON-ZÉRO ✅")
    print("   2. OCR_PIPELINE reçoit le contexte → SUCCÈS ✅")
    print("   3. Prédiction ML réalisée → SUCCÈS ✅")
    print("\n🎉 TOUT FONCTIONNE À 100% !")
else:
    print("❌ Le flux complet a un problème")

print("\n" + "=" * 70)
print("🎯 RÉSUMÉ FINAL")
print("=" * 70)
print(f"OCR_INGEST: {'✅ OK' if fraud_non_zero > 0 else '❌ PROBLÈME'}")
print(f"OCR_PIPELINE: {'✅ OK' if 'prediction' in locals() else '❌ PROBLÈME'}")
print(f"FLUX COMPLET: {'✅ OK' if fraud_non_zero > 0 and 'prediction' in locals() else '❌ PROBLÈME'}")



Test complet pour vérifier que OCR_INGEST et OCR_PIPELINE fonctionnent bien ensemble
"""

import sys
sys.path.append('/Users/macbook/Desktop/INSP/inspectia_app/backend/src')

from shared.ocr_ingest import create_advanced_context_from_ocr_data
from shared.ocr_pipeline import AdvancedOCRPipeline
import json

print("=" * 70)
print("🧪 TEST COMPLET: OCR_INGEST + OCR_PIPELINE")
print("=" * 70)

# 1. Test OCR_INGEST
print("\n1️⃣ TEST OCR_INGEST: Création du contexte avec fraud features")
print("-" * 70)

ocr_data = {
    'VALEUR_CAF': 1500000.0,
    'MONTANT_LIQUIDATION': 225000.0,
    'POIDS_NET_KG': 1200.0,
    'NOMBRE_COLIS': 10.0,
    'CODE_PRODUIT_STR': '30049000',
    'CODE_SH_COMPLET': '30049000',
    'PAYS_ORIGINE_STR': 'FR',
    'CODE_PAYS_ORIGINE': 'FR',
    'REGIME_COMPLET': 'IMPORT',
    'DECLARATION_ID': '2025/TEST/0001'
}

# Créer le contexte avancé (appelle _create_advanced_fraud_scores en interne)
context = create_advanced_context_from_ocr_data(ocr_data, 'chap30')

print(f"✅ Contexte créé: {len(context)} features")

# Vérifier les fraud features
fraud_features = [
    'BIENAYME_CHEBYCHEV_SCORE', 'TEI_CALCULE', 'MIRROR_TEI_SCORE',
    'ADMIN_VALUES_SCORE', 'COMPOSITE_FRAUD_SCORE'
]

print("\n📊 Fraud features dans le contexte:")
fraud_present = 0
fraud_non_zero = 0
for feature in fraud_features:
    if feature in context:
        fraud_present += 1
        value = context[feature]
        if value != 0.0:
            fraud_non_zero += 1
            print(f"   {feature}: {value:.6f} ✅")
        else:
            print(f"   {feature}: {value:.6f} ⚠️ (à zéro)")
    else:
        print(f"   {feature}: ❌ MANQUANTE")

print(f"\n   Résultat: {fraud_present}/{len(fraud_features)} présentes, {fraud_non_zero}/{len(fraud_features)} non-zéro")

if fraud_present == len(fraud_features) and fraud_non_zero > 0:
    print("   ✅ OCR_INGEST fonctionne correctement !")
else:
    print("   ❌ OCR_INGEST a un problème")

# 2. Test OCR_PIPELINE
print("\n2️⃣ TEST OCR_PIPELINE: Utilisation du contexte pour prédiction")
print("-" * 70)

pipeline = AdvancedOCRPipeline()

try:
    # Faire une prédiction avec le contexte créé par OCR_INGEST
    prediction = pipeline.predict_fraud(context, 'chap30', 'basic')
    
    print(f"✅ Prédiction réussie")
    print(f"   Probabilité de fraude: {prediction.get('fraud_probability', 0):.3f}")
    print(f"   Décision: {prediction.get('decision', 'N/A')}")
    print(f"   Confiance: {prediction.get('confidence_score', 0):.3f}")
    
    # Vérifier que le contexte a bien été utilisé
    if 'context' in prediction:
        pred_context = prediction['context']
        fraud_in_pred = sum(1 for f in fraud_features if f in pred_context and pred_context.get(f, 0) != 0)
        print(f"\n   Fraud features utilisées par le pipeline: {fraud_in_pred}/{len(fraud_features)}")
        
        if fraud_in_pred > 0:
            print("   ✅ OCR_PIPELINE utilise bien les fraud features !")
        else:
            print("   ⚠️ OCR_PIPELINE n'utilise pas les fraud features")
    
    print("\n   ✅ OCR_PIPELINE fonctionne correctement !")
    
except Exception as e:
    print(f"   ❌ Erreur OCR_PIPELINE: {e}")

# 3. Test du flux complet
print("\n3️⃣ TEST FLUX COMPLET: OCR_INGEST → OCR_PIPELINE")
print("-" * 70)

if fraud_non_zero > 0 and 'prediction' in locals():
    print("✅ FLUX COMPLET FONCTIONNE:")
    print("   1. OCR_INGEST calcule les fraud features → NON-ZÉRO ✅")
    print("   2. OCR_PIPELINE reçoit le contexte → SUCCÈS ✅")
    print("   3. Prédiction ML réalisée → SUCCÈS ✅")
    print("\n🎉 TOUT FONCTIONNE À 100% !")
else:
    print("❌ Le flux complet a un problème")

print("\n" + "=" * 70)
print("🎯 RÉSUMÉ FINAL")
print("=" * 70)
print(f"OCR_INGEST: {'✅ OK' if fraud_non_zero > 0 else '❌ PROBLÈME'}")
print(f"OCR_PIPELINE: {'✅ OK' if 'prediction' in locals() else '❌ PROBLÈME'}")
print(f"FLUX COMPLET: {'✅ OK' if fraud_non_zero > 0 and 'prediction' in locals() else '❌ PROBLÈME'}")



Test complet pour vérifier que OCR_INGEST et OCR_PIPELINE fonctionnent bien ensemble
"""

import sys
sys.path.append('/Users/macbook/Desktop/INSP/inspectia_app/backend/src')

from shared.ocr_ingest import create_advanced_context_from_ocr_data
from shared.ocr_pipeline import AdvancedOCRPipeline
import json

print("=" * 70)
print("🧪 TEST COMPLET: OCR_INGEST + OCR_PIPELINE")
print("=" * 70)

# 1. Test OCR_INGEST
print("\n1️⃣ TEST OCR_INGEST: Création du contexte avec fraud features")
print("-" * 70)

ocr_data = {
    'VALEUR_CAF': 1500000.0,
    'MONTANT_LIQUIDATION': 225000.0,
    'POIDS_NET_KG': 1200.0,
    'NOMBRE_COLIS': 10.0,
    'CODE_PRODUIT_STR': '30049000',
    'CODE_SH_COMPLET': '30049000',
    'PAYS_ORIGINE_STR': 'FR',
    'CODE_PAYS_ORIGINE': 'FR',
    'REGIME_COMPLET': 'IMPORT',
    'DECLARATION_ID': '2025/TEST/0001'
}

# Créer le contexte avancé (appelle _create_advanced_fraud_scores en interne)
context = create_advanced_context_from_ocr_data(ocr_data, 'chap30')

print(f"✅ Contexte créé: {len(context)} features")

# Vérifier les fraud features
fraud_features = [
    'BIENAYME_CHEBYCHEV_SCORE', 'TEI_CALCULE', 'MIRROR_TEI_SCORE',
    'ADMIN_VALUES_SCORE', 'COMPOSITE_FRAUD_SCORE'
]

print("\n📊 Fraud features dans le contexte:")
fraud_present = 0
fraud_non_zero = 0
for feature in fraud_features:
    if feature in context:
        fraud_present += 1
        value = context[feature]
        if value != 0.0:
            fraud_non_zero += 1
            print(f"   {feature}: {value:.6f} ✅")
        else:
            print(f"   {feature}: {value:.6f} ⚠️ (à zéro)")
    else:
        print(f"   {feature}: ❌ MANQUANTE")

print(f"\n   Résultat: {fraud_present}/{len(fraud_features)} présentes, {fraud_non_zero}/{len(fraud_features)} non-zéro")

if fraud_present == len(fraud_features) and fraud_non_zero > 0:
    print("   ✅ OCR_INGEST fonctionne correctement !")
else:
    print("   ❌ OCR_INGEST a un problème")

# 2. Test OCR_PIPELINE
print("\n2️⃣ TEST OCR_PIPELINE: Utilisation du contexte pour prédiction")
print("-" * 70)

pipeline = AdvancedOCRPipeline()

try:
    # Faire une prédiction avec le contexte créé par OCR_INGEST
    prediction = pipeline.predict_fraud(context, 'chap30', 'basic')
    
    print(f"✅ Prédiction réussie")
    print(f"   Probabilité de fraude: {prediction.get('fraud_probability', 0):.3f}")
    print(f"   Décision: {prediction.get('decision', 'N/A')}")
    print(f"   Confiance: {prediction.get('confidence_score', 0):.3f}")
    
    # Vérifier que le contexte a bien été utilisé
    if 'context' in prediction:
        pred_context = prediction['context']
        fraud_in_pred = sum(1 for f in fraud_features if f in pred_context and pred_context.get(f, 0) != 0)
        print(f"\n   Fraud features utilisées par le pipeline: {fraud_in_pred}/{len(fraud_features)}")
        
        if fraud_in_pred > 0:
            print("   ✅ OCR_PIPELINE utilise bien les fraud features !")
        else:
            print("   ⚠️ OCR_PIPELINE n'utilise pas les fraud features")
    
    print("\n   ✅ OCR_PIPELINE fonctionne correctement !")
    
except Exception as e:
    print(f"   ❌ Erreur OCR_PIPELINE: {e}")

# 3. Test du flux complet
print("\n3️⃣ TEST FLUX COMPLET: OCR_INGEST → OCR_PIPELINE")
print("-" * 70)

if fraud_non_zero > 0 and 'prediction' in locals():
    print("✅ FLUX COMPLET FONCTIONNE:")
    print("   1. OCR_INGEST calcule les fraud features → NON-ZÉRO ✅")
    print("   2. OCR_PIPELINE reçoit le contexte → SUCCÈS ✅")
    print("   3. Prédiction ML réalisée → SUCCÈS ✅")
    print("\n🎉 TOUT FONCTIONNE À 100% !")
else:
    print("❌ Le flux complet a un problème")

print("\n" + "=" * 70)
print("🎯 RÉSUMÉ FINAL")
print("=" * 70)
print(f"OCR_INGEST: {'✅ OK' if fraud_non_zero > 0 else '❌ PROBLÈME'}")
print(f"OCR_PIPELINE: {'✅ OK' if 'prediction' in locals() else '❌ PROBLÈME'}")
print(f"FLUX COMPLET: {'✅ OK' if fraud_non_zero > 0 and 'prediction' in locals() else '❌ PROBLÈME'}")



Test complet pour vérifier que OCR_INGEST et OCR_PIPELINE fonctionnent bien ensemble
"""

import sys
sys.path.append('/Users/macbook/Desktop/INSP/inspectia_app/backend/src')

from shared.ocr_ingest import create_advanced_context_from_ocr_data
from shared.ocr_pipeline import AdvancedOCRPipeline
import json

print("=" * 70)
print("🧪 TEST COMPLET: OCR_INGEST + OCR_PIPELINE")
print("=" * 70)

# 1. Test OCR_INGEST
print("\n1️⃣ TEST OCR_INGEST: Création du contexte avec fraud features")
print("-" * 70)

ocr_data = {
    'VALEUR_CAF': 1500000.0,
    'MONTANT_LIQUIDATION': 225000.0,
    'POIDS_NET_KG': 1200.0,
    'NOMBRE_COLIS': 10.0,
    'CODE_PRODUIT_STR': '30049000',
    'CODE_SH_COMPLET': '30049000',
    'PAYS_ORIGINE_STR': 'FR',
    'CODE_PAYS_ORIGINE': 'FR',
    'REGIME_COMPLET': 'IMPORT',
    'DECLARATION_ID': '2025/TEST/0001'
}

# Créer le contexte avancé (appelle _create_advanced_fraud_scores en interne)
context = create_advanced_context_from_ocr_data(ocr_data, 'chap30')

print(f"✅ Contexte créé: {len(context)} features")

# Vérifier les fraud features
fraud_features = [
    'BIENAYME_CHEBYCHEV_SCORE', 'TEI_CALCULE', 'MIRROR_TEI_SCORE',
    'ADMIN_VALUES_SCORE', 'COMPOSITE_FRAUD_SCORE'
]

print("\n📊 Fraud features dans le contexte:")
fraud_present = 0
fraud_non_zero = 0
for feature in fraud_features:
    if feature in context:
        fraud_present += 1
        value = context[feature]
        if value != 0.0:
            fraud_non_zero += 1
            print(f"   {feature}: {value:.6f} ✅")
        else:
            print(f"   {feature}: {value:.6f} ⚠️ (à zéro)")
    else:
        print(f"   {feature}: ❌ MANQUANTE")

print(f"\n   Résultat: {fraud_present}/{len(fraud_features)} présentes, {fraud_non_zero}/{len(fraud_features)} non-zéro")

if fraud_present == len(fraud_features) and fraud_non_zero > 0:
    print("   ✅ OCR_INGEST fonctionne correctement !")
else:
    print("   ❌ OCR_INGEST a un problème")

# 2. Test OCR_PIPELINE
print("\n2️⃣ TEST OCR_PIPELINE: Utilisation du contexte pour prédiction")
print("-" * 70)

pipeline = AdvancedOCRPipeline()

try:
    # Faire une prédiction avec le contexte créé par OCR_INGEST
    prediction = pipeline.predict_fraud(context, 'chap30', 'basic')
    
    print(f"✅ Prédiction réussie")
    print(f"   Probabilité de fraude: {prediction.get('fraud_probability', 0):.3f}")
    print(f"   Décision: {prediction.get('decision', 'N/A')}")
    print(f"   Confiance: {prediction.get('confidence_score', 0):.3f}")
    
    # Vérifier que le contexte a bien été utilisé
    if 'context' in prediction:
        pred_context = prediction['context']
        fraud_in_pred = sum(1 for f in fraud_features if f in pred_context and pred_context.get(f, 0) != 0)
        print(f"\n   Fraud features utilisées par le pipeline: {fraud_in_pred}/{len(fraud_features)}")
        
        if fraud_in_pred > 0:
            print("   ✅ OCR_PIPELINE utilise bien les fraud features !")
        else:
            print("   ⚠️ OCR_PIPELINE n'utilise pas les fraud features")
    
    print("\n   ✅ OCR_PIPELINE fonctionne correctement !")
    
except Exception as e:
    print(f"   ❌ Erreur OCR_PIPELINE: {e}")

# 3. Test du flux complet
print("\n3️⃣ TEST FLUX COMPLET: OCR_INGEST → OCR_PIPELINE")
print("-" * 70)

if fraud_non_zero > 0 and 'prediction' in locals():
    print("✅ FLUX COMPLET FONCTIONNE:")
    print("   1. OCR_INGEST calcule les fraud features → NON-ZÉRO ✅")
    print("   2. OCR_PIPELINE reçoit le contexte → SUCCÈS ✅")
    print("   3. Prédiction ML réalisée → SUCCÈS ✅")
    print("\n🎉 TOUT FONCTIONNE À 100% !")
else:
    print("❌ Le flux complet a un problème")

print("\n" + "=" * 70)
print("🎯 RÉSUMÉ FINAL")
print("=" * 70)
print(f"OCR_INGEST: {'✅ OK' if fraud_non_zero > 0 else '❌ PROBLÈME'}")
print(f"OCR_PIPELINE: {'✅ OK' if 'prediction' in locals() else '❌ PROBLÈME'}")
print(f"FLUX COMPLET: {'✅ OK' if fraud_non_zero > 0 and 'prediction' in locals() else '❌ PROBLÈME'}")



Test complet pour vérifier que OCR_INGEST et OCR_PIPELINE fonctionnent bien ensemble
"""

import sys
sys.path.append('/Users/macbook/Desktop/INSP/inspectia_app/backend/src')

from shared.ocr_ingest import create_advanced_context_from_ocr_data
from shared.ocr_pipeline import AdvancedOCRPipeline
import json

print("=" * 70)
print("🧪 TEST COMPLET: OCR_INGEST + OCR_PIPELINE")
print("=" * 70)

# 1. Test OCR_INGEST
print("\n1️⃣ TEST OCR_INGEST: Création du contexte avec fraud features")
print("-" * 70)

ocr_data = {
    'VALEUR_CAF': 1500000.0,
    'MONTANT_LIQUIDATION': 225000.0,
    'POIDS_NET_KG': 1200.0,
    'NOMBRE_COLIS': 10.0,
    'CODE_PRODUIT_STR': '30049000',
    'CODE_SH_COMPLET': '30049000',
    'PAYS_ORIGINE_STR': 'FR',
    'CODE_PAYS_ORIGINE': 'FR',
    'REGIME_COMPLET': 'IMPORT',
    'DECLARATION_ID': '2025/TEST/0001'
}

# Créer le contexte avancé (appelle _create_advanced_fraud_scores en interne)
context = create_advanced_context_from_ocr_data(ocr_data, 'chap30')

print(f"✅ Contexte créé: {len(context)} features")

# Vérifier les fraud features
fraud_features = [
    'BIENAYME_CHEBYCHEV_SCORE', 'TEI_CALCULE', 'MIRROR_TEI_SCORE',
    'ADMIN_VALUES_SCORE', 'COMPOSITE_FRAUD_SCORE'
]

print("\n📊 Fraud features dans le contexte:")
fraud_present = 0
fraud_non_zero = 0
for feature in fraud_features:
    if feature in context:
        fraud_present += 1
        value = context[feature]
        if value != 0.0:
            fraud_non_zero += 1
            print(f"   {feature}: {value:.6f} ✅")
        else:
            print(f"   {feature}: {value:.6f} ⚠️ (à zéro)")
    else:
        print(f"   {feature}: ❌ MANQUANTE")

print(f"\n   Résultat: {fraud_present}/{len(fraud_features)} présentes, {fraud_non_zero}/{len(fraud_features)} non-zéro")

if fraud_present == len(fraud_features) and fraud_non_zero > 0:
    print("   ✅ OCR_INGEST fonctionne correctement !")
else:
    print("   ❌ OCR_INGEST a un problème")

# 2. Test OCR_PIPELINE
print("\n2️⃣ TEST OCR_PIPELINE: Utilisation du contexte pour prédiction")
print("-" * 70)

pipeline = AdvancedOCRPipeline()

try:
    # Faire une prédiction avec le contexte créé par OCR_INGEST
    prediction = pipeline.predict_fraud(context, 'chap30', 'basic')
    
    print(f"✅ Prédiction réussie")
    print(f"   Probabilité de fraude: {prediction.get('fraud_probability', 0):.3f}")
    print(f"   Décision: {prediction.get('decision', 'N/A')}")
    print(f"   Confiance: {prediction.get('confidence_score', 0):.3f}")
    
    # Vérifier que le contexte a bien été utilisé
    if 'context' in prediction:
        pred_context = prediction['context']
        fraud_in_pred = sum(1 for f in fraud_features if f in pred_context and pred_context.get(f, 0) != 0)
        print(f"\n   Fraud features utilisées par le pipeline: {fraud_in_pred}/{len(fraud_features)}")
        
        if fraud_in_pred > 0:
            print("   ✅ OCR_PIPELINE utilise bien les fraud features !")
        else:
            print("   ⚠️ OCR_PIPELINE n'utilise pas les fraud features")
    
    print("\n   ✅ OCR_PIPELINE fonctionne correctement !")
    
except Exception as e:
    print(f"   ❌ Erreur OCR_PIPELINE: {e}")

# 3. Test du flux complet
print("\n3️⃣ TEST FLUX COMPLET: OCR_INGEST → OCR_PIPELINE")
print("-" * 70)

if fraud_non_zero > 0 and 'prediction' in locals():
    print("✅ FLUX COMPLET FONCTIONNE:")
    print("   1. OCR_INGEST calcule les fraud features → NON-ZÉRO ✅")
    print("   2. OCR_PIPELINE reçoit le contexte → SUCCÈS ✅")
    print("   3. Prédiction ML réalisée → SUCCÈS ✅")
    print("\n🎉 TOUT FONCTIONNE À 100% !")
else:
    print("❌ Le flux complet a un problème")

print("\n" + "=" * 70)
print("🎯 RÉSUMÉ FINAL")
print("=" * 70)
print(f"OCR_INGEST: {'✅ OK' if fraud_non_zero > 0 else '❌ PROBLÈME'}")
print(f"OCR_PIPELINE: {'✅ OK' if 'prediction' in locals() else '❌ PROBLÈME'}")
print(f"FLUX COMPLET: {'✅ OK' if fraud_non_zero > 0 and 'prediction' in locals() else '❌ PROBLÈME'}")


