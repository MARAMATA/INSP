#!/usr/bin/env python3
"""Test des scores de fraude pour comprendre pourquoi tout est frauduleux"""
import sys
sys.path.append('inspectia_app/backend/src')

from shared.ocr_ingest import _create_advanced_fraud_scores

# Test avec des données normales
test_context = {
    'VALEUR_CAF': 1000000.0,  # 1M FCFA - valeur normale
    'POIDS_NET_KG': 100.0,
    'MONTANT_LIQUIDATION': 150000.0,  # 15% de VALEUR_CAF
    'CODE_PRODUIT_STR': '8517120000',
    'PAYS_ORIGINE_STR': 'CN',
    'PRECISION_UEMOA': 90
}

print("="*80)
print("🧪 TEST DES SCORES DE FRAUDE")
print("="*80)

for chapter in ['chap30', 'chap84', 'chap85']:
    print(f"\n📊 CHAPITRE: {chapter}")
    scores = _create_advanced_fraud_scores(test_context, chapter)
    
    print(f"   BIENAYME_CHEBYCHEV_SCORE: {scores.get('BIENAYME_CHEBYCHEV_SCORE', 0):.4f}")
    print(f"   MIRROR_TEI_SCORE: {scores.get('MIRROR_TEI_SCORE', 0):.4f}")
    print(f"   ADMIN_VALUES_SCORE: {scores.get('ADMIN_VALUES_SCORE', 0):.4f}")
    print(f"   COMPOSITE_FRAUD_SCORE: {scores.get('COMPOSITE_FRAUD_SCORE', 0):.4f}")
    print(f"   TEI_CALCULE: {scores.get('TEI_CALCULE', 0):.4f}%")

