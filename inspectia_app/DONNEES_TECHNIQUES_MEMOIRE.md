# DONNÉES TECHNIQUES INTÉGRÉES DANS LE MÉMOIRE

## 📊 FICHIERS DE RÉSULTATS INTÉGRÉS

### Fichiers JSON de résultats par chapitre

**Chapitre 30 - Produits pharmaceutiques :**
- `backend/results/chap30/ml_robust_report.json` - Performances ML détaillées
- `backend/results/chap30/optimal_thresholds.json` - Seuils optimaux de décision
- `backend/results/chap30/shap_analysis.json` - Analyse SHAP complète

**Chapitre 84 - Machines et équipements mécaniques :**
- `backend/results/chap84/ml_robust_report.json` - Performances ML détaillées
- `backend/results/chap84/optimal_thresholds.json` - Seuils optimaux de décision
- `backend/results/chap84/shap_analysis.json` - Analyse SHAP complète

**Chapitre 85 - Machines et équipements électriques :**
- `backend/results/chap85/ml_robust_report.json` - Performances ML détaillées
- `backend/results/chap85/optimal_thresholds.json` - Seuils optimaux de décision
- `backend/results/chap85/shap_analysis.json` - Analyse SHAP complète

## 🎯 INTÉGRATION DANS LE MÉMOIRE

### Section 3.2.3 - Résultats expérimentaux et performances des modèles
**Données extraites des fichiers JSON :**
- Performances exactes (F1-Score, AUC, Précision, Rappel)
- Taux de fraude par chapitre
- Nombre d'échantillons (train/test)
- Métriques de calibration (Brier Score, ECE, BSS)

### Section 4.2.5 - Analyse SHAP et interprétabilité des modèles
**Données extraites des fichiers SHAP :**
- Top features par chapitre avec corrélations
- Importance des features business
- Analyses d'interprétabilité des modèles

### Section 4.3 - Évaluation et validation des modèles
**Données extraites des rapports ML :**
- Comparaisons inter-algorithmes
- Métriques de performance détaillées
- Validation croisée et régularisation

## 📈 VISUALISATIONS INTÉGRÉES

### Images PNG organisées dans `images/`
- **48 graphiques** générés automatiquement
- **Matrices de confusion** pour chaque chapitre
- **Courbes ROC** avec AUC exactes
- **Analyses SHAP** avec importance des features
- **Comparaisons inter-algorithmes**

### Emplacements dans le mémoire
- **Figures 3.1 à 3.18** : Performances par chapitre
- **Figures 3.19 à 3.24** : Comparaisons inter-algorithmes
- **Index des figures** : Référencement complet

## 🔧 DONNÉES TECHNIQUES UTILISÉES

### Métriques de performance confirmées
```json
{
  "chap30": {
    "model": "xgboost",
    "f1_score": 0.971,
    "auc_score": 0.996,
    "precision": 0.997,
    "recall": 0.946,
    "brier_score": 0.0058,
    "ece": 0.0024,
    "bss": 0.9403
  },
  "chap84": {
    "model": "catboost",
    "f1_score": 0.997,
    "auc_score": 0.999,
    "precision": 0.996,
    "recall": 0.999,
    "brier_score": 0.0003,
    "ece": 0.0000,
    "bss": 0.9964
  },
  "chap85": {
    "model": "xgboost",
    "f1_score": 0.965,
    "auc_score": 0.994,
    "precision": 0.990,
    "recall": 0.942,
    "brier_score": 0.0030,
    "ece": 0.0006,
    "bss": 0.9891
  }
}
```

### Seuils optimaux confirmés
```json
{
  "chap30": {
    "conforme": 0.2,
    "fraude": 0.8,
    "optimal_threshold": 0.5
  },
  "chap84": {
    "conforme": 0.1,
    "fraude": 0.9,
    "optimal_threshold": 0.5
  },
  "chap85": {
    "conforme": 0.192,
    "fraude": 0.557,
    "optimal_threshold": 0.5
  }
}
```

### Features SHAP confirmées
```json
{
  "chap30": {
    "top_features": [
      "BUSINESS_POIDS_NET_KG_EXCEPTIONNEL: 0.2883",
      "BUSINESS_VALEUR_CAF_EXCEPTIONNEL: 0.2883",
      "BUSINESS_SOUS_EVALUATION: 0.2883"
    ]
  },
  "chap84": {
    "top_features": [
      "BUSINESS_RISK_PAYS_ORIGINE: 0.4803",
      "BUSINESS_IS_ELECTROMENAGER: 0.4436",
      "BUSINESS_DETOURNEMENT_REGIME: 0.4376"
    ]
  },
  "chap85": {
    "top_features": [
      "BUSINESS_FAUSSE_DECLARATION_ESPECE: 0.6891",
      "BUSINESS_TAUX_DROITS_ELEVE: -0.4443",
      "BUSINESS_TAUX_DROITS_TRES_ELEVE: -0.4413"
    ]
  }
}
```

## ✅ VALIDATION ACADÉMIQUE

### Sources de données
- **Base DGD** : Données réelles de la Direction Générale des Douanes du Sénégal
- **Période** : 2018-2023 (6 années de données historiques)
- **Volume** : 324,220 échantillons au total
- **Validation** : Données anonymisées et validées par les services techniques

### Reproductibilité
- **Fichiers JSON** : Résultats reproductibles et vérifiables
- **Code source** : Algorithmes ML documentés et versionnés
- **Paramètres** : Hyperparamètres exacts pour chaque modèle
- **Métriques** : Calculs standardisés et validés

### Transparence
- **Analyses SHAP** : Interprétabilité complète des modèles
- **Visualisations** : Graphiques générés automatiquement
- **Rapports** : Documentation technique complète
- **Traçabilité** : Historique des expérimentations

## 🎯 CONCLUSION

Les fichiers de résultats JSON, YAML et les visualisations PNG ne sont **PAS** des annexes mais des **données techniques intégrées** dans le contenu principal du mémoire. Ils fournissent :

1. **Preuves concrètes** des performances des modèles
2. **Données reproductibles** pour validation académique
3. **Analyses détaillées** de l'interprétabilité
4. **Visualisations** des résultats expérimentaux

Ces données techniques renforcent la crédibilité académique du mémoire en fournissant des preuves tangibles et reproductibles des performances exceptionnelles d'InspectIA.
