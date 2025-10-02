# INDEX DES FIGURES - MÉMOIRE INSPECTIA

## 📊 FIGURES DE PERFORMANCE PAR CHAPITRE

### Chapitre 30 - Produits pharmaceutiques
- **Figure 3.1** : `images/chap30/confusion_matrix_best.png` - Matrice de confusion XGBoost (F1: 0.971, AUC: 0.996)
- **Figure 3.2** : `images/chap30/roc_curve_best.png` - Courbe ROC avec AUC = 0.996
- **Figure 3.3** : `images/chap30/precision_recall_curve_best.png` - Courbe Précision-Rappel
- **Figure 3.4** : `images/chap30/metrics_best.png` - Métriques de performance détaillées
- **Figure 3.5** : `images/chap30/shap_summary_plot_20.png` - Analyse SHAP des 20 features principales
- **Figure 3.6** : `images/chap30/shap_feature_importance_20.png` - Importance des features SHAP

### Chapitre 84 - Machines et équipements mécaniques
- **Figure 3.7** : `images/chap84/confusion_matrix_best.png` - Matrice de confusion CatBoost (F1: 0.997, AUC: 0.999)
- **Figure 3.8** : `images/chap84/roc_curve_best.png` - Courbe ROC avec AUC = 0.999
- **Figure 3.9** : `images/chap84/precision_recall_curve_best.png` - Courbe Précision-Rappel
- **Figure 3.10** : `images/chap84/metrics_best.png` - Métriques de performance détaillées
- **Figure 3.11** : `images/chap84/shap_summary_plot_20.png` - Analyse SHAP des 20 features principales
- **Figure 3.12** : `images/chap84/shap_feature_importance_20.png` - Importance des features SHAP

### Chapitre 85 - Machines et équipements électriques
- **Figure 3.13** : `images/chap85/confusion_matrix_best.png` - Matrice de confusion XGBoost (F1: 0.965, AUC: 0.994)
- **Figure 3.14** : `images/chap85/roc_curve_best.png` - Courbe ROC avec AUC = 0.994
- **Figure 3.15** : `images/chap85/precision_recall_curve_best.png` - Courbe Précision-Rappel
- **Figure 3.16** : `images/chap85/metrics_best.png` - Métriques de performance détaillées
- **Figure 3.17** : `images/chap85/shap_summary_plot_20.png` - Analyse SHAP des 20 features principales
- **Figure 3.18** : `images/chap85/shap_feature_importance_20.png` - Importance des features SHAP

## 📈 FIGURES DE COMPARAISON INTER-ALGORITHMES

- **Figure 3.19** : `images/chap30/confusion_matrices_all.png` - Matrices de confusion pour tous les algorithmes (Chap 30)
- **Figure 3.20** : `images/chap30/roc_curves_all.png` - Comparaison des courbes ROC (Chap 30)
- **Figure 3.21** : `images/chap30/precision_recall_curves_all.png` - Comparaison des courbes Précision-Rappel (Chap 30)
- **Figure 3.22** : `images/chap30/metrics_comparison_all.png` - Comparaison des métriques de performance (Chap 30)
- **Figure 3.23** : `images/chap30/roc_comparison_all_algorithms.png` - Comparaison ROC inter-algorithmes (Chap 30)
- **Figure 3.24** : `images/chap30/metrics_comparison_all_algorithms.png` - Comparaison métriques inter-algorithmes (Chap 30)

## 🎯 EMPLACEMENTS DANS LE MÉMOIRE

### Section 3.2.3 - Résultats expérimentaux et performances des modèles
- **Figures 3.1 à 3.18** : Illustrer les performances par chapitre
- **Utilisation** : Montrer les résultats concrets des modèles ML

### Section 4.2.5 - Analyse SHAP et interprétabilité des modèles
- **Figures 3.5, 3.6, 3.11, 3.12, 3.17, 3.18** : Analyses SHAP
- **Utilisation** : Expliquer l'interprétabilité et l'importance des features

### Section 4.3 - Évaluation et validation des modèles
- **Figures 3.19 à 3.24** : Comparaisons inter-algorithmes
- **Utilisation** : Analyser les performances relatives des 5 algorithmes ML

## 📁 STRUCTURE DES FICHIERS

```
images/
├── chap30/
│   ├── confusion_matrix_best.png
│   ├── roc_curve_best.png
│   ├── precision_recall_curve_best.png
│   ├── metrics_best.png
│   ├── shap_summary_plot_20.png
│   ├── shap_feature_importance_20.png
│   ├── confusion_matrices_all.png
│   ├── roc_curves_all.png
│   ├── precision_recall_curves_all.png
│   ├── metrics_comparison_all.png
│   ├── roc_comparison_all_algorithms.png
│   └── metrics_comparison_all_algorithms.png
├── chap84/
│   ├── confusion_matrix_best.png
│   ├── roc_curve_best.png
│   ├── precision_recall_curve_best.png
│   ├── metrics_best.png
│   ├── shap_summary_plot_20.png
│   ├── shap_feature_importance_20.png
│   └── [autres fichiers de comparaison]
└── chap85/
    ├── confusion_matrix_best.png
    ├── roc_curve_best.png
    ├── precision_recall_curve_best.png
    ├── metrics_best.png
    ├── shap_summary_plot_20.png
    ├── shap_feature_importance_20.png
    └── [autres fichiers de comparaison]
```

## 📋 DONNÉES RÉELLES INTÉGRÉES

### Performances confirmées par les fichiers JSON :
- **Chap 30** : XGBoost - F1: 0.971, AUC: 0.996, Brier: 0.0058
- **Chap 84** : CatBoost - F1: 0.997, AUC: 0.999, Brier: 0.0003
- **Chap 85** : XGBoost - F1: 0.965, AUC: 0.994, Brier: 0.0030

### Seuils optimaux confirmés :
- **Chap 30** : conforme < 0.2, fraude > 0.8
- **Chap 84** : conforme < 0.1, fraude > 0.9
- **Chap 85** : conforme < 0.192, fraude > 0.557

### Features SHAP confirmées :
- **Chap 30** : BUSINESS_POIDS_NET_KG_EXCEPTIONNEL (corr: +0.2883)
- **Chap 84** : BUSINESS_RISK_PAYS_ORIGINE (corr: +0.4803)
- **Chap 85** : BUSINESS_FAUSSE_DECLARATION_ESPECE (corr: +0.6891)
