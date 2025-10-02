# 📊 SCHÉMA POSTGRESQL COMPLET - INSPECTIA

## 🎯 Vue d'ensemble

Le schéma PostgreSQL pour InspectIA a été conçu pour supporter l'ensemble du système de détection de fraude douanière avec ML-RL, incluant **TOUTES** les features exactes utilisées par chaque modèle ML.

## 📋 Tables Principales

### 1. **Tables de Configuration**
- `chapters` - Chapitres douaniers (30, 84, 85)
- `models` - Modèles ML entraînés
- `features` - Features utilisées par les modèles
- `chapter_features` - Associations chapitres-features

### 2. **Tables de Données**
- `declarations` - Déclarations douanières
- `predictions` - Prédictions ML
- `declaration_features` - Features extraites par déclaration

### 3. **Tables Système RL**
- `rl_decisions` - Décisions du système RL
- `inspector_profiles` - Profils d'inspecteurs
- `feedback_history` - Historique des feedbacks

### 4. **Tables d'Analyse**
- `analysis_results` - Résultats d'analyse détaillée
- `model_thresholds` - Seuils et configurations
- `performance_metrics` - Métriques de performance
- `system_logs` - Logs système

## 🔧 Features Exactes par Chapitre

### **Chapitre 30 - Pharmaceutique (22 features)**
```
1. POIDS_NET_KG
2. NOMBRE_COLIS
3. QUANTITE_COMPLEMENT
4. TAUX_DROITS_PERCENT
5. BUSINESS_POIDS_NET_KG_EXCEPTIONNEL
6. BUSINESS_VALEUR_CAF_EXCEPTIONNEL
7. BUSINESS_SOUS_EVALUATION
8. BUSINESS_QUANTITE_COMPLEMENT_EXCEPTIONNEL
9. BUSINESS_NOMBRE_COLIS_EXCEPTIONNEL
10. BUSINESS_DROITS_EXCEPTIONNELS
11. BUSINESS_LIQUIDATION_COMPLEMENTAIRE
12. BUSINESS_RATIO_LIQUIDATION_CAF
13. BUSINESS_ALERTE_SUSPECT
14. BUSINESS_INCOHERENCE_CONDITIONNEMENT
15. CODE_SH_COMPLET
16. CODE_PAYS_ORIGINE
17. CODE_PAYS_PROVENANCE
18. REGIME_COMPLET
19. STATUT_BAE
20. TYPE_REGIME
21. REGIME_DOUANIER
22. REGIME_FISCAL
```

### **Chapitre 84 - Mécanique (21 features)**
```
1. POIDS_NET_KG
2. NOMBRE_COLIS
3. QUANTITE_COMPLEMENT
4. TAUX_DROITS_PERCENT
5. BUSINESS_RISK_PAYS_ORIGINE
6. BUSINESS_IS_ELECTROMENAGER
7. BUSINESS_DETOURNEMENT_REGIME
8. BUSINESS_FAUSSE_DECLARATION_ASSEMBLAGE
9. BUSINESS_FAUSSE_DECLARATION_ESPECE
10. BUSINESS_SOUS_EVALUATION
11. BUSINESS_QUANTITE_ANORMALE
12. BUSINESS_IS_MACHINE_BUREAU
13. BUSINESS_VALEUR_ELEVEE
14. CODE_SH_COMPLET
15. CODE_PAYS_ORIGINE
16. CODE_PAYS_PROVENANCE
17. REGIME_COMPLET
18. STATUT_BAE
19. TYPE_REGIME
20. REGIME_DOUANIER
21. REGIME_FISCAL
```

### **Chapitre 85 - Électrique (23 features)**
```
1. POIDS_NET_KG
2. NOMBRE_COLIS
3. QUANTITE_COMPLEMENT
4. TAUX_DROITS_PERCENT
5. BUSINESS_FAUSSE_DECLARATION_ESPECE
6. BUSINESS_TAUX_DROITS_ELEVE
7. BUSINESS_TAUX_DROITS_TRES_ELEVE
8. BUSINESS_RATIO_LIQUIDATION_CAF
9. BUSINESS_INCOHERENCE_CLASSIFICATION
10. BUSINESS_IS_TELEPHONES
11. BUSINESS_DETOURNEMENT_REGIME
12. BUSINESS_VALEUR_ELEVEE
13. BUSINESS_IS_GROUPES_ELECTROGENES
14. BUSINESS_IS_MACHINES_ELECTRIQUES
15. BUSINESS_SOUS_EVALUATION
16. CODE_SH_COMPLET
17. CODE_PAYS_ORIGINE
18. CODE_PAYS_PROVENANCE
19. REGIME_COMPLET
20. STATUT_BAE
21. TYPE_REGIME
22. REGIME_DOUANIER
23. REGIME_FISCAL
```

## 🏗️ Architecture des Relations

### **Relations Principales**
```
chapters (1) ←→ (N) models
chapters (1) ←→ (N) declarations
chapters (1) ←→ (N) predictions
chapters (1) ←→ (N) rl_decisions
chapters (1) ←→ (N) feedback_history
chapters (1) ←→ (N) analysis_results

declarations (1) ←→ (N) predictions
declarations (1) ←→ (N) declaration_features
declarations (1) ←→ (N) rl_decisions
declarations (1) ←→ (N) feedback_history
declarations (1) ←→ (N) analysis_results

models (1) ←→ (N) predictions
models (1) ←→ (N) model_thresholds
models (1) ←→ (N) performance_metrics

features (1) ←→ (N) declaration_features
features (N) ←→ (N) chapters (via chapter_features)

inspector_profiles (1) ←→ (N) feedback_history
```

## 📊 Types de Features

### **Features Numériques (4)**
- `POIDS_NET_KG` - Poids net en kilogrammes
- `NOMBRE_COLIS` - Nombre de colis
- `QUANTITE_COMPLEMENT` - Quantité complémentaire
- `TAUX_DROITS_PERCENT` - Taux de droits en pourcentage
- `VALEUR_CAF` - Valeur CAF (pour calculs business)

### **Features Catégorielles (8)**
- `CODE_SH_COMPLET` - Code SH complet
- `CODE_PAYS_ORIGINE` - Code pays d'origine
- `CODE_PAYS_PROVENANCE` - Code pays de provenance
- `REGIME_COMPLET` - Régime complet
- `STATUT_BAE` - Statut BAE
- `TYPE_REGIME` - Type de régime
- `REGIME_DOUANIER` - Régime douanier
- `REGIME_FISCAL` - Régime fiscal

### **Features Business (Spécifiques par chapitre)**
- **Chapitre 30**: 10 features business pharmaceutiques
- **Chapitre 84**: 9 features business mécaniques
- **Chapitre 85**: 11 features business électriques

## 🚀 Installation et Configuration

### **1. Installation PostgreSQL**
```bash
brew install postgresql@15
brew services start postgresql@15
createdb inspectia_db
```

### **2. Création du schéma**
```bash
cd backend/database
python3 install_database.py
```

### **3. Vérification des features**
```bash
python3 verify_features.py
```

## 🔍 Vérifications Incluses

Le schéma inclut des vérifications complètes pour s'assurer que :

✅ **Toutes les features des modèles ML sont incluses**
✅ **L'ordre des features correspond exactement aux modèles**
✅ **Les associations chapitres-features sont correctes**
✅ **Les types de données sont appropriés**
✅ **Les contraintes d'intégrité sont respectées**

## 📈 Optimisations

### **Index de Performance**
- Index sur les clés étrangères
- Index sur les colonnes de recherche fréquente
- Index sur les timestamps pour les requêtes temporelles

### **Vues Utilitaires**
- `declarations_with_predictions` - Vue combinée déclarations-prédictions
- `chapter_statistics` - Statistiques par chapitre

### **Triggers**
- Mise à jour automatique des timestamps `updated_at`
- Validation des données d'entrée

## 🎯 Avantages du Schéma

1. **Complet** - Inclut toutes les features exactes des modèles
2. **Extensible** - Facile d'ajouter de nouveaux chapitres/features
3. **Performant** - Index optimisés pour les requêtes fréquentes
4. **Intégré** - Support complet du système ML-RL
5. **Traçable** - Historique complet des décisions et feedbacks
6. **Analytique** - Métriques et logs pour le monitoring

## 📝 Prochaines Étapes

1. **Installation** - Exécuter le script d'installation
2. **Vérification** - Valider que toutes les features sont incluses
3. **Intégration Backend** - Adapter le code pour utiliser PostgreSQL
4. **Migration** - Migrer les données SQLite existantes
5. **Tests** - Tester l'intégration complète

