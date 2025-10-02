# 📊 Features par Chapitre - InspectIA

## 🧪 Chapitre 30 (Pharmaceutique) - 22 Features

### Features Numériques (4)
- `POIDS_NET_KG` - Poids net en kilogrammes
- `NOMBRE_COLIS` - Nombre de colis
- `QUANTITE_COMPLEMENT` - Quantité complémentaire
- `TAUX_DROITS_PERCENT` - Taux de droits en pourcentage

### Features Business (10)
- `BUSINESS_POIDS_NET_KG_EXCEPTIONNEL` (corr: +0.2883) - Poids exceptionnel
- `BUSINESS_VALEUR_CAF_EXCEPTIONNEL` (corr: +0.2883) - Valeur CAF exceptionnelle
- `BUSINESS_SOUS_EVALUATION` (corr: +0.2883) - Sous-évaluation
- `BUSINESS_QUANTITE_COMPLEMENT_EXCEPTIONNEL` (corr: +0.2880) - Quantité complémentaire exceptionnelle
- `BUSINESS_NOMBRE_COLIS_EXCEPTIONNEL` (corr: +0.2877) - Nombre de colis exceptionnel
- `BUSINESS_DROITS_EXCEPTIONNELS` (corr: +0.2875) - Droits exceptionnels
- `BUSINESS_LIQUIDATION_COMPLEMENTAIRE` - Liquidation complémentaire
- `BUSINESS_RATIO_LIQUIDATION_CAF` (corr: +0.2241) - Ratio liquidation CAF
- `BUSINESS_ALERTE_SUSPECT` (corr: +0.1645) - Alerte suspect
- `BUSINESS_INCOHERENCE_CONDITIONNEMENT` (corr: +0.1641) - Incohérence conditionnement

### Features Catégorielles (8)
- `CODE_SH_COMPLET` - Code SH complet
- `CODE_PAYS_ORIGINE` - Code pays d'origine
- `CODE_PAYS_PROVENANCE` - Code pays de provenance
- `REGIME_COMPLET` - Régime complet
- `STATUT_BAE` - Statut BAE
- `TYPE_REGIME` - Type de régime
- `REGIME_DOUANIER` - Régime douanier
- `REGIME_FISCAL` - Régime fiscal

---

## ⚙️ Chapitre 84 (Machines) - 21 Features

### Features Numériques (4)
- `POIDS_NET_KG` - Poids net en kilogrammes
- `NOMBRE_COLIS` - Nombre de colis
- `QUANTITE_COMPLEMENT` - Quantité complémentaire
- `TAUX_DROITS_PERCENT` - Taux de droits en pourcentage

### Features Business (10)
- `BUSINESS_RISK_PAYS_ORIGINE` (corr: +0.4803) 🏆 - Risque pays origine
- `BUSINESS_IS_ELECTROMENAGER` (corr: +0.4436) 🏆 - Électroménager
- `BUSINESS_DETOURNEMENT_REGIME` (corr: +0.4376) 🏆 - Détournement régime
- `BUSINESS_FAUSSE_DECLARATION_ASSEMBLAGE` (corr: +0.4231) 🏆 - Fausse déclaration assemblage
- `BUSINESS_FAUSSE_DECLARATION_ESPECE` (corr: +0.4130) 🏆 - Fausse déclaration espèce
- `BUSINESS_SOUS_EVALUATION` (corr: +0.4059) 🏆 - Sous-évaluation
- `BUSINESS_QUANTITE_ANORMALE` (corr: +0.4046) 🏆 - Quantité anormale
- `BUSINESS_IS_MACHINE_BUREAU` (corr: +0.3363) 🏆 - Machine bureau
- `BUSINESS_VALEUR_ELEVEE` (corr: +0.3117) 🏆 - Valeur élevée
- `BUSINESS_ALERTE_SUSPECT` (corr: +0.1645) 🏆 - Alerte suspect

### Features Catégorielles (7)
- `CODE_SH_COMPLET` - Code SH complet
- `CODE_PAYS_ORIGINE` - Code pays d'origine
- `CODE_PAYS_PROVENANCE` - Code pays de provenance
- `REGIME_COMPLET` - Régime complet
- `STATUT_BAE` - Statut BAE
- `TYPE_REGIME` - Type de régime
- `REGIME_DOUANIER` - Régime douanier
- `REGIME_FISCAL` - Régime fiscal

---

## ⚡ Chapitre 85 (Électrique) - 22 Features

### Features Numériques (4)
- `POIDS_NET_KG` - Poids net en kilogrammes
- `NOMBRE_COLIS` - Nombre de colis
- `QUANTITE_COMPLEMENT` - Quantité complémentaire
- `TAUX_DROITS_PERCENT` - Taux de droits en pourcentage

### Features Business (11)
- `BUSINESS_FAUSSE_DECLARATION_ESPECE` (corr: +0.6891) 🏆 - Fausse déclaration espèce
- `BUSINESS_TAUX_DROITS_ELEVE` (corr: -0.4443) - Taux droits élevé
- `BUSINESS_TAUX_DROITS_TRES_ELEVE` (corr: -0.4413) - Taux droits très élevé
- `BUSINESS_RATIO_LIQUIDATION_CAF` (corr: -0.4330) - Ratio liquidation CAF
- `BUSINESS_INCOHERENCE_CLASSIFICATION` (corr: +0.3991) - Incohérence classification
- `BUSINESS_IS_TELEPHONES` (corr: +0.3952) - Téléphones
- `BUSINESS_DETOURNEMENT_REGIME` (corr: +0.3477) - Détournement régime
- `BUSINESS_VALEUR_ELEVEE` (corr: +0.2535) - Valeur élevée
- `BUSINESS_IS_GROUPES_ELECTROGENES` (corr: +0.2165) - Groupes électrogènes
- `BUSINESS_IS_MACHINES_ELECTRIQUES` (corr: +0.1706) - Machines électriques
- `BUSINESS_SOUS_EVALUATION` (corr: +0.1931) - Sous-évaluation

### Features Catégorielles (7)
- `CODE_SH_COMPLET` - Code SH complet
- `CODE_PAYS_ORIGINE` - Code pays d'origine
- `CODE_PAYS_PROVENANCE` - Code pays de provenance
- `REGIME_COMPLET` - Régime complet
- `STATUT_BAE` - Statut BAE
- `TYPE_REGIME` - Type de régime
- `REGIME_DOUANIER` - Régime douanier
- `REGIME_FISCAL` - Régime fiscal

---

## 🔍 Observations Importantes

1. **Features Communes** : Tous les chapitres partagent les mêmes features numériques et catégorielles de base
2. **Features Business Spécifiques** : Chaque chapitre a ses propres features business optimisées selon les caractéristiques du secteur
3. **Protection Data Leakage** : Toutes les features excluent les variables post-événement comme `VALEUR_CAF`, `MONTANT_LIQUIDATION`
4. **Corrélations** : Les features business sont classées par ordre de corrélation avec `FRAUD_FLAG`
5. **Alignement** : Le nombre de features doit correspondre exactement entre le contexte et le modèle ML

## ⚠️ Features Exclues (Data Leakage)

Ces features ne sont PAS utilisées dans les modèles car elles créent du data leakage :
- `VALEUR_CAF` - Utilisée pour calculer les features business mais pas dans le modèle
- `MONTANT_LIQUIDATION` - Post-événement
- `VALEUR_UNITAIRE_PAR_KG` - Calculée à partir de VALEUR_CAF
- `CODE_DECLARANT`, `CODE_DESTINATAIRE` - Identifiants spécifiques
- `BUREAU` - Peut être corrélé avec FRAUD_FLAG
