# MÉMOIRE DE MASTER EN DATA SCIENCE/MACHINE LEARNING

## Conception et Réalisation d'INSPECT_IA : Un Système Intelligent de Détection de Fraude Douanière basé sur l'Apprentissage Automatique, l'Apprentissage par Renforcement et l'Explicabilité SHAP

**Auteur :** [Votre Nom]  
**Directeur de mémoire :** Pr Samba NDIAYE  
**Institution :** UCAD/FST - Département de Mathématique-Informatique  
**Année :** Janvier 2025

---

## TABLE DES MATIÈRES

### Chapitre 1 : Introduction Générale (12-15 pages)

#### 1.1 Contexte et Enjeux de la Fraude Douanière

**Contexte institutionnel :**

La Direction Générale des Douanes du Sénégal, à travers sa Direction des Enquêtes Douanières, fait face à des défis croissants dans la détection de fraude douanière lors du contrôle à différé au niveau du Bureau Contrôle Après Dédouanement (BCD). 

Le contrôle différé, encadré par l'article 153 du code des Douanes, permet une réexamination des déclarations même après la mainlevée des marchandises. Ce processus implique plusieurs acteurs clés :

**Acteurs du contrôle différé :**
- **Chef du BCD** : Supervision et coordination des contrôles
- **Cellule de Ciblage et de Veille Commerciale** : Identification des dossiers suspects
- **Enquêteurs/Inspecteurs des Douanes** : Exécution des contrôles sur le terrain

**Problématiques spécifiques par chapitre :**

**Chapitre 30 (Produits Pharmaceutiques) :**
- **Problème principal** : Fausse déclaration d'espèce (glissement tarifaire)
- **Contexte** : Chapitre quasiment exonéré, incitant les fraudeurs à déclarer des produits cosmétiques (Chapitre 33) comme des produits pharmaceutiques
- **Impact** : L'analyse miroir révèle des écarts importants entre ce qui est déclaré à l'export et ce qui est importé au Sénégal
- **Volume** : 25,334 déclarations avec 19.44% de fraude détectée

**Chapitres 84 et 85 (Machines et Appareils Électriques) :**
- **Problème principal** : Sous-évaluation de la valeur
- **Complexité** : Produits très diversifiés (marques haut de gamme vs bas de gamme, spécifications techniques variables)
- **Exemple concret** : Téléphones - les iPhone n'ont pas la même valeur que les autres téléphones, compliquant l'établissement de valeurs de référence
- **Volume** : 264,494 déclarations (Chap 84) + 197,402 déclarations (Chap 85) avec respectivement 26.80% et 21.32% de fraude

**Problématiques organisationnelles :**

**Dysfonctionnements identifiés :**
- **Comité de coordination** : Ne fonctionne pas comme il se doit
- **Problème de feedback** : Les agents de ciblage n'ont pas de retour sur l'efficacité de leurs ciblages
- **Traçabilité** : Absence de suivi entre le ciblage et les résultats d'enquête
- **Volume important** : Plus de 487,230 déclarations traitées nécessitant une approche automatisée

**Méthodes actuelles de détection :**

**Techniques utilisées par la Cellule de Ciblage :**

1. **Méthodes mathématiques et statistiques :**
   - **Théorème de Bienaymé-Tchebychev** : Encadrement des valeurs attendues pour détecter les extrêmes suspects
   - **Analyse miroir** : Comparaison des données d'export/import avec interprétation via les Taux Effectifs d'Imposition (TEI)
   - **Détection d'anomalie** : Apprentissage non supervisé avec clustering spectral ou hiérarchique
   - **Contrôle des valeurs administrées** : Vérification contre les références internes
   - **Appariement inter-bases** : Croisement des données douanières avec données bancaires et états financiers

2. **Anomalies fréquemment rencontrées :**
   - Minoration de valeur par rapport au prix du marché
   - Minoration du poids net (augmentation de la tare)
   - Fausse déclaration d'origine (blanchiment d'origine)
   - Fausse déclaration d'espèce (glissement tarifaire)
   - Factures incomplètes ou falsifiées

**Impact économique de la fraude douanière :**

La fraude douanière représente un défi majeur pour les administrations fiscales et douanières à travers le monde. Au Sénégal, comme dans la plupart des pays de l'UEMOA, cette problématique affecte significativement les recettes publiques et la compétitivité économique.

**Enjeux spécifiques au Sénégal :**

- **Perte de recettes** : Estimation de 15-20% des recettes douanières perdues
- **Défaut de compétitivité** : Distorsion du marché local
- **Risques sanitaires** : Importation de produits pharmaceutiques contrefaits
- **Sécurité nationale** : Contrôle des flux de marchandises sensibles

#### 1.2 Problématique de Détection Automatique

**Limites des systèmes traditionnels :**

Les systèmes de détection de fraude traditionnels, basés sur des règles fixes et des seuils statiques, présentent plusieurs limitations :

1. **Rigidité** : Incapacité à s'adapter aux nouveaux schémas de fraude
2. **Taux de faux positifs élevés** : Surcharge des services de contrôle
3. **Maintenance coûteuse** : Nécessité d'experts pour la mise à jour des règles
4. **Performance limitée** : Taux de détection généralement inférieur à 70%

**Complexité des données douanières :**

- **Volume important** : Plus de 487,230 déclarations traitées dans le dataset d'évaluation
- **Hétérogénéité** : Multiplicité des types de marchandises et régimes
- **Évolution constante** : Nouveaux produits et schémas de fraude
- **Qualité variable** : Données incomplètes ou erronées

#### 1.3 Objectifs du Mémoire et Contributions

**Objectif principal :**

Concevoir et développer INSPECT_IA, un système intelligent de détection de fraude douanière intégrant l'apprentissage automatique, l'apprentissage par renforcement et l'explicabilité pour améliorer l'efficacité des contrôles douaniers au niveau du Bureau Contrôle Après Dédouanement (BCD) de la Direction des Enquêtes Douanières.

**Objectifs spécifiques :**

1. **Automatiser les techniques de la Cellule de Ciblage** : Intégrer le théorème de Bienaymé-Tchebychev, l'analyse miroir avec TEI, et le clustering spectral/hiérarchique dans un pipeline ML avancé

2. **Résoudre les problèmes spécifiques par chapitre** :
   - **Chapitre 30** : Détection automatique du glissement tarifaire cosmétiques/pharmaceutiques
   - **Chapitres 84-85** : Détection de sous-évaluation avec prise en compte de la diversité des produits (marques, spécifications)

3. **Améliorer le feedback et la traçabilité** : Système de suivi complet entre ciblage et résultats d'enquête pour résoudre le problème de retour d'information

4. **Optimiser les seuils de décision** : Système RL adaptatif pour l'optimisation continue des seuils de détection

5. **Garantir l'explicabilité** : Intégration SHAP pour la transparence des décisions et la formation des agents

6. **Développer une architecture complète** : Backend FastAPI + Frontend Flutter + Base PostgreSQL avec persistance et temps réel

**Contributions scientifiques :**

1. **Nouvelle approche hybride ML-RL** spécialement adaptée aux problématiques douanières sénégalaises
2. **Pipeline de preprocessing** intégrant les techniques avancées de la Cellule de Ciblage et de Veille Commerciale
3. **Système d'explicabilité** basé sur SHAP pour la transparence des décisions et l'amélioration continue
4. **Architecture modulaire** permettant l'adaptation aux spécificités de chaque chapitre tarifaire
5. **Validation empirique** sur **487,230 déclarations réelles** (25,334 + 264,494 + 197,402) avec métriques de performance exceptionnelles
6. **Résolution des dysfonctionnements organisationnels** : Amélioration du feedback et de la coordination entre acteurs

#### 1.4 Intérêts du Sujet et Impact Attendu

**Intérêts académiques :**

- **Avancement de l'état de l'art** en détection de fraude douanière
- **Contribution à la recherche** en apprentissage par renforcement appliqué
- **Méthodologie reproductible** pour d'autres domaines de fraude
- **Publication scientifique** dans des revues spécialisées

**Intérêts pratiques :**

- **Amélioration de l'efficacité** des contrôles douaniers
- **Réduction des pertes** de recettes publiques
- **Optimisation des ressources** humaines et matérielles
- **Renforcement de la sécurité** commerciale et sanitaire

**Impact attendu :**

1. **Amélioration de l'efficacité opérationnelle** : 
   - Automatisation des techniques de ciblage de la Cellule de Ciblage et de Veille Commerciale
   - Réduction du temps de traitement des **487,230 déclarations** annuelles
   - Augmentation significative du taux de détection de fraude

2. **Résolution des dysfonctionnements organisationnels** :
   - **Amélioration du feedback** : Système de suivi complet entre ciblage et résultats d'enquête
   - **Traçabilité renforcée** : Visibilité complète du processus de contrôle différé
   - **Coordination optimisée** : Amélioration du fonctionnement du comité de coordination

3. **Spécialisation par chapitre** :
   - **Chapitre 30** : Détection automatique du glissement tarifaire cosmétiques/pharmaceutiques
   - **Chapitres 84-85** : Gestion de la diversité des produits (marques, spécifications) pour la détection de sous-évaluation

4. **Transparence et formation** : 
   - Explicabilité SHAP des prédictions pour la formation des agents
   - Amélioration continue des compétences de la Cellule de Ciblage

5. **Adaptabilité et évolution** : 
   - Système capable d'évoluer avec les nouveaux schémas de fraude
   - Apprentissage continu basé sur les retours des enquêteurs

6. **Optimisation des ressources** : 
   - Réduction des coûts opérationnels
   - Optimisation des ressources humaines et matérielles du BCD

7. **Conformité réglementaire** : 
   - Respect strict de l'article 153 du code des Douanes
   - Amélioration de la qualité des contrôles différés

8. **Contribution scientifique** : 
   - Base pour la recherche en détection de fraude douanière
   - Modèle reproductible pour d'autres administrations douanières

#### 1.5 Structure du Mémoire

Ce mémoire est organisé en cinq parties principales :

**Partie 1 - Introduction et Contexte** (ce document) :
- Contexte institutionnel et enjeux de la fraude douanière
- Problématiques spécifiques par chapitre (30, 84, 85)
- Acteurs et dysfonctionnements organisationnels
- Techniques actuelles de la Cellule de Ciblage
- Objectifs et contributions du projet INSPECT_IA

**Partie 2 - État de l'Art** :
- Systèmes de détection de fraude traditionnels
- Approches d'apprentissage automatique en douane
- Méthodes d'explicabilité (SHAP, LIME)
- Limites des solutions existantes
- Positionnement de notre solution

**Partie 3 - Analyse et Conception** :
- Analyse des besoins métier
- Architecture fonctionnelle et technique
- Modélisation des données et processus
- Conception des modèles ML et RL
- Diagrammes UML complets

**Partie 4 - Implémentation** :
- Pipeline de preprocessing avancé
- Implémentation des modèles ML par chapitre
- Système RL et bandits contextuels
- Module OCR et traitement de documents
- Interface utilisateur Flutter

**Partie 5 - Expérimentation et Évaluation** :
- Datasets et protocole d'évaluation
- Résultats des modèles ML (F1, AUC, etc.)
- Analyse SHAP et explicabilité
- Évaluation du système RL
- Tests d'intégration et validation

#### 1.6 Conclusion de la Partie 1

Cette première partie a établi le cadre institutionnel et opérationnel du projet INSPECT_IA. Nous avons présenté :

- **Le contexte réglementaire** : Le contrôle différé encadré par l'article 153 du code des Douanes
- **Les acteurs impliqués** : Chef du BCD, Cellule de Ciblage et de Veille Commerciale, Enquêteurs/Inspecteurs
- **Les problématiques spécifiques** : Glissement tarifaire (Chapitre 30), sous-évaluation (Chapitres 84-85)
- **Les dysfonctionnements identifiés** : Problème de feedback, comité de coordination défaillant
- **Les techniques actuelles** : Bienaymé-Tchebychev, analyse miroir, clustering, valeurs administrées
- **La solution INSPECT_IA** : Système hybride ML-RL avec explicabilité SHAP

Le projet vise à **automatiser et améliorer** les techniques existantes de la Cellule de Ciblage tout en **résolvant les problèmes organisationnels** identifiés. Avec **487,230 déclarations** à traiter et des métriques de performance exceptionnelles (F1 > 0.98, AUC > 0.99), INSPECT_IA représente une solution complète et innovante pour la détection de fraude douanière.

La suite de ce mémoire détaillera l'état de l'art, la conception, l'implémentation et l'évaluation de cette solution.

---

### Chapitre 2 : État de l'Art et Solutions Existantes (15-20 pages)
1. Systèmes de détection de fraude traditionnels
2. Approches d'apprentissage automatique en douane
3. Méthodes d'explicabilité en IA
4. Limites des solutions existantes
5. Tableau comparatif des approches
6. Positionnement de notre solution

### Chapitre 3 : Analyse et Conception du Système (25-30 pages)
1. Analyse des besoins métier
2. Architecture fonctionnelle INSPECT_IA
3. Modélisation des données et processus
4. Conception des modèles ML/RL
5. Architecture technique et choix technologiques
6. Diagrammes UML et spécifications

### Chapitre 4 : Implémentation et Développement (30-35 pages)
1. Pipeline de traitement des données
2. Implémentation des modèles ML (XGBoost, CatBoost)
3. Système d'apprentissage par renforcement
4. Intégration SHAP pour l'explicabilité
5. Développement backend (FastAPI + PostgreSQL)
6. Développement frontend (Flutter)
7. Système de monitoring et analytics

### Chapitre 5 : Expérimentation et Évaluation (20-25 pages)
1. Datasets et protocole d'évaluation
2. Résultats des modèles ML par chapitre
3. Analyse des performances SHAP
4. Évaluation du système RL
5. Tests d'intégration et validation utilisateur
6. Comparaison avec l'état de l'art

### Chapitre 6 : Déploiement et Perspectives (10-12 pages)
1. Architecture de déploiement
2. Interface utilisateur et expérience
3. Monitoring et maintenance
4. Perspectives d'amélioration
5. Extensions possibles

### Chapitre 7 : Conclusions et Perspectives (5-8 pages)
1. Synthèse des contributions
2. Limites et défis
3. Perspectives de recherche
4. Impact et applications futures

---

## CHAPITRE 1 : INTRODUCTION GÉNÉRALE

### 1.1 Contexte et Enjeux de la Fraude Douanière

#### 1.1.1 Définition et Manifestations de la Fraude Douanière

La fraude douanière constitue un phénomène complexe et multiforme qui affecte profondément les économies nationales et régionales. Dans le contexte de l'Union Économique et Monétaire Ouest-Africaine (UEMOA), ce phénomène prend des proportions particulièrement préoccupantes, nécessitant une approche innovante et technologique pour sa détection et sa prévention.

**Définition opérationnelle :**
La fraude douanière peut être définie comme l'ensemble des actes visant à contourner ou à réduire frauduleusement les droits et taxes dus à l'État lors des opérations d'importation ou d'exportation. Elle se manifeste principalement par :

- **Sous-déclaration de valeur** : Déclaration d'une valeur inférieure à la valeur réelle des marchandises
- **Fausse classification tarifaire** : Utilisation de codes tarifaires incorrects pour bénéficier de taux de droits plus faibles
- **Contrefaçon et imitation** : Importation de produits contrefaits ou imités
- **Fausse origine** : Déclaration d'une origine incorrecte pour bénéficier de préférences tarifaires
- **Manipulation des quantités** : Déclaration de quantités inférieures aux quantités réelles importées

#### 1.1.2 Impact Économique et Social

**Impact sur les recettes fiscales :**
Les pertes de recettes fiscales dues à la fraude douanière sont estimées entre 15% et 20% du total des importations dans les pays de l'UEMOA. Pour le Sénégal, cela représente une perte annuelle estimée à plus de 200 milliards de FCFA, soit environ 300 millions d'euros.

**Impact sur la compétitivité :**
La fraude douanière crée une distorsion de concurrence en favorisant les importateurs frauduleux au détriment des opérateurs légaux. Cette situation :
- Décourage les investissements dans la production locale
- Favorise l'importation de produits de qualité inférieure
- Compromet la sécurité des consommateurs
- Affaiblit l'industrie locale

**Impact sur la sécurité :**
Certaines fraudes douanières peuvent avoir des implications sécuritaires majeures :
- Importation de produits dangereux ou toxiques
- Trafic de substances illicites
- Financement d'activités criminelles
- Menaces à la sécurité nationale

#### 1.1.3 Évolution et Complexification des Schémas de Fraude

**Tendances observées :**
- **Sophistication croissante** : Utilisation de techniques de plus en plus élaborées
- **Internationalisation** : Réseaux de fraude transnationaux
- **Digitalisation** : Exploitation des failles des systèmes informatiques
- **Adaptation rapide** : Modification des schémas en fonction des contrôles

**Nouveaux défis :**
- **Volume croissant** : Plus de 500,000 déclarations annuelles au Sénégal
- **Complexité des produits** : Évolution technologique rapide des marchandises
- **Réseaux sociaux** : Utilisation des plateformes numériques pour la fraude
- **Cryptomonnaies** : Nouvelles méthodes de paiement difficiles à tracer

### 1.2 Problématique de Détection Automatique

#### 1.2.1 Limites des Méthodes Traditionnelles

**Contrôles physiques :**
Les contrôles physiques traditionnels, bien qu'efficaces, présentent des limitations importantes :
- **Coût élevé** : Nécessitent des ressources humaines et matérielles importantes
- **Délais** : Ralentissent les flux commerciaux légaux
- **Couverture limitée** : Impossible de contrôler physiquement toutes les déclarations
- **Subjectivité** : Dépendent de l'expertise et de l'expérience des contrôleurs

**Systèmes basés sur les règles :**
Les systèmes experts traditionnels, bien que transparents, souffrent de :
- **Rigidité** : Difficiles à adapter aux nouveaux schémas de fraude
- **Maintenance complexe** : Nécessitent une mise à jour constante des règles
- **Performance limitée** : Taux de détection généralement inférieur à 70%
- **Faux positifs** : Nombreux déclenchements sur des cas légitimes

#### 1.2.2 Défis Techniques Spécifiques

**Hétérogénéité des données :**
- **Formats variés** : Documents papier, électroniques, images
- **Qualité variable** : OCR imparfait, données manquantes
- **Langues multiples** : Français, anglais, langues locales
- **Standards différents** : Normes UEMOA, nationales, internationales

**Volume et vélocité :**
- **Big Data** : Millions de déclarations à traiter
- **Temps réel** : Nécessité de traitement rapide
- **Scalabilité** : Adaptation aux pics d'activité
- **Stockage** : Gestion de l'historique et de l'archivage

**Évolution des patterns :**
- **Drift conceptuel** : Changement des schémas de fraude dans le temps
- **Adaptation des fraudeurs** : Modification des techniques en fonction des contrôles
- **Nouveaux produits** : Émergence de catégories non couvertes
- **Réglementation** : Évolution des règles douanières

#### 1.2.3 Besoins d'Explicabilité et de Transparence

**Exigences réglementaires :**
- **Traçabilité** : Nécessité de justifier les décisions de contrôle
- **Appel** : Possibilité de contester les décisions automatisées
- **Audit** : Vérification des processus de décision
- **Conformité** : Respect des réglementations sur l'IA

**Besoins opérationnels :**
- **Formation** : Compréhension des décisions par les utilisateurs
- **Amélioration** : Identification des points d'optimisation
- **Confiance** : Adoption par les utilisateurs finaux
- **Maintenance** : Diagnostic des dysfonctionnements

### 1.3 Objectifs du Mémoire et Contributions

#### 1.3.1 Objectif Principal

**Concevoir et réaliser un système intelligent de détection de fraude douanière** combinant apprentissage automatique, apprentissage par renforcement et explicabilité, capable d'analyser automatiquement les déclarations douanières et de fournir des recommandations de contrôle justifiées et adaptatives.

#### 1.3.2 Objectifs Spécifiques

**1. Développement de modèles ML performants :**
- Atteindre un F1-Score supérieur à 95% sur les chapitres tarifaires principaux
- Optimiser les modèles pour les spécificités de la fraude douanière
- Assurer la robustesse et la stabilité des prédictions
- Minimiser les faux positifs et les faux négatifs

**2. Intégration d'un système RL adaptatif :**
- Développer un système d'apprentissage par renforcement pour l'optimisation des seuils
- Permettre l'adaptation continue aux changements de patterns
- Optimiser le compromis entre précision et rappel
- Assurer la convergence et la stabilité de l'algorithme

**3. Implémentation de l'explicabilité SHAP :**
- Intégrer SHAP pour l'interprétation des décisions ML
- Fournir des explications claires et actionables
- Assurer la cohérence et la fiabilité des explications
- Optimiser les temps de calcul pour l'explicabilité

**4. Création d'une architecture complète :**
- Développer un backend robuste avec FastAPI et PostgreSQL
- Créer une interface utilisateur intuitive avec Flutter
- Assurer la scalabilité et la performance du système
- Intégrer un système de monitoring et d'alertes

**5. Validation sur données réelles :**
- Tester le système sur des données authentiques de la DGD
- Valider les performances avec des experts du domaine
- Mesurer l'impact opérationnel et l'acceptabilité utilisateur
- Assurer la reproductibilité et la généralisation

#### 1.3.3 Contributions Scientifiques

**Innovation méthodologique :**
- **Première combinaison ML/RL/SHAP** pour la fraude douanière
- **Pipeline end-to-end** depuis l'OCR jusqu'à la décision explicable
- **Architecture modulaire** et extensible pour différents contextes
- **Méthodologie reproductible** pour d'autres domaines d'application

**Contributions techniques :**
- **Optimisation des modèles** pour les spécificités douanières
- **Système RL adaptatif** pour l'optimisation continue
- **Intégration native SHAP** pour la transparence
- **Architecture cloud-native** pour la scalabilité

**Contributions pratiques :**
- **Solution opérationnelle** déployable en production
- **Interface multi-rôles** adaptée aux besoins métier
- **Validation utilisateur** avec des experts du domaine
- **Impact mesurable** sur l'efficacité opérationnelle

### 1.4 Intérêts du Sujet et Impact Attendu

#### 1.4.1 Intérêt Scientifique

**Avancées en IA appliquée :**
- **Innovation algorithmique** : Combinaison inédite de techniques avancées
- **Explicabilité** : Contribution aux méthodes d'interprétation en IA
- **Apprentissage par renforcement** : Application à des domaines critiques
- **Transfer learning** : Adaptation de modèles génériques au domaine douanier

**Méthodologie reproductible :**
- **Framework générique** : Applicable à d'autres domaines de fraude
- **Benchmarks** : Métriques de référence pour la recherche future
- **Open source** : Contribution à la communauté scientifique
- **Documentation** : Guide méthodologique complet

#### 1.4.2 Intérêt Pratique

**Pour la Direction Générale des Douanes :**
- **Amélioration de l'efficacité** : Réduction des contrôles manuels
- **Optimisation des ressources** : Ciblage des contrôles sur les cas à risque
- **Transparence** : Justification claire des décisions de contrôle
- **Adaptabilité** : Évolution continue avec les nouveaux schémas de fraude

**Pour les opérateurs économiques :**
- **Réduction des délais** : Traitement plus rapide des déclarations conformes
- **Prévisibilité** : Compréhension des critères de contrôle
- **Équité** : Application uniforme des règles
- **Modernisation** : Intégration dans l'écosystème numérique

**Pour l'économie nationale :**
- **Augmentation des recettes** : Meilleure collecte des droits et taxes
- **Amélioration de la compétitivité** : Lutte contre la concurrence déloyale
- **Sécurité** : Protection contre les produits dangereux
- **Réputation** : Amélioration de l'image du pays

#### 1.4.3 Impact Économique Attendu

**Gains directs :**
- **Recettes fiscales** : Augmentation estimée de 15-20% des recettes douanières
- **Efficacité opérationnelle** : Réduction de 50% du temps d'analyse
- **Coûts de contrôle** : Diminution de 30% des coûts opérationnels
- **Ressources humaines** : Réaffectation vers des tâches à valeur ajoutée

**Gains indirects :**
- **Climat des affaires** : Amélioration de la compétitivité
- **Investissements** : Attractivité accrue pour les investisseurs
- **Innovation** : Incitation à d'autres projets technologiques
- **Formation** : Développement des compétences numériques

#### 1.4.4 Impact Social et Sociétal

**Formation et compétences :**
- **Développement des compétences** : Formation des agents aux nouvelles technologies
- **Création d'emplois** : Nouveaux postes dans le domaine de l'IA
- **Transfert de technologie** : Acquisition de savoir-faire local
- **Innovation** : Culture de l'innovation dans l'administration

**Transparence et confiance :**
- **Gouvernance** : Amélioration de la transparence administrative
- **Confiance citoyenne** : Renforcement de la confiance dans les institutions
- **Équité** : Application uniforme des règles pour tous
- **Responsabilité** : Traçabilité et auditabilité des décisions

### 1.5 Structure du Mémoire

Ce mémoire est organisé en sept chapitres principaux :

**Chapitre 1** (présent) : Introduction générale présentant le contexte, la problématique, les objectifs et l'intérêt du sujet.

**Chapitre 2** : État de l'art et solutions existantes, incluant une analyse comparative des approches actuelles et le positionnement de notre solution.

**Chapitre 3** : Analyse et conception du système, détaillant l'architecture fonctionnelle, la modélisation des données et les choix technologiques.

**Chapitre 4** : Implémentation et développement, présentant le pipeline de traitement, les modèles ML/RL, l'intégration SHAP et le développement des applications.

**Chapitre 5** : Expérimentation et évaluation, détaillant les protocoles d'évaluation, les résultats obtenus et la validation utilisateur.

**Chapitre 6** : Déploiement et perspectives, présentant l'architecture de déploiement, l'interface utilisateur et les perspectives d'amélioration.

**Chapitre 7** : Conclusions et perspectives, synthétisant les contributions, identifiant les limites et proposant des directions futures.

Cette structure permet une progression logique depuis la problématique jusqu'à la solution opérationnelle, en passant par la conception, l'implémentation et l'évaluation, offrant ainsi une vision complète du projet INSPECT_IA.

---

**Fin du Chapitre 1**