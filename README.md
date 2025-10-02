# INSPECT_IA - Système Intelligent de Détection de Fraude Douanière

## 🎯 Description du Projet

INSPECT_IA est un système avancé d'intelligence artificielle pour la détection de fraudes dans les déclarations douanières. Le système utilise des techniques de Machine Learning, Reinforcement Learning et OCR pour analyser automatiquement les documents et identifier les déclarations suspectes.

## 🚀 Fonctionnalités Principales

### 🤖 Machine Learning Avancé
- **Modèles Supervisés** : Random Forest, XGBoost, LightGBM, SVM
- **Détection de Drift** : Surveillance continue des modèles
- **Retraining Automatique** : Mise à jour des modèles basée sur le feedback
- **Features Engineering** : Extraction automatique de features métier

### 🧠 Reinforcement Learning
- **Stratégies Multi-Armed Bandit** : Epsilon-Greedy, UCB, Thompson Sampling
- **Apprentissage Adaptatif** : Optimisation continue des décisions
- **Profils d'Inspecteurs** : Personnalisation selon l'expertise
- **Feedback Loop** : Amélioration continue basée sur les retours

### 📄 OCR et Traitement de Documents
- **Pipeline OCR Avancé** : Extraction de texte depuis images/PDF
- **Validation Automatique** : Vérification des données extraites
- **Preprocessing Intelligent** : Nettoyage et normalisation des données

### 📊 Analytics et Reporting
- **Dashboard Temps Réel** : Métriques de performance
- **Rapports Détaillés** : Analyses approfondies par chapitre
- **Génération de PV** : Procès-verbaux automatiques
- **Visualisations** : Graphiques et courbes ROC

## 🏗️ Architecture

### Backend (Python)
```
backend/
├── src/
│   ├── shared/           # Modules partagés
│   ├── chapters/         # Logique par chapitre douanier
│   ├── ml/              # Machine Learning
│   ├── rl/              # Reinforcement Learning
│   └── api/             # API REST
├── database/            # Gestion PostgreSQL
├── results/             # Résultats et rapports
└── tests/              # Tests unitaires
```

### Frontend (Flutter)
```
inspectia_app_frontend/
├── lib/
│   ├── screens/         # Écrans de l'application
│   ├── services/        # Services backend
│   ├── utils/           # Utilitaires et constantes
│   ├── widgets/         # Composants réutilisables
│   └── middleware/      # Gestion des routes
├── test/               # Tests frontend
└── assets/             # Ressources
```

## 📋 Chapitres Supportés

- **Chapitre 30** : Produits pharmaceutiques
- **Chapitre 84** : Machines et équipements mécaniques
- **Chapitre 85** : Machines et appareils électriques

## 🛠️ Installation

### Prérequis
- Python 3.8+
- Flutter 3.0+
- PostgreSQL 12+
- Node.js 16+ (pour l'OCR)

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn src.main:app --reload
```

### Frontend
```bash
cd inspectia_app_frontend
flutter pub get
flutter run
```

### Base de Données
```bash
# Créer la base de données
createdb INSPECT_IA

# Exécuter les migrations
python backend/database/migrations.py
```

## 📚 Documentation

### API Endpoints
- `/api/v1/health` - Santé du système
- `/api/v1/chapters` - Chapitres disponibles
- `/api/v1/predict/{chapter}` - Prédiction de fraude
- `/api/v1/upload` - Upload de fichiers
- `/api/v1/rl/performance` - Métriques RL
- `/api/v1/ml/dashboard` - Dashboard ML

### Configuration
Le fichier `constants.dart` centralise tous les endpoints et configurations.

## 🧪 Tests

### Backend
```bash
cd backend
python -m pytest tests/
```

### Frontend
```bash
cd inspectia_app_frontend
flutter test
```

### Tests d'Intégration
```bash
cd inspectia_app_frontend
flutter test test_frontend_backend.dart
```

## 📈 Performance

### Métriques ML
- **Précision** : >95% sur les chapitres testés
- **Recall** : >90% pour la détection de fraude
- **F1-Score** : >92% en moyenne

### Métriques RL
- **Taux d'Exploration** : Adaptatif (5-20%)
- **Temps de Réponse** : <2 secondes
- **Feedback Loop** : <24h pour retraining

## 🔧 Configuration

### Variables d'Environnement
```bash
DATABASE_URL=postgresql://user:pass@localhost/INSPECT_IA
OCR_SERVICE_URL=http://localhost:3000
ML_MODEL_PATH=/path/to/models
```

### Paramètres RL
```python
RL_CONFIG = {
    "epsilon": 0.1,
    "learning_rate": 0.01,
    "exploration_decay": 0.995,
    "min_exploration": 0.05
}
```

## 📊 Monitoring

### Métriques Système
- Santé des services
- Performance des modèles
- Taux d'erreur OCR
- Latence des prédictions

### Alertes
- Drift détecté
- Performance dégradée
- Erreurs critiques
- Retraining requis

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👥 Équipe

- **Développement ML/RL** : Équipe IA
- **Développement Frontend** : Équipe Flutter
- **Développement Backend** : Équipe Python
- **DevOps** : Équipe Infrastructure

## 📞 Support

Pour toute question ou support :
- 📧 Email : support@inspect-ia.com
- 💬 Discord : [Serveur INSPECT_IA]
- 📖 Wiki : [Documentation complète]

## 🔄 Changelog

### Version 2.0.0 (Actuelle)
- ✅ Migration complète vers le nouveau système
- ✅ Intégration RL avancée
- ✅ Dashboard temps réel
- ✅ API REST complète
- ✅ Tests d'intégration

### Version 1.0.0
- ✅ Système de base ML
- ✅ OCR simple
- ✅ Interface Flutter
- ✅ Base de données PostgreSQL

---

**INSPECT_IA** - Intelligence Artificielle pour la Sécurité Douanière 🇫🇷
