# 🚀 INSPECT_IA - Système Intelligent de Détection de Fraude Douanière

## 📋 Description

INSPECT_IA est un système complet de détection de fraude douanière utilisant l'intelligence artificielle et l'apprentissage automatique. Le système analyse les déclarations douanières et détecte automatiquement les fraudes potentielles avec des explications SHAP détaillées.

## 🏗️ Architecture

### Backend (FastAPI + Python)
- **API REST** : Endpoints pour prédiction, analyse et gestion des déclarations
- **Modèles ML** : XGBoost, CatBoost, LightGBM pour les chapitres 30, 84, 85
- **Base de données** : PostgreSQL pour la persistance des données
- **SHAP** : Explications des prédictions avec importance des features
- **OCR** : Traitement automatique des documents scannés

### Frontend (Flutter)
- **Interface multi-rôles** : Inspecteur, Expert ML, Chef de Service
- **Dashboard ML** : Monitoring des performances et recommandations
- **Analytics** : Visualisation des tendances et patterns de fraude
- **Upload** : Interface d'upload et analyse des déclarations

## 🎯 Fonctionnalités Principales

### 🔍 Détection de Fraude
- **Analyse automatique** des déclarations douanières
- **Probabilités de fraude** avec seuils adaptatifs
- **Explications SHAP** pour comprendre les décisions
- **Support multi-chapitres** (30, 84, 85)

### 📊 Dashboard ML
- **Performances en temps réel** des modèles
- **Détection de drift** basée sur les données PostgreSQL
- **Recommandations intelligentes** pour l'entraînement
- **Statistiques dynamiques** avec simulation temporelle

### 👥 Gestion des Rôles
- **Inspecteur** : Upload, analyse, génération de PV
- **Expert ML** : Dashboard, analytics, configuration des modèles
- **Chef de Service** : Vue d'ensemble et supervision

### 🔄 Persistance Temps Réel
- **Stockage PostgreSQL** de toutes les prédictions
- **Synchronisation** frontend/backend en temps réel
- **Historique complet** des analyses

## 🚀 Installation et Démarrage

### Prérequis
- Python 3.8+
- Flutter 3.0+
- PostgreSQL 12+
- Docker (optionnel)

### Backend
```bash
cd inspectia_app/backend
pip install -r requirements.txt
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd inspectia_app/inspectia_app_frontend
flutter pub get
flutter run -d chrome --debug
```

### Base de Données
```bash
# Configuration PostgreSQL
createdb inspect_ia
# Les tables sont créées automatiquement au premier démarrage
```

## 📁 Structure du Projet

```
INSP/
├── inspectia_app/
│   ├── backend/                 # API FastAPI
│   │   ├── api/                # Endpoints REST
│   │   ├── src/                # Logique métier
│   │   │   ├── chapters/       # Modèles par chapitre
│   │   │   ├── shared/         # Composants partagés
│   │   │   └── utils/          # Utilitaires
│   │   └── config/             # Configuration
│   └── inspectia_app_frontend/ # Application Flutter
│       ├── lib/
│       │   ├── screens/        # Écrans de l'application
│       │   ├── services/       # Services API
│       │   ├── models/         # Modèles de données
│       │   └── utils/          # Utilitaires
│       └── assets/             # Ressources
├── docs/                       # Documentation
└── scripts/                    # Scripts utilitaires
```

## 🔧 Configuration

### Variables d'Environnement
```bash
# Backend
DATABASE_URL=postgresql://user:password@localhost:5432/inspect_ia
ML_MODELS_PATH=/path/to/models
API_HOST=0.0.0.0
API_PORT=8000

# Frontend
API_BASE_URL=http://localhost:8000
```

### Modèles ML
Les modèles sont automatiquement téléchargés et entraînés au premier démarrage. Les performances sont optimisées pour chaque chapitre :
- **Chapitre 30** : CatBoost (F1: 0.9831, AUC: 0.9997)
- **Chapitre 84** : XGBoost (F1: 0.9887, AUC: 0.9997)
- **Chapitre 85** : XGBoost (F1: 0.9808, AUC: 0.9993)

## 📊 API Endpoints

### Prédiction
- `POST /api/v2/predict/{chapter}` - Prédiction de fraude
- `GET /api/v2/declarations/{chapter}` - Liste des déclarations
- `GET /api/v2/declarations/{chapter}/{id}` - Détails d'une déclaration

### Dashboard ML
- `GET /api/v2/ml-dashboard` - Dashboard complet
- `GET /api/v2/ml-performance` - Performances des modèles
- `GET /api/v2/ml-drift` - Détection de drift
- `GET /api/v2/ml-alerts` - Alertes et recommandations

### Analytics
- `GET /api/v2/analytics/fraud` - Analytics de fraude
- `GET /api/v2/analytics/trends` - Tendances temporelles
- `GET /api/v2/analytics/patterns` - Patterns de fraude

## 🎨 Interface Utilisateur

### Rôles et Permissions
- **Inspecteur** : Accès aux fonctionnalités de base (upload, analyse, PV)
- **Expert ML** : Accès complet + dashboard ML et analytics
- **Chef de Service** : Vue d'ensemble et supervision

### Pages Principales
- **Home** : Tableau de bord principal
- **Upload** : Upload et analyse des déclarations
- **Analytics** : Visualisation des tendances
- **ML Dashboard** : Monitoring des modèles (Expert ML)
- **PV** : Génération de procès-verbaux

## 🔍 Fonctionnalités Avancées

### SHAP (SHapley Additive exPlanations)
- **Explications détaillées** des prédictions
- **Importance des features** pour chaque décision
- **Visualisation interactive** des contributions

### Détection de Drift
- **Monitoring continu** des performances
- **Alertes automatiques** en cas de dégradation
- **Recommandations** d'entraînement

### Persistance Temps Réel
- **Sauvegarde automatique** de toutes les prédictions
- **Synchronisation** entre frontend et backend
- **Historique complet** des analyses

## 🧪 Tests

### Tests Backend
```bash
cd inspectia_app/backend
python -m pytest tests/ -v
```

### Tests Frontend
```bash
cd inspectia_app/inspectia_app_frontend
flutter test
```

### Tests d'Intégration
```bash
# Tests complets du système
python test_all_endpoints_comprehensive.py
python test_coherence_frontend_backend.sh
```

## 📈 Performance

### Métriques de Performance
- **Temps de réponse** : < 2s pour la prédiction
- **Précision** : > 98% sur les données de test
- **Débit** : 100+ déclarations/minute
- **Disponibilité** : 99.9% uptime

### Optimisations
- **Cache Redis** pour les prédictions fréquentes
- **Parallélisation** des calculs ML
- **Compression** des réponses API
- **Lazy loading** dans le frontend

## 🚀 Déploiement

### Docker
```bash
# Démarrage complet avec Docker Compose
docker-compose up -d
```

### Production
```bash
# Backend en production
gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Frontend en production
flutter build web --release
```

## 📚 Documentation

- [Guide d'utilisation rapide](GUIDE_UTILISATION_RAPIDE.md)
- [Corrections appliquées](CORRECTIONS_APPLIQUEES.md)
- [Cohérence frontend/backend](COHERENCE_FRONTEND_BACKEND.md)
- [Synthèse finale](SYNTHESE_FINALE_COHERENCE.md)

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👥 Équipe

- **Développement** : Équipe INSPECT_IA
- **ML/AI** : Experts en Machine Learning
- **DevOps** : Infrastructure et déploiement

## 📞 Support

Pour toute question ou problème :
- **Issues GitHub** : [Créer une issue](https://github.com/MARAMATA/INSPECT_IA/issues)
- **Email** : support@inspect-ia.com
- **Documentation** : [Wiki du projet](https://github.com/MARAMATA/INSPECT_IA/wiki)

---

## 🎯 Roadmap

### Version 2.0
- [ ] Support de nouveaux chapitres douaniers
- [ ] Interface mobile native
- [ ] Intégration avec systèmes douaniers existants
- [ ] API GraphQL
- [ ] Monitoring avancé avec Prometheus/Grafana

### Version 2.1
- [ ] Apprentissage fédéré
- [ ] Détection de fraude en temps réel
- [ ] Interface de configuration avancée
- [ ] Support multi-langues
- [ ] Intégration blockchain pour l'audit

---

**🚀 INSPECT_IA - L'avenir de la détection de fraude douanière avec l'IA !**