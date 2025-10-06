# 🐳 DOCKERISATION COMPLÈTE INSPECTIA

## 🎯 Vue d'ensemble

Dockerisation complète du système INSPECTIA avec **21 tables PostgreSQL**, **98+ endpoints API**, **système ML-RL hybride**, et **frontend Flutter multi-plateforme**.

## 📊 Architecture Docker

### **Services Inclus (8 services)**

| Service | Port | Description | Tables |
|---------|------|-------------|--------|
| **PostgreSQL** | 5432 | Base de données (21 tables) | 21 tables |
| **Backend API** | 8000 | FastAPI + ML/RL (98 endpoints) | - |
| **Frontend** | 3000 | Flutter Web | - |
| **Streamlit** | 8501 | Dashboard d'analyse | - |
| **MLflow** | 5000 | Tracking des modèles ML | - |
| **Redis** | 6379 | Cache et sessions | - |
| **Nginx** | 80/443 | Reverse Proxy | - |
| **Monitoring** | 9090/3001 | Prometheus/Grafana | - |

## 🗄️ Base de Données (21 Tables)

### **Tables de Configuration (4)**
- `chapters` - Chapitres douaniers (30, 84, 85)
- `models` - Modèles ML entraînés
- `features` - Features utilisées par les modèles
- `chapter_features` - Associations chapitres-features

### **Tables de Données (3)**
- `declarations` - Déclarations douanières
- `predictions` - Prédictions ML
- `declaration_features` - Features extraites par déclaration

### **Tables Système RL (3)**
- `rl_decisions` - Décisions du système RL
- `inspector_profiles` - Profils d'inspecteurs
- `feedback_history` - Historique des feedbacks

### **Tables d'Analyse (4)**
- `analysis_results` - Résultats d'analyse détaillée
- `model_thresholds` - Seuils et configurations
- `performance_metrics` - Métriques de performance
- `system_logs` - Logs système

### **Tables Avancées (7)**
- `advanced_decisions` - Décisions avancées RL
- `advanced_feedbacks` - Feedbacks avancés
- `advanced_policies` - Politiques avancées
- `pv_inspection` - Procès-verbaux d'inspection
- `pvs` - Procès-verbaux
- `rl_bandits` - Bandits RL
- `rl_performance_metrics` - Métriques RL

## 🚀 Démarrage Rapide

### **Option 1: Script de démarrage (Recommandé)**

```bash
# Démarrer tous les services
./docker-start.sh start

# Démarrer avec monitoring
./docker-start.sh start --with-monitoring

# Voir les logs
./docker-start.sh logs

# Arrêter les services
./docker-start.sh stop
```

### **Option 2: Docker Compose manuel**

```bash
# Démarrer tous les services
docker-compose up -d

# Démarrer avec monitoring
docker-compose --profile monitoring up -d

# Voir les logs
docker-compose logs -f
```

## 🔧 Configuration des Services

### **PostgreSQL (21 tables)**
```yaml
environment:
  POSTGRES_DB: inspectia_db
  POSTGRES_USER: inspectia_user
  POSTGRES_PASSWORD: inspectia_pass
volumes:
  - postgres_data:/var/lib/postgresql/data
  - ./backend/database/schema_INSPECT_IA.sql:/docker-entrypoint-initdb.d/
```

### **Backend API (98 endpoints)**
```yaml
environment:
  - DATABASE_URL=postgresql+asyncpg://inspectia_user:inspectia_pass@postgres:5432/inspectia_db
  - MLFLOW_TRACKING_URI=http://mlflow:5000
volumes:
  - backend_logs:/app/logs
  - backend_results:/app/results
  - backend_models:/app/models
```

### **Frontend Flutter**
```yaml
environment:
  - BACKEND_URL=http://backend:8000
  - API_BASE_URL=http://backend:8000
```

## 📋 Endpoints API (98 endpoints)

### **Router Principal (/predict) - 84 endpoints**
- Prédiction et analyse (13 endpoints)
- Configuration et informations (11 endpoints)
- Système RL (15 endpoints)
- Retraining ML (3 endpoints)
- Feedback et PV (4 endpoints)
- Tests et seuils (4 endpoints)
- Statistiques avancées (4 endpoints)
- Features business (6 endpoints)
- Cache et système (2 endpoints)
- OCR et ingestion (9 endpoints)
- Tests et debug (6 endpoints)
- Features sélectionnées (1 endpoint)
- Fonctions utilitaires (7 endpoints)

### **ML Router (/ml) - 7 endpoints**
- Test ML
- Dashboard de performance
- Détection de drift
- Alertes ML
- Dashboard ML
- Dashboard Chef
- Retraining par chapitre

### **PostgreSQL Router (/api/v2) - 7 endpoints**
- Statut système
- Santé base de données
- Test simple
- Upload déclarations
- Liste déclarations

## 🎯 Chapitres Supportés

### **Chapitre 30 - Pharmaceutique**
- **Modèle**: XGBoost
- **Performance**: F1=0.9821, AUC=0.9997
- **Features**: 22 features (10 business)
- **Seuil optimal**: 0.55
- **Données**: 25,334 échantillons

### **Chapitre 84 - Mécanique**
- **Modèle**: XGBoost
- **Performance**: F1=0.9891, AUC=0.9997
- **Features**: 21 features (9 business)
- **Seuil optimal**: 0.42
- **Données**: 264,494 échantillons

### **Chapitre 85 - Électrique**
- **Modèle**: XGBoost
- **Performance**: F1=0.9781, AUC=0.9993
- **Features**: 23 features (11 business)
- **Seuil optimal**: 0.51
- **Données**: 197,402 échantillons

## 🔍 Monitoring et Observabilité

### **Prometheus**
- Métriques système
- Métriques ML/RL
- Métriques de performance
- Alertes automatiques

### **Grafana**
- Dashboards temps réel
- Métriques des modèles
- Performance du système
- Analytics avancés

## 🛠️ Développement

### **Volumes de Développement**
```yaml
volumes:
  - ./backend:/app:ro  # Code source en lecture seule
  - backend_logs:/app/logs
  - backend_results:/app/results
  - backend_models:/app/models
```

### **Hot Reload**
```bash
# Backend avec rechargement automatique
docker-compose up backend

# Frontend avec rechargement automatique
docker-compose up frontend
```

## 📊 Performance et Optimisation

### **Ressources Recommandées**
- **RAM**: 8GB minimum (16GB recommandé)
- **CPU**: 4 cœurs minimum
- **Stockage**: 50GB SSD
- **Réseau**: Connexion stable

### **Optimisations Incluses**
- Cache Redis pour les modèles
- Index PostgreSQL optimisés
- Compression des volumes
- Health checks automatiques

## 🔒 Sécurité

### **Configuration Production**
- Variables d'environnement sécurisées
- Certificats SSL/TLS
- Authentification des services
- Audit des accès

### **Isolation des Services**
- Réseau Docker privé
- Volumes isolés
- Contrôle d'accès par service

## 📈 Scaling et Haute Disponibilité

### **Scaling Horizontal**
```bash
# Scaling du backend
docker-compose up --scale backend=3

# Scaling du frontend
docker-compose up --scale frontend=2
```

### **Load Balancing**
- Nginx comme reverse proxy
- Distribution des requêtes
- Health checks automatiques

## 🆘 Dépannage

### **Problèmes Courants**

#### 1. Port déjà utilisé
```bash
# Vérifier les ports
lsof -i :8000
lsof -i :3000

# Arrêter les services
./docker-start.sh stop
```

#### 2. Base de données non accessible
```bash
# Vérifier les logs
docker-compose logs postgres

# Redémarrer la DB
docker-compose restart postgres
```

#### 3. Problème de mémoire
```bash
# Vérifier l'utilisation
docker stats

# Augmenter la mémoire Docker
# Docker Desktop > Settings > Resources
```

### **Nettoyage Complet**
```bash
# Nettoyage complet
./docker-start.sh clean-all

# Ou manuellement
docker-compose down --rmi all --volumes --remove-orphans
docker system prune -f
```

## 📚 Commandes Utiles

### **Gestion des Services**
```bash
# Démarrer
./docker-start.sh start

# Arrêter
./docker-start.sh stop

# Redémarrer
./docker-start.sh restart

# Logs
./docker-start.sh logs

# Statut
./docker-start.sh status
```

### **Base de Données**
```bash
# Connexion à PostgreSQL
docker-compose exec postgres psql -U inspectia_user -d inspectia_db

# Sauvegarder la base
docker-compose exec postgres pg_dump -U inspectia_user inspectia_db > backup.sql

# Restaurer la base
docker-compose exec -T postgres psql -U inspectia_user inspectia_db < backup.sql
```

### **Développement**
```bash
# Entrer dans un conteneur
docker-compose exec backend bash
docker-compose exec frontend sh

# Copier des fichiers
docker cp local_file.txt inspectia_backend:/app/

# Voir les logs d'un service
docker-compose logs -f backend
```

## 🎯 Avantages de cette Dockerisation

1. **Complète** - Tous les services inclus
2. **Optimisée** - Performance et ressources
3. **Sécurisée** - Isolation et contrôle d'accès
4. **Scalable** - Haute disponibilité
5. **Maintenable** - Scripts automatisés
6. **Documentée** - Guide complet
7. **Testée** - Validation des composants

---

**INSPECTIA** - Système de détection de fraude douanière avec IA 🚀
