# 🐳 InspectIA - Configuration Docker

Ce document explique comment utiliser la configuration Docker pour InspectIA.

## 📋 Prérequis

- Docker Desktop installé et en cours d'exécution
- Au moins 4GB de RAM disponible pour Docker
- Ports libres : 3000, 8000, 8501, 5000, 5432

## 🚀 Démarrage Rapide

### Option 1: Script de démarrage (Recommandé)

```bash
# Démarrer en mode développement
./start.sh dev

# Démarrer en mode production
./start.sh prod

# Voir les logs
./start.sh logs

# Arrêter les services
./start.sh stop

# Nettoyer Docker
./start.sh clean
```

### Option 2: Docker Compose manuel

```bash
cd backend
docker-compose up --build -d
```

## 🏗️ Architecture des Services

### Services Inclus

| Service | Port | Description |
|---------|------|-------------|
| **Frontend Flutter** | 3000 | Interface utilisateur web |
| **Backend API** | 8000 | API FastAPI avec ML/RL |
| **Streamlit** | 8501 | Dashboard d'analyse |
| **MLflow** | 5000 | Suivi des modèles ML |
| **PostgreSQL** | 5432 | Base de données |

### Volumes Docker

- `db_data`: Données PostgreSQL persistantes
- `mlruns`: Artifacts MLflow
- `./logs`: Logs de l'application
- `./results`: Résultats des analyses
- `./data`: Données d'entraînement
- `./models`: Modèles ML sauvegardés

## 🔧 Configuration

### Variables d'Environnement

#### Backend API
```env
PORT=8000
DB_HOST=db
DB_PORT=5432
DB_USER=inspectia_user
DB_PASSWORD=inspectia_pass
DB_NAME=inspectia_db
DATABASE_URL=postgresql+asyncpg://inspectia_user:inspectia_pass@db:5432/inspectia_db
MLFLOW_TRACKING_URI=http://mlflow:5000
```

#### Frontend
```env
BACKEND_URL=http://api:8000
```

### Health Checks

Tous les services incluent des health checks automatiques :
- **PostgreSQL**: Vérification de la connexion
- **Backend API**: Endpoint `/health`
- **Frontend**: Vérification HTTP
- **MLflow**: Vérification du serveur

## 📊 Monitoring

### Vérifier le statut des services

```bash
# Via le script
./start.sh status

# Via Docker Compose
cd backend && docker-compose ps
```

### Logs en temps réel

```bash
# Tous les services
./start.sh logs

# Service spécifique
cd backend && docker-compose logs -f api
```

## 🛠️ Développement

### Mode Développement

Le mode développement monte les volumes locaux pour permettre :
- Modification du code en temps réel
- Accès aux logs et résultats
- Debugging facilité

### Rebuild des images

```bash
# Rebuild complet
docker-compose up --build --force-recreate

# Rebuild d'un service spécifique
docker-compose up --build api
```

## 🔍 Dépannage

### Problèmes Courants

#### 1. Port déjà utilisé
```bash
# Vérifier les ports utilisés
lsof -i :8000
lsof -i :3000

# Arrêter les services
./start.sh stop
```

#### 2. Erreur de permissions Docker
```bash
# Redémarrer Docker Desktop
# Ou ajouter l'utilisateur au groupe docker
sudo usermod -aG docker $USER
```

#### 3. Problème de mémoire
```bash
# Augmenter la mémoire allouée à Docker
# Docker Desktop > Settings > Resources > Memory
```

#### 4. Base de données non accessible
```bash
# Vérifier les logs de la DB
docker-compose logs db

# Redémarrer la DB
docker-compose restart db
```

### Nettoyage Complet

```bash
# Arrêter et supprimer tout
./start.sh clean

# Ou manuellement
docker-compose down --rmi all --volumes --remove-orphans
docker system prune -f
```

## 📈 Performance

### Optimisations Recommandées

1. **Mémoire Docker**: Au moins 4GB
2. **CPU**: Au moins 2 cœurs
3. **Stockage**: SSD recommandé
4. **Réseau**: Connexion stable

### Monitoring des Ressources

```bash
# Utilisation des ressources
docker stats

# Espace disque
docker system df
```

## 🔒 Sécurité

### Configuration Production

Pour la production, modifiez :
1. Mots de passe de la base de données
2. Configuration CORS
3. Certificats SSL/TLS
4. Variables d'environnement sensibles

### Exemple de configuration sécurisée

```yaml
# docker-compose.prod.yml
services:
  db:
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
  api:
    environment:
      - DB_PASSWORD=${DB_PASSWORD}
      - SECRET_KEY=${SECRET_KEY}
```

## 📚 Commandes Utiles

```bash
# Entrer dans un conteneur
docker exec -it inspectia_api bash

# Copier des fichiers
docker cp local_file.txt inspectia_api:/app/

# Sauvegarder la base de données
docker exec inspectia_db pg_dump -U inspectia_user inspectia_db > backup.sql

# Restaurer la base de données
docker exec -i inspectia_db psql -U inspectia_user inspectia_db < backup.sql
```

## 🆘 Support

En cas de problème :
1. Vérifiez les logs : `./start.sh logs`
2. Consultez le statut : `./start.sh status`
3. Redémarrez : `./start.sh stop && ./start.sh dev`
4. Nettoyez : `./start.sh clean`

---

**InspectIA** - Système d'analyse intelligente des déclarations douanières 🚀
