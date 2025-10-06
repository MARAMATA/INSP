# =============================================================================
# MAKEFILE INSPECTIA - DOCKERISATION COMPLÈTE
# =============================================================================

.PHONY: help build start stop restart logs status clean clean-all deploy rollback

# Variables
COMPOSE_FILE = docker-compose.yml
PROJECT_NAME = inspectia

# Couleurs
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
RED = \033[0;31m
NC = \033[0m # No Color

# Aide
help:
	@echo "$(BLUE)╔══════════════════════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║                          🚀 INSPECTIA DOCKER COMPLET                        ║$(NC)"
	@echo "$(BLUE)║                    Système ML-RL avec 21 tables PostgreSQL                 ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Commandes disponibles:$(NC)"
	@echo ""
	@echo "  $(YELLOW)build$(NC)        - Construire toutes les images Docker"
	@echo "  $(YELLOW)start$(NC)        - Démarrer tous les services"
	@echo "  $(YELLOW)start-monitoring$(NC) - Démarrer avec monitoring (Prometheus/Grafana)"
	@echo "  $(YELLOW)stop$(NC)         - Arrêter tous les services"
	@echo "  $(YELLOW)restart$(NC)      - Redémarrer tous les services"
	@echo "  $(YELLOW)logs$(NC)         - Afficher les logs de tous les services"
	@echo "  $(YELLOW)logs-backend$(NC) - Afficher les logs du backend"
	@echo "  $(YELLOW)logs-frontend$(NC) - Afficher les logs du frontend"
	@echo "  $(YELLOW)status$(NC)       - Afficher le statut des services"
	@echo "  $(YELLOW)clean$(NC)        - Nettoyer les conteneurs arrêtés"
	@echo "  $(YELLOW)clean-all$(NC)    - Nettoyage complet (supprime tout)"
	@echo "  $(YELLOW)deploy$(NC)       - Déploiement complet en production"
	@echo "  $(YELLOW)rollback$(NC)     - Rollback vers la version précédente"
	@echo "  $(YELLOW)backup$(NC)       - Sauvegarder les données"
	@echo "  $(YELLOW)restore$(NC)      - Restaurer les données"
	@echo "  $(YELLOW)health$(NC)       - Vérifier la santé des services"
	@echo "  $(YELLOW)test$(NC)         - Exécuter les tests"
	@echo ""
	@echo "$(GREEN)Services inclus:$(NC)"
	@echo "  • PostgreSQL (21 tables)"
	@echo "  • Backend FastAPI (98 endpoints)"
	@echo "  • Frontend Flutter (Web)"
	@echo "  • Streamlit Dashboard"
	@echo "  • MLflow (ML tracking)"
	@echo "  • Redis (Cache)"
	@echo "  • Nginx (Reverse Proxy)"
	@echo "  • Prometheus/Grafana (Monitoring)"
	@echo ""

# Construire les images
build:
	@echo "$(BLUE)🔨 Construction des images Docker...$(NC)"
	docker-compose build --no-cache
	@echo "$(GREEN)✅ Images construites avec succès$(NC)"

# Démarrer tous les services
start:
	@echo "$(BLUE)🚀 Démarrage de tous les services...$(NC)"
	./docker-start.sh start
	@echo "$(GREEN)✅ Services démarrés$(NC)"

# Démarrer avec monitoring
start-monitoring:
	@echo "$(BLUE)🚀 Démarrage avec monitoring...$(NC)"
	./docker-start.sh start --with-monitoring
	@echo "$(GREEN)✅ Services démarrés avec monitoring$(NC)"

# Arrêter tous les services
stop:
	@echo "$(BLUE)🛑 Arrêt de tous les services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✅ Services arrêtés$(NC)"

# Redémarrer tous les services
restart:
	@echo "$(BLUE)🔄 Redémarrage de tous les services...$(NC)"
	docker-compose restart
	@echo "$(GREEN)✅ Services redémarrés$(NC)"

# Afficher les logs
logs:
	@echo "$(BLUE)📋 Affichage des logs...$(NC)"
	docker-compose logs -f

# Logs du backend
logs-backend:
	@echo "$(BLUE)📋 Logs du backend...$(NC)"
	docker-compose logs -f backend

# Logs du frontend
logs-frontend:
	@echo "$(BLUE)📋 Logs du frontend...$(NC)"
	docker-compose logs -f frontend

# Statut des services
status:
	@echo "$(BLUE)📊 Statut des services:$(NC)"
	docker-compose ps

# Nettoyer les conteneurs arrêtés
clean:
	@echo "$(BLUE)🧹 Nettoyage des conteneurs arrêtés...$(NC)"
	docker-compose down --remove-orphans
	docker system prune -f
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

# Nettoyage complet
clean-all:
	@echo "$(BLUE)🧹 Nettoyage complet...$(NC)"
	docker-compose down --rmi all --volumes --remove-orphans
	docker system prune -af
	@echo "$(GREEN)✅ Nettoyage complet terminé$(NC)"

# Déploiement complet
deploy:
	@echo "$(BLUE)🚀 Déploiement complet...$(NC)"
	./scripts/deploy.sh deploy
	@echo "$(GREEN)✅ Déploiement terminé$(NC)"

# Rollback
rollback:
	@echo "$(BLUE)🔄 Rollback...$(NC)"
	./scripts/deploy.sh rollback
	@echo "$(GREEN)✅ Rollback terminé$(NC)"

# Sauvegarder les données
backup:
	@echo "$(BLUE)💾 Sauvegarde des données...$(NC)"
	./scripts/backup.sh
	@echo "$(GREEN)✅ Sauvegarde terminée$(NC)"

# Restaurer les données
restore:
	@echo "$(BLUE)📥 Restauration des données...$(NC)"
	./scripts/restore.sh
	@echo "$(GREEN)✅ Restauration terminée$(NC)"

# Vérifier la santé des services
health:
	@echo "$(BLUE)🔍 Vérification de la santé des services...$(NC)"
	@echo "Backend API:"
	@curl -f http://localhost:8000/health && echo " ✅" || echo " ❌"
	@echo "Frontend:"
	@curl -f http://localhost:3000 && echo " ✅" || echo " ❌"
	@echo "Streamlit:"
	@curl -f http://localhost:8501 && echo " ✅" || echo " ❌"
	@echo "MLflow:"
	@curl -f http://localhost:5000 && echo " ✅" || echo " ❌"
	@echo "PostgreSQL:"
	@docker-compose exec postgres pg_isready -U inspectia_user -d inspectia_db && echo " ✅" || echo " ❌"

# Exécuter les tests
test:
	@echo "$(BLUE)🧪 Exécution des tests...$(NC)"
	docker-compose exec backend python -m pytest tests/
	@echo "$(GREEN)✅ Tests terminés$(NC)"

# Entrer dans un conteneur
shell-backend:
	@echo "$(BLUE)🐚 Connexion au backend...$(NC)"
	docker-compose exec backend bash

shell-frontend:
	@echo "$(BLUE)🐚 Connexion au frontend...$(NC)"
	docker-compose exec frontend sh

shell-db:
	@echo "$(BLUE)🐚 Connexion à la base de données...$(NC)"
	docker-compose exec postgres psql -U inspectia_user -d inspectia_db

# Afficher les informations des services
info:
	@echo "$(BLUE)🌐 Services disponibles:$(NC)"
	@echo ""
	@echo "  📱 Frontend Flutter:     http://localhost:3000"
	@echo "  🔧 Backend API:          http://localhost:8000"
	@echo "  📊 Streamlit Dashboard:  http://localhost:8501"
	@echo "  🤖 MLflow:              http://localhost:5000"
	@echo "  🗄️  PostgreSQL:         localhost:5432"
	@echo "  🔴 Redis:               localhost:6379"
	@echo "  📈 Prometheus:          http://localhost:9090"
	@echo "  📊 Grafana:             http://localhost:3001"
	@echo ""

# Mise à jour des services
update:
	@echo "$(BLUE)🔄 Mise à jour des services...$(NC)"
	git pull
	docker-compose pull
	docker-compose up -d
	@echo "$(GREEN)✅ Mise à jour terminée$(NC)"

# Surveillance des ressources
monitor:
	@echo "$(BLUE)📊 Surveillance des ressources...$(NC)"
	docker stats

# Par défaut, afficher l'aide
.DEFAULT_GOAL := help
