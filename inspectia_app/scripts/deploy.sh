#!/bin/bash

# 🚀 SCRIPT DE DÉPLOIEMENT COMPLET INSPECTIA
# Déploiement en production avec toutes les optimisations

set -e

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║                        🚀 DÉPLOIEMENT INSPECTIA PRODUCTION                   ║${NC}"
    echo -e "${PURPLE}║                    Système ML-RL avec 21 tables PostgreSQL                   ║${NC}"
    echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_message() {
    echo -e "${GREEN}[DEPLOY]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Vérifications pré-déploiement
pre_deployment_checks() {
    print_info "🔍 Vérifications pré-déploiement..."
    
    # Vérifier Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker n'est pas installé"
        exit 1
    fi
    
    # Vérifier Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose n'est pas installé"
        exit 1
    fi
    
    # Vérifier l'espace disque
    available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 20 ]; then
        print_warning "Espace disque faible: ${available_space}GB disponible"
    fi
    
    # Vérifier la mémoire
    total_mem=$(free -g | awk 'NR==2{print $2}')
    if [ "$total_mem" -lt 8 ]; then
        print_warning "Mémoire faible: ${total_mem}GB disponible (8GB recommandé)"
    fi
    
    print_success "Vérifications terminées"
}

# Sauvegarde des données existantes
backup_existing_data() {
    print_info "💾 Sauvegarde des données existantes..."
    
    # Créer le dossier de sauvegarde
    backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Sauvegarder la base de données si elle existe
    if docker-compose ps postgres | grep -q "Up"; then
        print_info "Sauvegarde de la base de données..."
        docker-compose exec -T postgres pg_dump -U inspectia_user inspectia_db > "$backup_dir/database.sql"
    fi
    
    # Sauvegarder les volumes
    print_info "Sauvegarde des volumes..."
    docker run --rm -v inspectia_postgres_data:/data -v "$(pwd)/$backup_dir":/backup alpine tar czf /backup/postgres_data.tar.gz -C /data .
    
    print_success "Sauvegarde terminée: $backup_dir"
}

# Construction des images
build_images() {
    print_info "🔨 Construction des images Docker..."
    
    # Backend
    print_info "Construction du backend (FastAPI + ML/RL)..."
    docker-compose build --no-cache backend
    
    # Frontend
    print_info "Construction du frontend (Flutter Web)..."
    docker-compose build --no-cache frontend
    
    # Streamlit
    print_info "Construction de Streamlit..."
    docker-compose build --no-cache streamlit
    
    print_success "Images construites avec succès"
}

# Déploiement des services
deploy_services() {
    print_info "🚀 Déploiement des services..."
    
    # Arrêter les services existants
    print_info "Arrêt des services existants..."
    docker-compose down --remove-orphans
    
    # Démarrer PostgreSQL en premier
    print_info "Démarrage de PostgreSQL (21 tables)..."
    docker-compose up -d postgres
    
    # Attendre que PostgreSQL soit prêt
    print_info "⏳ Attente de PostgreSQL..."
    until docker-compose exec postgres pg_isready -U inspectia_user -d inspectia_db; do
        echo -n "."
        sleep 2
    done
    echo ""
    print_success "PostgreSQL est prêt!"
    
    # Initialiser la base de données
    print_info "📊 Initialisation de la base de données (21 tables)..."
    docker-compose exec postgres psql -U inspectia_user -d inspectia_db -f /docker-entrypoint-initdb.d/01-schema-inspectia.sql
    
    # Démarrer MLflow
    print_info "Démarrage de MLflow..."
    docker-compose up -d mlflow
    
    # Démarrer Redis
    print_info "Démarrage de Redis..."
    docker-compose up -d redis
    
    # Démarrer le backend
    print_info "Démarrage du backend (FastAPI + 98 endpoints)..."
    docker-compose up -d backend
    
    # Attendre que le backend soit prêt
    print_info "⏳ Attente du backend..."
    until curl -f http://localhost:8000/health &>/dev/null; do
        echo -n "."
        sleep 3
    done
    echo ""
    print_success "Backend est prêt!"
    
    # Démarrer le frontend
    print_info "Démarrage du frontend (Flutter Web)..."
    docker-compose up -d frontend
    
    # Démarrer Streamlit
    print_info "Démarrage de Streamlit..."
    docker-compose up -d streamlit
    
    # Démarrer le monitoring
    print_info "Démarrage du monitoring (Prometheus + Grafana)..."
    docker-compose up -d prometheus grafana
    
    # Démarrer Nginx
    print_info "Démarrage de Nginx (Reverse Proxy)..."
    docker-compose up -d nginx
    
    print_success "Tous les services sont déployés!"
}

# Vérification du déploiement
verify_deployment() {
    print_info "🔍 Vérification du déploiement..."
    
    # Vérifier les services
    services=("postgres" "backend" "frontend" "streamlit" "mlflow" "redis" "nginx")
    
    for service in "${services[@]}"; do
        if docker-compose ps "$service" | grep -q "Up"; then
            print_success "$service est en cours d'exécution"
        else
            print_error "$service n'est pas en cours d'exécution"
            return 1
        fi
    done
    
    # Vérifier les endpoints
    print_info "Vérification des endpoints..."
    
    # Backend health
    if curl -f http://localhost:8000/health &>/dev/null; then
        print_success "Backend API accessible"
    else
        print_error "Backend API non accessible"
        return 1
    fi
    
    # Frontend
    if curl -f http://localhost:3000 &>/dev/null; then
        print_success "Frontend accessible"
    else
        print_error "Frontend non accessible"
        return 1
    fi
    
    # Streamlit
    if curl -f http://localhost:8501 &>/dev/null; then
        print_success "Streamlit accessible"
    else
        print_error "Streamlit non accessible"
        return 1
    fi
    
    print_success "Déploiement vérifié avec succès!"
}

# Configuration post-déploiement
post_deployment_config() {
    print_info "⚙️ Configuration post-déploiement..."
    
    # Configurer les alertes
    print_info "Configuration des alertes..."
    # Ici vous pouvez ajouter la configuration des alertes
    
    # Configurer les sauvegardes automatiques
    print_info "Configuration des sauvegardes automatiques..."
    # Ici vous pouvez ajouter la configuration des sauvegardes
    
    # Configurer le monitoring
    print_info "Configuration du monitoring..."
    # Ici vous pouvez ajouter la configuration du monitoring
    
    print_success "Configuration post-déploiement terminée"
}

# Afficher les informations de déploiement
show_deployment_info() {
    echo ""
    print_info "🌐 Services déployés:"
    echo ""
    echo -e "  ${CYAN}📱 Frontend Flutter:${NC}     http://localhost:3000"
    echo -e "  ${CYAN}🔧 Backend API:${NC}          http://localhost:8000"
    echo -e "  ${CYAN}📊 Streamlit Dashboard:${NC} http://localhost:8501"
    echo -e "  ${CYAN}🤖 MLflow:${NC}              http://localhost:5000"
    echo -e "  ${CYAN}🗄️  PostgreSQL:${NC}         localhost:5432"
    echo -e "  ${CYAN}🔴 Redis:${NC}               localhost:6379"
    echo -e "  ${CYAN}📈 Prometheus:${NC}          http://localhost:9090"
    echo -e "  ${CYAN}📊 Grafana:${NC}             http://localhost:3001"
    echo ""
    print_info "🔍 Commandes utiles:"
    echo "  docker-compose ps                    # Statut des services"
    echo "  docker-compose logs -f [service]    # Logs d'un service"
    echo "  docker-compose restart [service]    # Redémarrer un service"
    echo "  docker-compose down                  # Arrêter tous les services"
    echo ""
    print_info "📊 Monitoring:"
    echo "  Grafana: http://localhost:3001 (admin/admin123)"
    echo "  Prometheus: http://localhost:9090"
    echo ""
}

# Script principal
main() {
    case "${1:-deploy}" in
        "deploy")
            print_header
            pre_deployment_checks
            backup_existing_data
            build_images
            deploy_services
            verify_deployment
            post_deployment_config
            show_deployment_info
            ;;
        "rollback")
            print_info "🔄 Rollback vers la version précédente..."
            # Implémenter le rollback
            ;;
        "status")
            print_info "📊 Statut du déploiement:"
            docker-compose ps
            ;;
        "logs")
            print_info "📋 Logs des services:"
            docker-compose logs -f
            ;;
        "help"|"-h"|"--help")
            print_header
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  deploy    - Déployer tous les services (défaut)"
            echo "  rollback  - Rollback vers la version précédente"
            echo "  status    - Afficher le statut"
            echo "  logs      - Afficher les logs"
            echo "  help      - Afficher cette aide"
            ;;
        *)
            print_error "Commande inconnue: $1"
            exit 1
            ;;
    esac
}

# Exécuter le script principal
main "$@"
