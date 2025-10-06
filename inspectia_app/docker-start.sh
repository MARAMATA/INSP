#!/bin/bash

# 🚀 SCRIPT DE DÉMARRAGE COMPLET INSPECTIA
# Dockerisation complète avec 21 tables PostgreSQL

set -e

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
print_header() {
    echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║                          🚀 INSPECTIA DOCKER COMPLET                        ║${NC}"
    echo -e "${PURPLE}║                    Système ML-RL avec 21 tables PostgreSQL                 ║${NC}"
    echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_message() {
    echo -e "${GREEN}[INSPECTIA]${NC} $1"
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

# Vérifier Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker n'est pas installé. Veuillez installer Docker Desktop."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        print_error "Docker n'est pas en cours d'exécution. Veuillez démarrer Docker Desktop."
        exit 1
    fi
}

# Nettoyer les conteneurs existants
cleanup() {
    print_info "🧹 Nettoyage des conteneurs existants..."
    docker-compose down --remove-orphans 2>/dev/null || true
    print_success "Nettoyage terminé"
}

# Construire les images
build_images() {
    print_info "🔨 Construction des images Docker..."
    
    # Backend
    print_info "Construction du backend (FastAPI + ML/RL)..."
    docker-compose build backend
    
    # Frontend
    print_info "Construction du frontend (Flutter Web)..."
    docker-compose build frontend
    
    # Streamlit
    print_info "Construction de Streamlit..."
    docker-compose build streamlit
    
    print_success "Images construites avec succès"
}

# Démarrer les services
start_services() {
    print_info "🚀 Démarrage des services..."
    
    # Démarrer PostgreSQL et MLflow en premier
    print_info "Démarrage de PostgreSQL (21 tables)..."
    docker-compose up -d postgres mlflow
    
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
    
    # Démarrer Redis
    print_info "Démarrage de Redis..."
    docker-compose up -d redis
    
    # Démarrer le monitoring (optionnel)
    if [ "$1" = "--with-monitoring" ]; then
        print_info "Démarrage du monitoring (Prometheus + Grafana)..."
        docker-compose up -d prometheus grafana
    fi
    
    # Démarrer Nginx
    print_info "Démarrage de Nginx (Reverse Proxy)..."
    docker-compose up -d nginx
    
    print_success "Tous les services sont démarrés!"
}

# Afficher les informations des services
show_services_info() {
    echo ""
    print_info "🌐 Services disponibles:"
    echo ""
    echo -e "  ${CYAN}📱 Frontend Flutter:${NC}     http://localhost:3000"
    echo -e "  ${CYAN}🔧 Backend API:${NC}          http://localhost:8000"
    echo -e "  ${CYAN}📊 Streamlit Dashboard:${NC}  http://localhost:8501"
    echo -e "  ${CYAN}🤖 MLflow:${NC}              http://localhost:5000"
    echo -e "  ${CYAN}🗄️  PostgreSQL:${NC}         localhost:5432"
    echo -e "  ${CYAN}🔴 Redis:${NC}               localhost:6379"
    
    if [ "$1" = "--with-monitoring" ]; then
        echo -e "  ${CYAN}📈 Prometheus:${NC}         http://localhost:9090"
        echo -e "  ${CYAN}📊 Grafana:${NC}           http://localhost:3001"
    fi
    
    echo ""
    print_info "🔍 Vérification des services:"
    echo "  docker-compose ps"
    echo "  docker-compose logs -f [service_name]"
    echo ""
    print_info "🛑 Arrêt des services:"
    echo "  docker-compose down"
    echo ""
}

# Afficher les logs
show_logs() {
    print_info "📋 Affichage des logs..."
    docker-compose logs -f
}

# Afficher le statut
show_status() {
    print_info "📊 Statut des services:"
    docker-compose ps
}

# Nettoyer complètement
clean_all() {
    print_warning "🧹 Nettoyage complet du système..."
    docker-compose down --rmi all --volumes --remove-orphans
    docker system prune -f
    print_success "Nettoyage complet terminé!"
}

# Afficher l'aide
show_help() {
    print_header
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  start                    - Démarrer tous les services"
    echo "  start --with-monitoring  - Démarrer avec monitoring (Prometheus/Grafana)"
    echo "  stop                     - Arrêter tous les services"
    echo "  restart                  - Redémarrer tous les services"
    echo "  logs                     - Afficher les logs"
    echo "  status                   - Afficher le statut"
    echo "  clean                    - Nettoyer les conteneurs"
    echo "  clean-all                - Nettoyage complet (supprime tout)"
    echo "  build                    - Construire les images"
    echo "  help                     - Afficher cette aide"
    echo ""
    echo "Services inclus:"
    echo "  • PostgreSQL (21 tables)"
    echo "  • Backend FastAPI (98 endpoints)"
    echo "  • Frontend Flutter (Web)"
    echo "  • Streamlit Dashboard"
    echo "  • MLflow (ML tracking)"
    echo "  • Redis (Cache)"
    echo "  • Nginx (Reverse Proxy)"
    echo "  • Prometheus/Grafana (Monitoring optionnel)"
    echo ""
}

# Script principal
main() {
    case "${1:-start}" in
        "start")
            print_header
            check_docker
            cleanup
            build_images
            start_services "$2"
            show_services_info "$2"
            ;;
        "stop")
            print_info "🛑 Arrêt des services..."
            docker-compose down
            print_success "Services arrêtés"
            ;;
        "restart")
            print_info "🔄 Redémarrage des services..."
            docker-compose restart
            print_success "Services redémarrés"
            ;;
        "logs")
            show_logs
            ;;
        "status")
            show_status
            ;;
        "clean")
            cleanup
            ;;
        "clean-all")
            clean_all
            ;;
        "build")
            check_docker
            build_images
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Commande inconnue: $1"
            show_help
            exit 1
            ;;
    esac
}

# Exécuter le script principal
main "$@"
