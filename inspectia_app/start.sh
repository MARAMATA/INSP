#!/bin/bash

# Script de démarrage pour InspectIA
# Usage: ./start.sh [dev|prod|stop|logs|clean]

set -e

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
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

# Vérifier si Docker est installé et en cours d'exécution
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

# Fonction d'aide
show_help() {
    echo "InspectIA - Script de démarrage Docker"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  dev     - Démarrer en mode développement (avec volumes)"
    echo "  prod    - Démarrer en mode production"
    echo "  stop    - Arrêter tous les services"
    echo "  logs    - Afficher les logs des services"
    echo "  clean   - Nettoyer les conteneurs et images"
    echo "  status  - Afficher le statut des services"
    echo "  help    - Afficher cette aide"
    echo ""
    echo "Services disponibles:"
    echo "  - Backend API: http://localhost:8000"
    echo "  - Frontend Flutter: http://localhost:3000"
    echo "  - Streamlit: http://localhost:8501"
    echo "  - MLflow: http://localhost:5000"
    echo "  - PostgreSQL: localhost:5432"
}

# Démarrer en mode développement
start_dev() {
    print_message "Démarrage d'InspectIA en mode développement..."
    cd backend
    docker-compose -f docker-compose.yml up --build -d
    print_message "Services démarrés avec succès!"
    show_services_info
}

# Démarrer en mode production
start_prod() {
    print_message "Démarrage d'InspectIA en mode production..."
    cd backend
    docker-compose -f docker-compose.yml up --build -d --no-dev
    print_message "Services démarrés en mode production!"
    show_services_info
}

# Arrêter les services
stop_services() {
    print_message "Arrêt des services InspectIA..."
    cd backend
    docker-compose down
    print_message "Services arrêtés avec succès!"
}

# Afficher les logs
show_logs() {
    print_message "Affichage des logs des services..."
    cd backend
    docker-compose logs -f
}

# Nettoyer les conteneurs et images
clean_docker() {
    print_warning "Nettoyage des conteneurs et images Docker..."
    cd backend
    docker-compose down --rmi all --volumes --remove-orphans
    docker system prune -f
    print_message "Nettoyage terminé!"
}

# Afficher le statut des services
show_status() {
    print_message "Statut des services InspectIA:"
    cd backend
    docker-compose ps
}

# Afficher les informations des services
show_services_info() {
    echo ""
    print_info "Services disponibles:"
    echo "  🌐 Frontend Flutter: http://localhost:3000"
    echo "  🔧 Backend API: http://localhost:8000"
    echo "  📊 Streamlit Dashboard: http://localhost:8501"
    echo "  🤖 MLflow: http://localhost:5000"
    echo "  🗄️  PostgreSQL: localhost:5432"
    echo ""
    print_info "Pour voir les logs: $0 logs"
    print_info "Pour arrêter: $0 stop"
}

# Script principal
main() {
    check_docker

    case "${1:-dev}" in
        "dev")
            start_dev
            ;;
        "prod")
            start_prod
            ;;
        "stop")
            stop_services
            ;;
        "logs")
            show_logs
            ;;
        "clean")
            clean_docker
            ;;
        "status")
            show_status
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
