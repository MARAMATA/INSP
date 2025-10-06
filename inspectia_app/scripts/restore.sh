#!/bin/bash

# 📥 SCRIPT DE RESTAURATION INSPECTIA
# Restauration complète depuis une sauvegarde

set -e

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_message() {
    echo -e "${GREEN}[RESTORE]${NC} $1"
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

# Configuration
DB_NAME="inspectia_db"
DB_USER="inspectia_user"

# Vérifier qu'un fichier de sauvegarde est fourni
if [ -z "$1" ]; then
    print_error "Usage: $0 <backup_file.tar.gz>"
    echo ""
    echo "Fichiers de sauvegarde disponibles:"
    ls -la backups/*.tar.gz 2>/dev/null || echo "Aucun fichier de sauvegarde trouvé"
    exit 1
fi

BACKUP_FILE="$1"
RESTORE_DIR="restore_$(date +%Y%m%d_%H%M%S)"

print_info "📥 Restauration INSPECTIA - $(date)"
print_info "Fichier de sauvegarde: $BACKUP_FILE"

# Vérifier que le fichier de sauvegarde existe
if [ ! -f "$BACKUP_FILE" ]; then
    print_error "Fichier de sauvegarde non trouvé: $BACKUP_FILE"
    exit 1
fi

# Extraire la sauvegarde
extract_backup() {
    print_info "📦 Extraction de la sauvegarde..."
    
    mkdir -p "$RESTORE_DIR"
    tar xzf "$BACKUP_FILE" -C "$RESTORE_DIR" --strip-components=1
    
    print_info "✅ Sauvegarde extraite dans: $RESTORE_DIR"
}

# Arrêter les services
stop_services() {
    print_info "🛑 Arrêt des services..."
    docker-compose down
    print_info "✅ Services arrêtés"
}

# Restaurer les volumes
restore_volumes() {
    print_info "📦 Restauration des volumes Docker..."
    
    # Créer les volumes s'ils n'existent pas
    docker volume create inspectia_postgres_data 2>/dev/null || true
    docker volume create inspectia_mlflow_artifacts 2>/dev/null || true
    docker volume create inspectia_redis_data 2>/dev/null || true
    docker volume create inspectia_backend_logs 2>/dev/null || true
    docker volume create inspectia_backend_results 2>/dev/null || true
    docker volume create inspectia_backend_models 2>/dev/null || true
    
    # Restaurer le volume PostgreSQL
    if [ -f "$RESTORE_DIR/postgres_data.tar.gz" ]; then
        print_info "Restauration du volume PostgreSQL..."
        docker run --rm -v inspectia_postgres_data:/data -v "$(pwd)/$RESTORE_DIR":/backup alpine tar xzf /backup/postgres_data.tar.gz -C /data
    fi
    
    # Restaurer le volume MLflow
    if [ -f "$RESTORE_DIR/mlflow_artifacts.tar.gz" ]; then
        print_info "Restauration du volume MLflow..."
        docker run --rm -v inspectia_mlflow_artifacts:/data -v "$(pwd)/$RESTORE_DIR":/backup alpine tar xzf /backup/mlflow_artifacts.tar.gz -C /data
    fi
    
    # Restaurer le volume Redis
    if [ -f "$RESTORE_DIR/redis_data.tar.gz" ]; then
        print_info "Restauration du volume Redis..."
        docker run --rm -v inspectia_redis_data:/data -v "$(pwd)/$RESTORE_DIR":/backup alpine tar xzf /backup/redis_data.tar.gz -C /data
    fi
    
    # Restaurer les volumes backend
    if [ -f "$RESTORE_DIR/backend_logs.tar.gz" ]; then
        print_info "Restauration des logs backend..."
        docker run --rm -v inspectia_backend_logs:/data -v "$(pwd)/$RESTORE_DIR":/backup alpine tar xzf /backup/backend_logs.tar.gz -C /data
    fi
    
    if [ -f "$RESTORE_DIR/backend_results.tar.gz" ]; then
        print_info "Restauration des résultats backend..."
        docker run --rm -v inspectia_backend_results:/data -v "$(pwd)/$RESTORE_DIR":/backup alpine tar xzf /backup/backend_results.tar.gz -C /data
    fi
    
    if [ -f "$RESTORE_DIR/backend_models.tar.gz" ]; then
        print_info "Restauration des modèles ML..."
        docker run --rm -v inspectia_backend_models:/data -v "$(pwd)/$RESTORE_DIR":/backup alpine tar xzf /backup/backend_models.tar.gz -C /data
    fi
    
    print_info "✅ Volumes restaurés"
}

# Démarrer PostgreSQL
start_postgres() {
    print_info "🗄️ Démarrage de PostgreSQL..."
    docker-compose up -d postgres
    
    # Attendre que PostgreSQL soit prêt
    print_info "⏳ Attente de PostgreSQL..."
    until docker-compose exec postgres pg_isready -U $DB_USER -d $DB_NAME; do
        echo -n "."
        sleep 2
    done
    echo ""
    print_info "✅ PostgreSQL est prêt"
}

# Restaurer la base de données
restore_database() {
    print_info "🗄️ Restauration de la base de données..."
    
    if [ -f "$RESTORE_DIR/database.sql" ]; then
        docker-compose exec -T postgres psql -U $DB_USER -d $DB_NAME -f /backup/database.sql
        print_info "✅ Base de données restaurée"
    else
        print_warning "Fichier database.sql non trouvé dans la sauvegarde"
    fi
}

# Démarrer tous les services
start_services() {
    print_info "🚀 Démarrage de tous les services..."
    docker-compose up -d
    print_info "✅ Services démarrés"
}

# Vérifier la restauration
verify_restore() {
    print_info "🔍 Vérification de la restauration..."
    
    # Attendre que les services soient prêts
    sleep 10
    
    # Vérifier PostgreSQL
    if docker-compose exec postgres pg_isready -U $DB_USER -d $DB_NAME; then
        print_info "✅ PostgreSQL accessible"
    else
        print_error "❌ PostgreSQL non accessible"
        return 1
    fi
    
    # Vérifier le backend
    if curl -f http://localhost:8000/health &>/dev/null; then
        print_info "✅ Backend API accessible"
    else
        print_warning "⚠️ Backend API non accessible (peut prendre du temps)"
    fi
    
    # Vérifier le frontend
    if curl -f http://localhost:3000 &>/dev/null; then
        print_info "✅ Frontend accessible"
    else
        print_warning "⚠️ Frontend non accessible (peut prendre du temps)"
    fi
    
    print_info "✅ Restauration vérifiée"
}

# Nettoyer les fichiers temporaires
cleanup() {
    print_info "🧹 Nettoyage des fichiers temporaires..."
    rm -rf "$RESTORE_DIR"
    print_info "✅ Nettoyage terminé"
}

# Afficher le résumé
show_summary() {
    print_info "📊 Résumé de la restauration:"
    echo ""
    echo "  📁 Fichier restauré: $BACKUP_FILE"
    echo "  📅 Date: $(date)"
    echo ""
    echo "  🌐 Services disponibles:"
    echo "    📱 Frontend Flutter:     http://localhost:3000"
    echo "    🔧 Backend API:          http://localhost:8000"
    echo "    📊 Streamlit Dashboard:  http://localhost:8501"
    echo "    🤖 MLflow:              http://localhost:5000"
    echo "    🗄️  PostgreSQL:         localhost:5432"
    echo "    🔴 Redis:               localhost:6379"
    echo ""
    print_info "✅ Restauration terminée avec succès!"
}

# Script principal
main() {
    extract_backup
    stop_services
    restore_volumes
    start_postgres
    restore_database
    start_services
    verify_restore
    cleanup
    show_summary
}

# Exécuter le script principal
main
