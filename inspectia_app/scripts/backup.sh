#!/bin/bash

# 💾 SCRIPT DE SAUVEGARDE INSPECTIA
# Sauvegarde complète de la base de données et des volumes

set -e

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_message() {
    echo -e "${GREEN}[BACKUP]${NC} $1"
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
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
DB_NAME="inspectia_db"
DB_USER="inspectia_user"

# Créer le dossier de sauvegarde
mkdir -p "$BACKUP_DIR"

print_info "💾 Sauvegarde INSPECTIA - $(date)"
print_info "Dossier de sauvegarde: $BACKUP_DIR"

# Sauvegarder la base de données
backup_database() {
    print_info "🗄️ Sauvegarde de la base de données PostgreSQL..."
    
    if docker-compose ps postgres | grep -q "Up"; then
        docker-compose exec -T postgres pg_dump -U $DB_USER $DB_NAME > "$BACKUP_DIR/database.sql"
        print_info "✅ Base de données sauvegardée: $BACKUP_DIR/database.sql"
    else
        print_warning "PostgreSQL n'est pas en cours d'exécution"
    fi
}

# Sauvegarder les volumes Docker
backup_volumes() {
    print_info "📦 Sauvegarde des volumes Docker..."
    
    # Volume PostgreSQL
    if docker volume ls | grep -q "inspectia_postgres_data"; then
        print_info "Sauvegarde du volume PostgreSQL..."
        docker run --rm -v inspectia_postgres_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/postgres_data.tar.gz -C /data .
    fi
    
    # Volume MLflow
    if docker volume ls | grep -q "inspectia_mlflow_artifacts"; then
        print_info "Sauvegarde du volume MLflow..."
        docker run --rm -v inspectia_mlflow_artifacts:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/mlflow_artifacts.tar.gz -C /data .
    fi
    
    # Volume Redis
    if docker volume ls | grep -q "inspectia_redis_data"; then
        print_info "Sauvegarde du volume Redis..."
        docker run --rm -v inspectia_redis_data:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/redis_data.tar.gz -C /data .
    fi
    
    # Volumes backend
    if docker volume ls | grep -q "inspectia_backend_logs"; then
        print_info "Sauvegarde des logs backend..."
        docker run --rm -v inspectia_backend_logs:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/backend_logs.tar.gz -C /data .
    fi
    
    if docker volume ls | grep -q "inspectia_backend_results"; then
        print_info "Sauvegarde des résultats backend..."
        docker run --rm -v inspectia_backend_results:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/backend_results.tar.gz -C /data .
    fi
    
    if docker volume ls | grep -q "inspectia_backend_models"; then
        print_info "Sauvegarde des modèles ML..."
        docker run --rm -v inspectia_backend_models:/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf /backup/backend_models.tar.gz -C /data .
    fi
}

# Sauvegarder les configurations
backup_configs() {
    print_info "⚙️ Sauvegarde des configurations..."
    
    # Docker Compose
    cp docker-compose.yml "$BACKUP_DIR/"
    
    # Scripts
    cp -r scripts/ "$BACKUP_DIR/"
    
    # Configurations
    cp -r config/ "$BACKUP_DIR/" 2>/dev/null || true
    cp -r monitoring/ "$BACKUP_DIR/" 2>/dev/null || true
    cp -r nginx/ "$BACKUP_DIR/" 2>/dev/null || true
    
    # Makefile
    cp Makefile "$BACKUP_DIR/"
    
    print_info "✅ Configurations sauvegardées"
}

# Créer un manifeste de sauvegarde
create_manifest() {
    print_info "📋 Création du manifeste de sauvegarde..."
    
    cat > "$BACKUP_DIR/manifest.txt" << EOF
INSPECTIA BACKUP MANIFEST
========================
Date: $(date)
Version: 2.0.0
Database: $DB_NAME
User: $DB_USER

FILES INCLUDED:
- database.sql (PostgreSQL dump)
- postgres_data.tar.gz (PostgreSQL volume)
- mlflow_artifacts.tar.gz (MLflow artifacts)
- redis_data.tar.gz (Redis data)
- backend_logs.tar.gz (Backend logs)
- backend_results.tar.gz (Backend results)
- backend_models.tar.gz (ML models)
- docker-compose.yml (Docker configuration)
- scripts/ (Backup and deployment scripts)
- config/ (Application configuration)
- monitoring/ (Monitoring configuration)
- nginx/ (Nginx configuration)
- Makefile (Build automation)

RESTORE INSTRUCTIONS:
1. Extract all files to the project directory
2. Run: docker-compose down
3. Run: docker volume create inspectia_postgres_data
4. Run: docker run --rm -v inspectia_postgres_data:/data -v \$(pwd):/backup alpine tar xzf /backup/postgres_data.tar.gz -C /data
5. Run: docker-compose up -d postgres
6. Run: docker-compose exec postgres psql -U $DB_USER -d $DB_NAME -f /backup/database.sql
7. Run: docker-compose up -d

EOF
    
    print_info "✅ Manifeste créé: $BACKUP_DIR/manifest.txt"
}

# Compresser la sauvegarde
compress_backup() {
    print_info "🗜️ Compression de la sauvegarde..."
    
    cd backups
    tar czf "inspectia_backup_$(date +%Y%m%d_%H%M%S).tar.gz" "$(basename "$BACKUP_DIR")"
    cd ..
    
    print_info "✅ Sauvegarde compressée: backups/inspectia_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
}

# Nettoyer les anciennes sauvegardes
cleanup_old_backups() {
    print_info "🧹 Nettoyage des anciennes sauvegardes..."
    
    # Garder seulement les 7 dernières sauvegardes
    cd backups
    ls -t | tail -n +8 | xargs -r rm -rf
    cd ..
    
    print_info "✅ Anciennes sauvegardes nettoyées"
}

# Afficher le résumé
show_summary() {
    print_info "📊 Résumé de la sauvegarde:"
    echo ""
    echo "  📁 Dossier: $BACKUP_DIR"
    echo "  📊 Taille: $(du -sh "$BACKUP_DIR" | cut -f1)"
    echo "  📅 Date: $(date)"
    echo ""
    echo "  📋 Fichiers sauvegardés:"
    ls -la "$BACKUP_DIR"
    echo ""
    print_info "✅ Sauvegarde terminée avec succès!"
}

# Script principal
main() {
    case "${1:-full}" in
        "full")
            backup_database
            backup_volumes
            backup_configs
            create_manifest
            compress_backup
            cleanup_old_backups
            show_summary
            ;;
        "db")
            backup_database
            show_summary
            ;;
        "volumes")
            backup_volumes
            show_summary
            ;;
        "configs")
            backup_configs
            show_summary
            ;;
        "help"|"-h"|"--help")
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  full      - Sauvegarde complète (défaut)"
            echo "  db        - Sauvegarde de la base de données uniquement"
            echo "  volumes   - Sauvegarde des volumes uniquement"
            echo "  configs   - Sauvegarde des configurations uniquement"
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
