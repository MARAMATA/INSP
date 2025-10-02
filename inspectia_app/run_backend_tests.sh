#!/bin/bash

# 🚀 SCRIPT DE TEST COMPLET BACKEND-FRONTEND
# Démarre le backend et exécute tous les tests de communication

set -e

echo "🚀 TESTS COMPLETS DE COMMUNICATION BACKEND-FRONTEND"
echo "=================================================="

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages colorés
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Vérifier les prérequis
log_info "Vérification des prérequis..."

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 n'est pas installé"
    exit 1
fi
log_success "Python 3 détecté"

# Vérifier les dépendances Python
log_info "Vérification des dépendances Python..."
cd backend

if ! python3 -c "import fastapi, uvicorn, requests" 2>/dev/null; then
    log_warning "Installation des dépendances Python..."
    pip3 install fastapi uvicorn requests
fi
log_success "Dépendances Python OK"

# Démarrer le backend en arrière-plan
log_info "Démarrage du backend..."
python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Attendre que le backend démarre
log_info "Attente du démarrage du backend..."
sleep 10

# Vérifier que le backend répond
for i in {1..30}; do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        log_success "Backend démarré et accessible"
        break
    fi
    if [ $i -eq 30 ]; then
        log_error "Backend non accessible après 30 secondes"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    sleep 1
done

# Retourner au répertoire principal
cd ..

# Exécuter les tests de communication
log_info "Exécution des tests de communication..."
python3 test_communication_complete.py
TEST_RESULT=$?

# Arrêter le backend
log_info "Arrêt du backend..."
kill $BACKEND_PID 2>/dev/null || true

# Résultat final
if [ $TEST_RESULT -eq 0 ]; then
    log_success "Tous les tests sont passés avec succès!"
    exit 0
else
    log_error "Certains tests ont échoué"
    exit 1
fi
