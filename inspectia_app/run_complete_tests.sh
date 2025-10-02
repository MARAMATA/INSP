#!/bin/bash

# 🧪 SCRIPT DE TESTS COMPLETS FRONTEND-BACKEND
# Démarre le backend et exécute tous les tests de communication

echo "🚀 TESTS COMPLETS DE COMMUNICATION FRONTEND-BACKEND"
echo "=================================================="

# Couleurs pour les logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher les logs colorés
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

# Vérifier pip
if ! command -v pip3 &> /dev/null; then
    log_error "pip3 n'est pas installé"
    exit 1
fi

# Vérifier Flutter (optionnel)
if command -v flutter &> /dev/null; then
    log_success "Flutter détecté: $(flutter --version | head -n 1)"
else
    log_warning "Flutter non détecté - les tests Flutter seront ignorés"
fi

# Vérifier les dépendances Python
log_info "Vérification des dépendances Python..."
cd backend

if [ ! -f "requirements.txt" ]; then
    log_error "Fichier requirements.txt non trouvé dans backend/"
    exit 1
fi

# Installer les dépendances si nécessaire
log_info "Installation des dépendances Python..."
pip3 install -r requirements.txt > /dev/null 2>&1
if [ $? -eq 0 ]; then
    log_success "Dépendances Python installées"
else
    log_warning "Erreur lors de l'installation des dépendances Python"
fi

cd ..

# Démarrer le backend en arrière-plan
log_info "Démarrage du backend..."
cd backend

# Tuer tout processus existant sur le port 8000
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Démarrer le backend
nohup python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload > ../backend.log 2>&1 &
BACKEND_PID=$!

cd ..

# Attendre que le backend démarre
log_info "Attente du démarrage du backend..."
sleep 5

# Vérifier que le backend est accessible
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        log_success "Backend démarré et accessible"
        break
    fi
    if [ $i -eq 30 ]; then
        log_error "Backend non accessible après 30 secondes"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# Exécuter les tests Python
log_info "Exécution des tests Python..."
python3 test_communication_complete.py

PYTHON_TEST_RESULT=$?

# Exécuter les tests Flutter si Flutter est disponible
if command -v flutter &> /dev/null; then
    log_info "Exécution des tests Flutter..."
    cd inspectia_app_frontend
    
    # Installer les dépendances Flutter
    flutter pub get > /dev/null 2>&1
    
    # Exécuter les tests Flutter
    flutter test test_frontend_backend.dart
    FLUTTER_TEST_RESULT=$?
    
    cd ..
else
    log_warning "Tests Flutter ignorés (Flutter non installé)"
    FLUTTER_TEST_RESULT=0
fi

# Arrêter le backend
log_info "Arrêt du backend..."
kill $BACKEND_PID 2>/dev/null
sleep 2

# Résumé final
echo ""
echo "=================================================="
echo "📊 RÉSUMÉ DES TESTS"
echo "=================================================="

if [ $PYTHON_TEST_RESULT -eq 0 ]; then
    log_success "Tests Python: RÉUSSIS"
else
    log_error "Tests Python: ÉCHOUÉS"
fi

if [ $FLUTTER_TEST_RESULT -eq 0 ]; then
    log_success "Tests Flutter: RÉUSSIS"
else
    log_error "Tests Flutter: ÉCHOUÉS"
fi

# Vérifier les logs du backend
if [ -f "backend.log" ]; then
    log_info "Logs du backend disponibles dans backend.log"
fi

# Vérifier les résultats des tests
if [ -f "test_communication_results.json" ]; then
    log_info "Résultats détaillés disponibles dans test_communication_results.json"
fi

# Code de sortie global
if [ $PYTHON_TEST_RESULT -eq 0 ] && [ $FLUTTER_TEST_RESULT -eq 0 ]; then
    log_success "Tous les tests sont passés avec succès!"
    exit 0
else
    log_error "Certains tests ont échoué"
    exit 1
fi
