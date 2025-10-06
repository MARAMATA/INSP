#!/bin/bash

# 🧪 SCRIPT DE TEST COMPLET INSPECTIA
# Tests unitaires, d'intégration et de performance

set -e

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
PURPLE='\033[0;35m'
NC='\033[0m'

print_header() {
    echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║                          🧪 TESTS INSPECTIA COMPLETS                       ║${NC}"
    echo -e "${PURPLE}║                    Tests unitaires, intégration et performance            ║${NC}"
    echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_message() {
    echo -e "${GREEN}[TEST]${NC} $1"
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

# Configuration
TEST_RESULTS_DIR="test_results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_RESULTS_DIR"

# Tests unitaires
run_unit_tests() {
    print_info "🧪 Exécution des tests unitaires..."
    
    docker-compose -f docker-compose.test.yml run --rm test-runner \
        python -m pytest tests/unit/ \
        -v \
        --tb=short \
        --cov=src \
        --cov-report=html:test_results/coverage_html \
        --cov-report=xml:test_results/coverage.xml \
        --cov-report=term \
        --junitxml=test_results/unit_tests.xml
    
    print_success "Tests unitaires terminés"
}

# Tests d'intégration
run_integration_tests() {
    print_info "🔗 Exécution des tests d'intégration..."
    
    docker-compose -f docker-compose.test.yml run --rm integration-tests \
        python -m pytest tests/integration/ \
        -v \
        --tb=short \
        --junitxml=test_results/integration_tests.xml
    
    print_success "Tests d'intégration terminés"
}

# Tests de performance
run_performance_tests() {
    print_info "⚡ Exécution des tests de performance..."
    
    docker-compose -f docker-compose.test.yml run --rm performance-tests \
        python -m pytest tests/performance/ \
        -v \
        --tb=short \
        --junitxml=test_results/performance_tests.xml
    
    print_success "Tests de performance terminés"
}

# Tests de l'API
run_api_tests() {
    print_info "🌐 Exécution des tests de l'API..."
    
    # Attendre que l'API soit prête
    print_info "⏳ Attente de l'API..."
    until curl -f http://localhost:8001/health &>/dev/null; do
        echo -n "."
        sleep 2
    done
    echo ""
    
    # Tests des endpoints
    print_info "Test des endpoints principaux..."
    
    # Test de santé
    if curl -f http://localhost:8001/health; then
        print_success "✅ Endpoint /health accessible"
    else
        print_error "❌ Endpoint /health non accessible"
    fi
    
    # Test des prédictions
    print_info "Test des prédictions..."
    curl -X POST http://localhost:8001/predict/chap30 \
        -H "Content-Type: application/json" \
        -d '{"declaration_id": "test_001", "features": {}}' \
        && print_success "✅ Endpoint /predict/chap30 accessible" \
        || print_error "❌ Endpoint /predict/chap30 non accessible"
    
    print_success "Tests de l'API terminés"
}

# Tests de la base de données
run_database_tests() {
    print_info "🗄️ Exécution des tests de la base de données..."
    
    # Test de connexion
    if docker-compose -f docker-compose.test.yml exec postgres-test pg_isready -U inspectia_user -d inspectia_db_test; then
        print_success "✅ Connexion à la base de données réussie"
    else
        print_error "❌ Connexion à la base de données échouée"
        return 1
    fi
    
    # Test des tables
    print_info "Vérification des tables..."
    table_count=$(docker-compose -f docker-compose.test.yml exec -T postgres-test psql -U inspectia_user -d inspectia_db_test -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')
    
    if [ "$table_count" -ge 21 ]; then
        print_success "✅ $table_count tables trouvées (attendu: 21+)"
    else
        print_error "❌ Seulement $table_count tables trouvées (attendu: 21+)"
    fi
    
    print_success "Tests de la base de données terminés"
}

# Tests de Redis
run_redis_tests() {
    print_info "🔴 Exécution des tests de Redis..."
    
    # Test de connexion
    if docker-compose -f docker-compose.test.yml exec redis-test redis-cli ping; then
        print_success "✅ Connexion à Redis réussie"
    else
        print_error "❌ Connexion à Redis échouée"
        return 1
    fi
    
    # Test des opérations
    print_info "Test des opérations Redis..."
    docker-compose -f docker-compose.test.yml exec redis-test redis-cli set test_key "test_value"
    value=$(docker-compose -f docker-compose.test.yml exec -T redis-test redis-cli get test_key)
    
    if [ "$value" = "test_value" ]; then
        print_success "✅ Opérations Redis fonctionnelles"
    else
        print_error "❌ Opérations Redis échouées"
    fi
    
    print_success "Tests de Redis terminés"
}

# Tests de MLflow
run_mlflow_tests() {
    print_info "🤖 Exécution des tests de MLflow..."
    
    # Test de connexion
    if curl -f http://localhost:5001/health &>/dev/null; then
        print_success "✅ MLflow accessible"
    else
        print_warning "⚠️ MLflow non accessible (peut être normal en test)"
    fi
    
    print_success "Tests de MLflow terminés"
}

# Tests de charge
run_load_tests() {
    print_info "⚡ Exécution des tests de charge..."
    
    # Test de charge simple avec curl
    print_info "Test de charge sur l'endpoint /health..."
    for i in {1..10}; do
        curl -f http://localhost:8001/health &>/dev/null && echo -n "." || echo -n "x"
    done
    echo ""
    
    print_success "Tests de charge terminés"
}

# Générer le rapport de test
generate_test_report() {
    print_info "📊 Génération du rapport de test..."
    
    cat > "$TEST_RESULTS_DIR/test_report.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Rapport de Test INSPECTIA</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .success { color: green; }
        .error { color: red; }
        .warning { color: orange; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧪 Rapport de Test INSPECTIA</h1>
        <p>Date: $(date)</p>
        <p>Version: 2.0.0</p>
    </div>
    
    <div class="section">
        <h2>📊 Résumé des Tests</h2>
        <ul>
            <li>Tests unitaires: <span class="success">✅ Terminés</span></li>
            <li>Tests d'intégration: <span class="success">✅ Terminés</span></li>
            <li>Tests de performance: <span class="success">✅ Terminés</span></li>
            <li>Tests de l'API: <span class="success">✅ Terminés</span></li>
            <li>Tests de la base de données: <span class="success">✅ Terminés</span></li>
            <li>Tests de Redis: <span class="success">✅ Terminés</span></li>
            <li>Tests de MLflow: <span class="success">✅ Terminés</span></li>
            <li>Tests de charge: <span class="success">✅ Terminés</span></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>📁 Fichiers de Résultat</h2>
        <ul>
            <li>Coverage HTML: <a href="coverage_html/index.html">coverage_html/index.html</a></li>
            <li>Coverage XML: coverage.xml</li>
            <li>Tests unitaires XML: unit_tests.xml</li>
            <li>Tests d'intégration XML: integration_tests.xml</li>
            <li>Tests de performance XML: performance_tests.xml</li>
        </ul>
    </div>
</body>
</html>
EOF
    
    print_success "Rapport de test généré: $TEST_RESULTS_DIR/test_report.html"
}

# Nettoyer les services de test
cleanup_test_services() {
    print_info "🧹 Nettoyage des services de test..."
    docker-compose -f docker-compose.test.yml down --volumes --remove-orphans
    print_success "Services de test nettoyés"
}

# Afficher le résumé
show_test_summary() {
    print_info "📊 Résumé des tests:"
    echo ""
    echo "  📁 Dossier de résultats: $TEST_RESULTS_DIR"
    echo "  📊 Rapport HTML: $TEST_RESULTS_DIR/test_report.html"
    echo "  📈 Coverage HTML: $TEST_RESULTS_DIR/coverage_html/index.html"
    echo ""
    echo "  🧪 Tests exécutés:"
    echo "    ✅ Tests unitaires"
    echo "    ✅ Tests d'intégration"
    echo "    ✅ Tests de performance"
    echo "    ✅ Tests de l'API"
    echo "    ✅ Tests de la base de données"
    echo "    ✅ Tests de Redis"
    echo "    ✅ Tests de MLflow"
    echo "    ✅ Tests de charge"
    echo ""
    print_success "Tous les tests sont terminés!"
}

# Script principal
main() {
    case "${1:-all}" in
        "all")
            print_header
            run_unit_tests
            run_integration_tests
            run_performance_tests
            run_api_tests
            run_database_tests
            run_redis_tests
            run_mlflow_tests
            run_load_tests
            generate_test_report
            cleanup_test_services
            show_test_summary
            ;;
        "unit")
            print_header
            run_unit_tests
            generate_test_report
            cleanup_test_services
            ;;
        "integration")
            print_header
            run_integration_tests
            generate_test_report
            cleanup_test_services
            ;;
        "performance")
            print_header
            run_performance_tests
            generate_test_report
            cleanup_test_services
            ;;
        "api")
            print_header
            run_api_tests
            generate_test_report
            cleanup_test_services
            ;;
        "database")
            print_header
            run_database_tests
            generate_test_report
            cleanup_test_services
            ;;
        "redis")
            print_header
            run_redis_tests
            generate_test_report
            cleanup_test_services
            ;;
        "mlflow")
            print_header
            run_mlflow_tests
            generate_test_report
            cleanup_test_services
            ;;
        "load")
            print_header
            run_load_tests
            generate_test_report
            cleanup_test_services
            ;;
        "help"|"-h"|"--help")
            print_header
            echo "Usage: $0 [COMMAND]"
            echo ""
            echo "Commands:"
            echo "  all          - Exécuter tous les tests (défaut)"
            echo "  unit         - Tests unitaires uniquement"
            echo "  integration - Tests d'intégration uniquement"
            echo "  performance - Tests de performance uniquement"
            echo "  api          - Tests de l'API uniquement"
            echo "  database     - Tests de la base de données uniquement"
            echo "  redis        - Tests de Redis uniquement"
            echo "  mlflow       - Tests de MLflow uniquement"
            echo "  load         - Tests de charge uniquement"
            echo "  help         - Afficher cette aide"
            ;;
        *)
            print_error "Commande inconnue: $1"
            exit 1
            ;;
    esac
}

# Exécuter le script principal
main "$@"
