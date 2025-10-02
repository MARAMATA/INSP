# Makefile pour InspectIA
# Système d'analyse automatisée des documents douaniers

.PHONY: help setup install test preprocess train predict rules validate clean

# Variables
PYTHON = python3
PIP = pip3
CHAPTERS = 30 84 85
DATA_DIR = data
RAW_DIR = $(DATA_DIR)/raw
PROCESSED_DIR = $(DATA_DIR)/processed
MODELS_DIR = models

# Aide
help:
	@echo "InspectIA - Système d'analyse automatisée des documents douaniers"
	@echo ""
	@echo "Commandes disponibles:"
	@echo "  setup      - Installation des dépendances"
	@echo "  install    - Installation complète du système"
	@echo "  test       - Exécution des tests"
	@echo "  preprocess - Prétraitement des données (chapitre 30)"
	@echo "  train      - Entraînement du modèle (chapitre 30)"
	@echo "  predict    - Prédiction sur nouvelles données"
	@echo "  rules      - Affichage des règles métier"
	@echo "  validate   - Validation des données"
	@echo "  clean      - Nettoyage des fichiers temporaires"
	@echo ""
	@echo "Exemples:"
	@echo "  make preprocess CHAPTER=30"
	@echo "  make train CHAPTER=30"
	@echo "  make predict INPUT=data.csv OUTPUT=results.json"

# Installation des dépendances
setup:
	@echo "Installation des dépendances..."
	$(PIP) install -r requirements.txt
	@echo "✅ Dépendances installées"

# Installation complète
install: setup
	@echo "Installation complète d'InspectIA..."
	@mkdir -p $(PROCESSED_DIR)
	@mkdir -p $(MODELS_DIR)
	@mkdir -p logs
	@echo "✅ Installation terminée"

# Tests
test:
	@echo "Exécution des tests..."
	$(PYTHON) test_system.py
	@echo "✅ Tests terminés"

# Prétraitement
preprocess:
	@echo "Prétraitement du chapitre $(CHAPTER)..."
	$(PYTHON) src/cli.py preprocess --chapter $(CHAPTER) --output-dir $(PROCESSED_DIR)
	@echo "✅ Prétraitement terminé"

# Entraînement
train:
	@echo "Entraînement du modèle pour le chapitre $(CHAPTER)..."
	$(PYTHON) src/cli.py train --chapter $(CHAPTER)
	@echo "✅ Entraînement terminé"

# Prédiction
predict:
	@echo "Prédiction sur $(INPUT)..."
	$(PYTHON) src/cli.py predict --chapter $(CHAPTER) --input $(INPUT) --output $(OUTPUT)
	@echo "✅ Prédiction terminée"

# Affichage des règles
rules:
	@echo "Affichage des règles métier..."
	$(PYTHON) src/cli.py rules
	@echo "✅ Règles affichées"

# Validation des données
validate:
	@echo "Validation des données du chapitre $(CHAPTER)..."
	$(PYTHON) src/cli.py validate --chapter $(CHAPTER)
	@echo "✅ Validation terminée"

# Pipeline complet pour un chapitre
pipeline: preprocess train
	@echo "Pipeline complet terminé pour le chapitre $(CHAPTER)"

# Pipeline complet pour tous les chapitres
pipeline-all:
	@echo "Pipeline complet pour tous les chapitres..."
	@for chapter in $(CHAPTERS); do \
		echo "Traitement du chapitre $$chapter..."; \
		$(MAKE) pipeline CHAPTER=$$chapter; \
	done
	@echo "✅ Pipeline complet terminé"

# Nettoyage
clean:
	@echo "Nettoyage des fichiers temporaires..."
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.log" -delete
	@echo "✅ Nettoyage terminé"

# Nettoyage complet (supprime aussi les données traitées)
clean-all: clean
	@echo "Nettoyage complet..."
	@rm -rf $(PROCESSED_DIR)/*
	@rm -rf $(MODELS_DIR)/*
	@echo "✅ Nettoyage complet terminé"

# Vérification de l'installation
check:
	@echo "Vérification de l'installation..."
	@echo "Python: $(shell which $(PYTHON))"
	@echo "Version Python: $(shell $(PYTHON) --version)"
	@echo "Pip: $(shell which $(PIP))"
	@echo "Données brutes: $(shell ls $(RAW_DIR)/*.csv 2>/dev/null | wc -l) fichiers"
	@echo "✅ Vérification terminée"

# Statistiques des données
stats:
	@echo "Statistiques des données..."
	@for file in $(RAW_DIR)/*.csv; do \
		if [ -f "$$file" ]; then \
			echo "$$(basename $$file): $$(wc -l < $$file) lignes"; \
		fi; \
	done
	@echo "✅ Statistiques affichées"

# Documentation
docs:
	@echo "Génération de la documentation..."
	@mkdir -p docs
	@echo "# InspectIA - Documentation" > docs/README.md
	@echo "" >> docs/README.md
	@echo "## Installation" >> docs/README.md
	@echo "\`\`\`bash" >> docs/README.md
	@echo "make install" >> docs/README.md
	@echo "\`\`\`" >> docs/README.md
	@echo "" >> docs/README.md
	@echo "## Utilisation" >> docs/README.md
	@echo "\`\`\`bash" >> docs/README.md
	@echo "make preprocess CHAPTER=30" >> docs/README.md
	@echo "make train CHAPTER=30" >> docs/README.md
	@echo "make predict INPUT=data.csv OUTPUT=results.json" >> docs/README.md
	@echo "\`\`\`" >> docs/README.md
	@echo "✅ Documentation générée"

# Mode développement
dev-setup: install
	@echo "Configuration du mode développement..."
	$(PIP) install -r requirements-dev.txt
	@echo "✅ Mode développement configuré"

# Linting
lint:
	@echo "Vérification du code..."
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src/ --max-line-length=120 --ignore=E501,W503; \
	else \
		echo "flake8 non installé, installation..."; \
		$(PIP) install flake8; \
		flake8 src/ --max-line-length=120 --ignore=E501,W503; \
	fi
	@echo "✅ Linting terminé"

# Formatage du code
format:
	@echo "Formatage du code..."
	@if command -v black >/dev/null 2>&1; then \
		black src/ --line-length=120; \
	else \
		echo "black non installé, installation..."; \
		$(PIP) install black; \
		black src/ --line-length=120; \
	fi
	@echo "✅ Formatage terminé"

# Démarrage rapide
quickstart: install test preprocess
	@echo "Démarrage rapide terminé!"
	@echo "Vous pouvez maintenant utiliser:"
	@echo "  make train CHAPTER=30"
	@echo "  make predict INPUT=votre_fichier.csv OUTPUT=resultats.json"

# Aide détaillée
help-detailed:
	@echo "InspectIA - Aide détaillée"
	@echo "=========================="
	@echo ""
	@echo "Commandes principales:"
	@echo ""
	@echo "  make install"
	@echo "    Installation complète du système"
	@echo ""
	@echo "  make test"
	@echo "    Exécution de tous les tests"
	@echo ""
	@echo "  make preprocess CHAPTER=30"
	@echo "    Prétraitement des données du chapitre 30"
	@echo ""
	@echo "  make train CHAPTER=30"
	@echo "    Entraînement du modèle pour le chapitre 30"
	@echo ""
	@echo "  make predict CHAPTER=30 INPUT=data.csv OUTPUT=results.json"
	@echo "    Prédiction sur de nouvelles données"
	@echo ""
	@echo "  make pipeline CHAPTER=30"
	@echo "    Prétraitement + entraînement pour un chapitre"
	@echo ""
	@echo "  make pipeline-all"
	@echo "    Pipeline complet pour tous les chapitres"
	@echo ""
	@echo "Commandes utilitaires:"
	@echo ""
	@echo "  make rules"
	@echo "    Affichage de toutes les règles métier"
	@echo ""
	@echo "  make validate CHAPTER=30"
	@echo "    Validation des données d'un chapitre"
	@echo ""
	@echo "  make stats"
	@echo "    Statistiques des données brutes"
	@echo ""
	@echo "  make clean"
	@echo "    Nettoyage des fichiers temporaires"
	@echo ""
	@echo "  make lint"
	@echo "    Vérification de la qualité du code"
	@echo ""
	@echo "  make format"
	@echo "    Formatage automatique du code"

