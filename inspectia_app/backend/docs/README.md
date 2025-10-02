# InspectIA Backend

## Démarrage rapide

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt -r backend/requirements-dev.txt
python -m backend.src.cli preprocess --chapter all
python -m backend.src.main
```

## Docker

```bash
cd backend
docker compose up --build
```

## Endpoints
Voir `docs/API_REFERENCE.md`.
