# MÉMOIRE INSPECTIA - PARTIE 3 - DIAGRAMMES UML CORRIGÉS

## 3.3 Diagrammes UML de la solution proposée

### 3.3.1 Diagrammes de cas d'utilisation

**Diagramme de cas d'utilisation - Inspecteur des douanes :**

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Inspecteur des douanes                   │   │
│  │                                                     │   │
│  │                    ┌─────────────────┐              │   │
│  │                    │ InspectIA       │              │   │
│  │                    │ System          │              │   │
│  │                    └─────────────────┘              │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ <<include>>                                     │ │   │
│  │  │ Upload Documents (PDF/CSV)                     │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ <<include>>                                     │ │   │
│  │  │ Consult ML-RL Predictions                       │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ <<include>>                                     │ │   │
│  │  │ Provide Feedback (RL System)                    │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ <<include>>                                     │ │   │
│  │  │ Consult PV List/Details                         │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ <<include>>                                     │ │   │
│  │  │ Consult RL Analytics/Performance                │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ <<include>>                                     │ │   │
│  │  │ Consult PostgreSQL Status/Backend Test          │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Diagramme de cas d'utilisation - Expert ML :**

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Expert ML                            │   │
│  │                                                     │   │
│  │                    ┌─────────────────┐              │   │
│  │                    │ InspectIA       │              │   │
│  │                    │ System          │              │   │
│  │                    └─────────────────┘              │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ <<include>>                                     │ │   │
│  │  │ Consult ML-RL Predictions                       │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ <<include>>                                     │ │   │
│  │  │ Provide Feedback (RL System)                    │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ <<include>>                                     │ │   │
│  │  │ Consult PV List/Details                         │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ <<include>>                                     │ │   │
│  │  │ Consult RL Analytics/Performance                │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ <<include>>                                     │ │   │
│  │  │ Optimize Decision Thresholds                    │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │                                                     │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │ <<include>>                                     │ │   │
│  │  │ Consult PostgreSQL Status/Backend Test          │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3.2 Diagrammes de séquence

**Diagramme de séquence - Analyse d'une déclaration avec prédiction de fraude :**

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Flutter    FastAPI    OCRPipeline    MLModel    RLManager  │
│    │           │           │            │           │       │
│    │           │           │            │           │       │
│    │───1: POST /upload─────▶│           │           │       │
│    │           │           │            │           │       │
│    │           │───2: process_declaration_file─────▶│       │
│    │           │           │            │           │       │
│    │           │◀──3: extracted_data────│           │       │
│    │           │           │            │           │       │
│    │           │───4: predict_fraud─────▶│           │       │
│    │           │           │            │           │       │
│    │           │           │───5: load_model───────▶│       │
│    │           │           │            │           │       │
│    │           │           │◀──6: prediction───────│       │
│    │           │           │            │           │       │
│    │           │───7: rl_predict───────▶│           │       │
│    │           │           │            │           │       │
│    │           │           │            │───8: choose_action▶│
│    │           │           │            │           │       │
│    │           │◀──9: rl_decision───────│           │       │
│    │           │           │            │           │       │
│    │           │───10: save_to_postgresql──────────────▶│   │
│    │           │           │            │           │       │
│    │◀──11: response────────│           │           │       │
│    │           │           │            │           │       │
│    │───12: Display Results │           │           │       │
│    │           │           │            │           │       │
└─────────────────────────────────────────────────────────────┘
```

### 3.3.3 Diagrammes de classe

**Diagramme de classe - Architecture ML-RL :**

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              AdvancedOCRPipeline                     │   │
│  │                                                     │   │
│  │  - CHAPTER_CONFIGS: Dict[str, Dict]                 │   │
│  │  - _MODEL_CACHE: Dict                               │   │
│  │  - _CACHE_LOCK: threading.Lock                      │   │
│  │                                                     │   │
│  │  + predict_fraud(data: Dict, chapter: str,          │   │
│  │    level: str) -> Dict[str, Any]                    │   │
│  │  + run_auto_predict(chapter: str, uploads: List,    │   │
│  │    declarations: List) -> Dict[str, Any]            │   │
│  │  + process_ocr_document(image_path: str,            │   │
│  │    chapter: str, level: str) -> Dict[str, Any]      │   │
│  │  + predict_fraud_from_ocr_data(ocr_data: Dict,      │   │
│  │    chapter: str, level: str) -> Dict[str, Any]      │   │
│  │  + get_chapter_config(chapter: str) -> Dict         │   │
│  │  + load_decision_thresholds(chapter: str) -> Dict   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                 │
│                          │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Chapitres ML (chap30/84/85)            │   │
│  │                                                     │   │
│  │  + preprocess.py (labellisation FRAUD_FLAG)         │   │
│  │  + ml_model.py (XGBoost, CatBoost, LightGBM,       │   │
│  │    RandomForest, LogisticRegression)                │   │
│  │  + business_rules.py (règles métier par chapitre)   │   │
│  │  + ocr_nlp.py (traitement NLP)                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                 │
│                          │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              AdvancedRLManager                       │   │
│  │                                                     │   │
│  │  - epsilon: float (0.03-0.04)                      │   │
│  │  - strategy: str ("hybrid")                         │   │
│  │  - inspector_profiles: Dict                         │   │
│  │  - bandit: MultiArmedBandit                         │   │
│  │  - store: RLDataStore                               │   │
│  │                                                     │   │
│  │  + predict(context: Dict, ml_probability: float,    │   │
│  │    ml_threshold: float) -> Dict[str, Any]           │   │
│  │  + update_feedback(feedback_data: Dict) -> Dict     │   │
│  │  + calculate_feedback_quality(agreement: bool,      │   │
│  │    confidence: float) -> float                      │   │
│  │  + get_performance_summary() -> Dict[str, Any]      │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                 │
│                          │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              AppState (Flutter)                     │   │
│  │                                                     │   │
│  │  - _lastAnalysisResult: Map<String, dynamic>        │   │
│  │  - _recentDeclarations: List<Map>                   │   │
│  │  - _pvList: List<Map>                               │   │
│  │  - _selectedChapter: String                         │   │
│  │                                                     │   │
│  │  + setLastAnalysisResult(result: Map) -> void       │   │
│  │  + addPV(pv: Map) -> void                           │   │
│  │  + autoPredict(chapter: String, requestData: Map)   │   │
│  │    -> Future<Map>                                   │   │
│  │  + processOcrDocument(chapter: String, fileBytes:   │   │
│  │    List<int>, fileName: String) -> Future<Map>      │   │
│  │  + rlPredict(chapter: String, context: Map)         │   │
│  │    -> Future<Map>                                   │   │
│  │  + addRlFeedback(chapter: String, feedbackData: Map)│   │
│  │    -> Future<bool>                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3.4 Diagrammes d'activité

**Diagramme d'activité - Pipeline complet de détection de fraude :**

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────┐                                        │
│  │   [DÉBUT]       │                                        │
│  │   Upload        │                                        │
│  │   Document      │                                        │
│  │   (PDF/CSV/     │                                        │
│  │   Image)        │                                        │
│  └─────────────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ ocr_ingest.py   │                                        │
│  │ process_declaration_file() │                             │
│  │ - process_pdf_declaration() │                            │
│  │ - process_csv_declaration() │                            │
│  │ - aggregate_csv_by_declaration() │                       │
│  └─────────────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ ocr_pipeline.py │                                        │
│  │ AdvancedOCRPipeline │                                    │
│  │ - predict_fraud() │                                      │
│  │ - run_auto_predict() │                                   │
│  │ - predict_fraud_from_ocr_data() │                        │
│  └─────────────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Chapitre ML     │                                        │
│  │ (chap30/84/85)  │                                        │
│  │ - preprocess.py (FRAUD_FLAG creation) │                  │
│  │ - ml_model.py (5 modèles calibrés) │                     │
│  │ - business_rules.py (règles métier) │                    │
│  └─────────────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ advanced_reinforcement_learning.py │                     │
│  │ AdvancedRLManager │                                      │
│  │ - predict() │                                            │
│  │ - MultiArmedBandit.choose() │                            │
│  │ - calculate_feedback_quality() │                         │
│  └─────────────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ routes_predict.py │                                      │
│  │ Endpoints API │                                          │
│  │ - /predict/{chapter} │                                   │
│  │ - /api/v2/declarations/upload/ │                         │
│  │ - save_declaration_to_postgresql() │                     │
│  └─────────────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ AppState        │                                        │
│  │ (Flutter)       │                                        │
│  │ State Management │                                       │
│  │ - setLastAnalysisResult() │                              │
│  │ - addPV() │                                              │
│  │ - SharedPreferences persistence │                        │
│  └─────────────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Frontend        │                                        │
│  │ Screens         │                                        │
│  │ (Flutter)       │                                        │
│  │ - HomeScreen (8 actions) │                               │
│  │ - UploadScreen (analyse) │                               │
│  │ - FeedbackScreen (RL feedback) │                         │
│  │ - PVListScreen (liste PV) │                              │
│  │ - PVDetailScreen (détails PV) │                          │
│  └─────────────────┘                                        │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │   [FIN]         │                                        │
│  └─────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3.5 Diagrammes de composants

**Diagramme de composants - Architecture système :**

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Frontend Layer                       │   │
│  │                                                     │   │
│  │  ┌─────────────────┐    ┌─────────────────┐        │   │
│  │  │   Flutter       │    │   Web Interface │        │   │
│  │  │   Mobile App    │    │   (Dashboard)   │        │   │
│  │  └─────────────────┘    └─────────────────┘        │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                 │
│                          │ HTTP/REST API                   │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                API Layer                            │   │
│  │                                                     │   │
│  │  ┌─────────────────┐    ┌─────────────────┐        │   │
│  │  │   FastAPI       │    │   CORS          │        │   │
│  │  │   Server        │    │   Middleware    │        │   │
│  │  └─────────────────┘    └─────────────────┘        │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                 │
│                          │ Internal Calls                  │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Business Layer                       │   │
│  │                                                     │   │
│  │  ┌─────────────────┐    ┌─────────────────┐        │   │
│  │  │   OCR Pipeline  │    │   ML Models     │        │   │
│  │  │   (Processing)  │    │   (Prediction)  │        │   │
│  │  └─────────────────┘    └─────────────────┘        │   │
│  │           │                       │                 │   │
│  │           ▼                       ▼                 │   │
│  │  ┌─────────────────┐    ┌─────────────────┐        │   │
│  │  │   RL Manager    │    │   Business      │        │   │
│  │  │   (Learning)    │    │   Rules Engine  │        │   │
│  │  └─────────────────┘    └─────────────────┘        │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                 │
│                          │ Database Connections            │
│                          ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                Data Layer                           │   │
│  │                                                     │   │
│  │  ┌─────────────────┐    ┌─────────────────┐        │   │
│  │  │   PostgreSQL    │    │   SQLite        │        │   │
│  │  │   (Main Data)   │    │   (RL Data)     │        │   │
│  │  └─────────────────┘    └─────────────────┘        │   │
│  │           │                       │                 │   │
│  │           ▼                       ▼                 │   │
│  │  ┌─────────────────┐    ┌─────────────────┐        │   │
│  │  │   declarations  │    │   advanced_     │        │   │
│  │  │   predictions   │    │   decisions     │        │   │
│  │  │   pv_inspection │    │   advanced_     │        │   │
│  │  └─────────────────┘    └─────────────────┘        │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

*Diagrammes UML conformes aux standards et conventions UML 2.5*
