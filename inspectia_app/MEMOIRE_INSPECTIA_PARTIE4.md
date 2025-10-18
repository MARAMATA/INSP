# CHAPITRE 4 : IMPLÉMENTATION ET DÉVELOPPEMENT

## 4.1 Pipeline de Traitement des Données

### 4.1.1 Architecture du Pipeline

Le pipeline de traitement des données d'INSPECT_IA suit une architecture modulaire et extensible, permettant le traitement automatique des déclarations depuis l'upload jusqu'à la prédiction finale.

**Architecture complète du pipeline :**

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE TRAITEMENT                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Upload    │    │   OCR       │    │   Parsing   │        │
│  │ Document    │───►│ Processing  │───►│ Fields      │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ Validation  │    │ Feature     │    │ ML          │        │
│  │ Data        │───►│ Extraction  │───►│ Prediction  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│         │                   │                   │              │
│         ▼                   ▼                   ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │ SHAP        │    │ RL          │    │ Storage     │        │
│  │ Explanation │───►│ Threshold   │───►│ Database    │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Composants du pipeline :**

1. **Upload et Validation** : Gestion des fichiers multi-format
2. **OCR Processing** : Extraction automatique des données
3. **Preprocessing Avancé** : Techniques de la cellule de ciblage
4. **Feature Engineering** : Création de features métier
5. **Prédiction ML** : Modèles optimisés par chapitre (validés sur 487,230 déclarations)
6. **Explication SHAP** : Interprétabilité des décisions
7. **Décision RL** : Optimisation des seuils
8. **Persistance** : Sauvegarde en base de données

### 4.1.2 Implémentation du Preprocessing Avancé

**Chapitre 30 - Produits Pharmaceutiques :**

```python
class Chap30PreprocessorComprehensive:
    def create_advanced_fraud_flag(self, df):
        """Création du FRAUD_FLAG basé sur les techniques avancées"""
        
        # Initialisation
        df['FRAUD_FLAG'] = 0
        
        # 1. MÉTHODES PROBABILISTES (Théorème de Bienaymé-Tchebychev)
        # Détection des valeurs aberrantes basée sur les écarts-types
        for col in ['VALEUR_CAF', 'VALEUR_UNITAIRE_KG', 'TAUX_DROITS_PERCENT']:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                threshold = mean_val + 3 * std_val  # Règle des 3 sigmas
                df.loc[df[col] > threshold, 'FRAUD_FLAG'] = 1
        
        # 2. ANALYSE MIROIR avec TEI (Taux Effectifs d'Imposition)
        # Comparaison avec les valeurs de référence par pays/produit
        for idx, row in df.iterrows():
            product_origin_key = f"{row['CODE_PRODUIT_STR']}_{row['PAYS_ORIGINE_STR']}"
            if product_origin_key in self.reference_stats:
                ref_stats = self.reference_stats[product_origin_key]
                # Vérification des écarts significatifs
                if (abs(row['VALEUR_CAF'] - ref_stats['mean']) > 2 * ref_stats['std']):
                    df.loc[idx, 'FRAUD_FLAG'] = 1
        
        # 3. DÉTECTION D'ANOMALIES (clustering spectral)
        # Identification des déclarations atypiques
        numeric_cols = ['VALEUR_CAF', 'POIDS_NET', 'VALEUR_UNITAIRE_KG']
        if len(df) > 100:  # Seuil minimum pour le clustering
            from sklearn.cluster import SpectralClustering
            X_numeric = df[numeric_cols].fillna(0)
            clustering = SpectralClustering(n_clusters=3, random_state=42)
            clusters = clustering.fit_predict(X_numeric)
            # Marquer les clusters minoritaires comme suspects
            cluster_counts = pd.Series(clusters).value_counts()
            minority_clusters = cluster_counts[cluster_counts < len(df) * 0.1].index
            df.loc[df.index.isin(minority_clusters), 'FRAUD_FLAG'] = 1
        
        # 4. CONTRÔLE DES VALEURS ADMINISTRÉES
        # Vérification des seuils réglementaires
        admin_thresholds = {
            'VALEUR_CAF': 1000000,  # 1 million FCFA
            'VALEUR_UNITAIRE_KG': 50000,  # 50k FCFA/kg
            'TAUX_DROITS_PERCENT': 50  # 50%
        }
        
        for col, threshold in admin_thresholds.items():
            if col in df.columns:
                df.loc[df[col] > threshold, 'FRAUD_FLAG'] = 1
        
        # 5. RÈGLES SPÉCIFIQUES CHAPITRE 30
        # Glissement tarifaire cosmétiques/médicaments
        mask_medicaments = df['CODE_PRODUIT_STR'].str.startswith('30', na=False)
        
        # Détection des cosmétiques classés comme médicaments
        cosmetic_keywords = ['cosmet', 'parfum', 'beauté', 'maquillage', 'soin']
        mask_cosmetic_description = df['DESCRIPTION_COMMERCIALE'].str.contains(
            '|'.join(cosmetic_keywords), case=False, na=False
        )
        
        # Valeur unitaire élevée + description cosmétique = glissement suspect
        seuil_cosmetique_luxe = df['VALEUR_UNITAIRE_KG'].quantile(0.90)
        mask_cosmetique_luxe = df['VALEUR_UNITAIRE_KG'] > seuil_cosmetique_luxe
        
        df.loc[mask_medicaments & mask_cosmetic_description & mask_cosmetique_luxe, 'FRAUD_FLAG'] = 1
        
        return df
    
    def create_business_features(self, df):
        """Création des features business optimisées"""
        
        # 1. FEATURES GLISSEMENT TARIFAIRE (les plus importantes)
        df['BUSINESS_GLISSEMENT_TARIFAIRE'] = (
            ~df['CODE_PRODUIT_STR'].str.startswith('30', na=False)
        ).astype(int)
        
        # 2. Détection "glissement" dans la description
        if 'DESCRIPTION_COMMERCIALE' in df.columns:
            df['BUSINESS_GLISSEMENT_DESCRIPTION'] = df['DESCRIPTION_COMMERCIALE'].str.contains(
                'glissement|cosmet|parfum|beauté|maquillage|soin|toilette', 
                case=False, 
                na=False
            ).astype(int)
        else:
            df['BUSINESS_GLISSEMENT_DESCRIPTION'] = 0
        
        # 3. FEATURES RISQUE PAYS (contrefaçon)
        high_risk_countries = ['IN', 'CN', 'PK', 'BD', 'LK']
        df['BUSINESS_RISK_PAYS_HIGH'] = df['PAYS_ORIGINE_STR'].isin(high_risk_countries).astype(int)
        
        # 4. FEATURES VALEUR (volumes suspects)
        df['BUSINESS_VALEUR_ELEVEE'] = (df['VALEUR_CAF'] > df['VALEUR_CAF'].quantile(0.9)).astype(int)
        df['BUSINESS_VALEUR_EXCEPTIONNELLE'] = (df['VALEUR_CAF'] > df['VALEUR_CAF'].quantile(0.95)).astype(int)
        
        # 5. FEATURES MÉDICAMENTS SPÉCIFIQUES
        df['BUSINESS_IS_ANTIPALUDEEN'] = df['CODE_PRODUIT_STR'].str.contains('300360|300460', na=False).astype(int)
        
        return df
```

**Chapitres 84 et 85 - Machines et Appareils Électriques :**

```python
class Chap84PreprocessorComprehensive:
    def create_advanced_fraud_flag(self, df):
        """Fraud detection spécifique aux machines et appareils mécaniques"""
        
        df['FRAUD_FLAG'] = 0
        
        # 1. GLISSEMENT TARIFAIRE - Machines classées comme électroniques
        mask_machines = df['CODE_PRODUIT_STR'].str.startswith('84', na=False)
        
        # Détection des équipements électroniques mal classés
        electronic_keywords = ['électronique', 'digital', 'smart', 'intelligent']
        mask_electronic_description = df['DESCRIPTION_COMMERCIALE'].str.contains(
            '|'.join(electronic_keywords), case=False, na=False
        )
        
        # Valeur unitaire élevée + description électronique = glissement suspect
        seuil_electronic_luxe = df['VALEUR_UNITAIRE_KG'].quantile(0.90)
        mask_electronic_luxe = df['VALEUR_UNITAIRE_KG'] > seuil_electronic_luxe
        
        df.loc[mask_machines & mask_electronic_description & mask_electronic_luxe, 'FRAUD_FLAG'] = 1
        
        # 2. DÉTECTION TÉLÉPHONES MAL CLASSÉS
        phone_keywords = ['téléphone', 'mobile', 'smartphone', 'iphone', 'samsung']
        mask_phone_description = df['DESCRIPTION_COMMERCIALE'].str.contains(
            '|'.join(phone_keywords), case=False, na=False
        )
        
        # Téléphones classés en 84 au lieu de 85
        df.loc[mask_phone_description & mask_machines, 'FRAUD_FLAG'] = 1
        
        return df

class Chap85PreprocessorComprehensive:
    def create_advanced_fraud_flag(self, df):
        """Fraud detection spécifique aux appareils électriques"""
        
        df['FRAUD_FLAG'] = 0
        
        # 1. GLISSEMENT TARIFAIRE - Électronique classée comme machines
        mask_electronic = df['CODE_PRODUIT_STR'].str.startswith('85', na=False)
        
        # Détection des machines mal classées comme électroniques
        machine_keywords = ['machine', 'moteur', 'mécanique', 'outil']
        mask_machine_description = df['DESCRIPTION_COMMERCIALE'].str.contains(
            '|'.join(machine_keywords), case=False, na=False
        )
        
        df.loc[mask_electronic & mask_machine_description, 'FRAUD_FLAG'] = 1
        
        # 2. DÉTECTION ÉQUIPEMENTS DE LUXE SOUS-ÉVALUÉS
        luxury_keywords = ['luxe', 'premium', 'haut de gamme', 'professionnel']
        mask_luxury_description = df['DESCRIPTION_COMMERCIALE'].str.contains(
            '|'.join(luxury_keywords), case=False, na=False
        )
        
        # Valeur unitaire faible + description luxe = sous-évaluation
        seuil_luxury_min = df['VALEUR_UNITAIRE_KG'].quantile(0.10)
        mask_luxury_undervalued = df['VALEUR_UNITAIRE_KG'] < seuil_luxury_min
        
        df.loc[mask_luxury_description & mask_luxury_undervalued, 'FRAUD_FLAG'] = 1
        
        return df
```

### 4.1.3 Module OCR et Extraction

**Implémentation du service OCR :**

```python
class OCRService:
    def __init__(self):
        self.tesseract_config = r'--oem 3 --psm 6'
        self.preprocessing_pipeline = self._setup_preprocessing()
        self.field_extractors = self._setup_field_extractors()
    
    def _setup_preprocessing(self):
        """Configuration du pipeline de préprocessing"""
        return [
            ('denoise', self._denoise_image),
            ('deskew', self._deskew_image),
            ('enhance', self._enhance_contrast),
            ('binarize', self._binarize_image)
        ]
    
    def _setup_field_extractors(self):
        """Configuration des extracteurs de champs"""
        return {
            'numero_declaration': self._extract_declaration_number,
            'date_declaration': self._extract_declaration_date,
            'nom_importateur': self._extract_importer_name,
            'valeur_declaree': self._extract_declared_value,
            'pays_origine': self._extract_origin_country,
            'code_sh': self._extract_sh_code,
            'description_marchandises': self._extract_goods_description
        }
    
    def process_document(self, file_path: str) -> dict:
        """Traitement complet d'un document"""
        try:
            # 1. Chargement et préprocessing de l'image
            image = self._load_image(file_path)
            processed_image = self._apply_preprocessing(image)
            
            # 2. OCR avec Tesseract
            text = self._extract_text(processed_image)
            
            # 3. Extraction des champs structurés
            extracted_fields = self._extract_structured_fields(text)
            
            # 4. Validation et correction
            validated_fields = self._validate_and_correct(extracted_fields)
            
            return {
                'success': True,
                'data': validated_fields,
                'raw_text': text,
                'confidence': self._calculate_confidence(validated_fields)
            }
            
        except Exception as e:
            logger.error(f"Erreur OCR: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': {}
            }
    
    def _extract_text(self, image) -> str:
        """Extraction du texte avec Tesseract"""
        try:
            # Configuration Tesseract optimisée pour les documents douaniers
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,-()/ '
            
            text = pytesseract.image_to_string(
                image, 
                config=custom_config,
                lang='fra+eng'  # Français et anglais
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Erreur extraction texte: {e}")
            raise
    
    def _extract_structured_fields(self, text: str) -> dict:
        """Extraction des champs structurés du texte"""
        fields = {}
        
        for field_name, extractor in self.field_extractors.items():
            try:
                fields[field_name] = extractor(text)
            except Exception as e:
                logger.warning(f"Erreur extraction {field_name}: {e}")
                fields[field_name] = None
        
        return fields
    
    def _extract_declaration_number(self, text: str) -> str:
        """Extraction du numéro de déclaration"""
        # Pattern pour numéro de déclaration (ex: 2024-001234)
        pattern = r'\b\d{4}-\d{6}\b'
        match = re.search(pattern, text)
        return match.group() if match else None
    
    def _extract_declared_value(self, text: str) -> float:
        """Extraction de la valeur déclarée"""
        # Pattern pour montants (ex: 1,234,567.89 FCFA)
        pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:FCFA|CFA|€|USD)?'
        matches = re.findall(pattern, text)
        
        if matches:
            # Prendre le montant le plus élevé (probablement la valeur déclarée)
            values = [float(match.replace(',', '')) for match in matches]
            return max(values)
        
        return None
```

### 4.1.4 Module de Validation et Normalisation

**Service de validation des données :**

```python
class DataValidationService:
    def __init__(self):
        self.validation_rules = self._setup_validation_rules()
        self.normalization_rules = self._setup_normalization_rules()
    
    def _setup_validation_rules(self):
        """Configuration des règles de validation"""
        return {
            'numero_declaration': {
                'required': True,
                'pattern': r'^\d{4}-\d{6}$',
                'message': 'Format invalide: YYYY-NNNNNN'
            },
            'date_declaration': {
                'required': True,
                'format': '%Y-%m-%d',
                'message': 'Format de date invalide'
            },
            'valeur_declaree': {
                'required': True,
                'min_value': 0.01,
                'max_value': 1000000000,
                'message': 'Valeur doit être entre 0.01 et 1,000,000,000'
            },
            'code_sh': {
                'required': True,
                'pattern': r'^\d{6}$',
                'message': 'Code SH doit contenir 6 chiffres'
            },
            'pays_origine': {
                'required': True,
                'allowed_values': self._get_country_codes(),
                'message': 'Code pays invalide'
            }
        }
    
    def validate_declaration_data(self, data: dict) -> dict:
        """Validation complète des données de déclaration"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'corrected_data': data.copy()
        }
        
        for field, rules in self.validation_rules.items():
            field_value = data.get(field)
            
            # Vérification des champs obligatoires
            if rules.get('required') and not field_value:
                validation_result['errors'].append(f"{field}: Champ obligatoire manquant")
                validation_result['is_valid'] = False
                continue
            
            if field_value is not None:
                # Validation du format
                if 'pattern' in rules:
                    if not re.match(rules['pattern'], str(field_value)):
                        validation_result['errors'].append(f"{field}: {rules['message']}")
                        validation_result['is_valid'] = False
                
                # Validation des valeurs numériques
                if 'min_value' in rules or 'max_value' in rules:
                    try:
                        numeric_value = float(field_value)
                        if 'min_value' in rules and numeric_value < rules['min_value']:
                            validation_result['errors'].append(f"{field}: Valeur trop faible")
                            validation_result['is_valid'] = False
                        if 'max_value' in rules and numeric_value > rules['max_value']:
                            validation_result['errors'].append(f"{field}: Valeur trop élevée")
                            validation_result['is_valid'] = False
                    except ValueError:
                        validation_result['errors'].append(f"{field}: Valeur numérique invalide")
                        validation_result['is_valid'] = False
                
                # Validation des valeurs autorisées
                if 'allowed_values' in rules:
                    if field_value not in rules['allowed_values']:
                        validation_result['warnings'].append(f"{field}: Valeur non standard")
        
        return validation_result
    
    def normalize_data(self, data: dict) -> dict:
        """Normalisation des données"""
        normalized_data = data.copy()
        
        for field, rules in self.normalization_rules.items():
            if field in normalized_data:
                normalized_data[field] = rules(normalized_data[field])
        
        return normalized_data
```

### 4.1.5 Module d'Extraction de Features

**Service d'extraction de features métier :**

```python
class BusinessFeatureExtractor:
    def __init__(self):
        self.market_data_provider = MarketDataProvider()
        self.risk_database = RiskDatabase()
        self.historical_data_provider = HistoricalDataProvider()
        self.feature_calculators = self._setup_feature_calculators()
    
    def _setup_feature_calculators(self):
        """Configuration des calculateurs de features"""
        return {
            'tariff_features': self._calculate_tariff_features,
            'consistency_features': self._calculate_consistency_features,
            'risk_features': self._calculate_risk_features,
            'temporal_features': self._calculate_temporal_features,
            'statistical_features': self._calculate_statistical_features
        }
    
    def extract_features(self, declaration_data: dict) -> dict:
        """Extraction complète des features"""
        features = {}
        
        for feature_type, calculator in self.feature_calculators.items():
            try:
                features[feature_type] = calculator(declaration_data)
            except Exception as e:
                logger.error(f"Erreur extraction {feature_type}: {e}")
                features[feature_type] = {}
        
        # Flattening des features pour les modèles ML
        flattened_features = self._flatten_features(features)
        
        return {
            'structured_features': features,
            'flattened_features': flattened_features,
            'feature_count': len(flattened_features)
        }
    
    def _calculate_tariff_features(self, data: dict) -> dict:
        """Calcul des features tarifaires"""
        features = {}
        
        # Prix unitaire
        if data.get('valeur_declaree') and data.get('quantite'):
            features['prix_unitaire'] = data['valeur_declaree'] / data['quantite']
        
        # Comparaison avec le marché
        market_price = self.market_data_provider.get_price(data.get('code_sh'))
        if market_price and features.get('prix_unitaire'):
            features['prix_vs_marche'] = features['prix_unitaire'] / market_price
            features['ecart_prix_marche'] = abs(features['prix_vs_marche'] - 1.0)
        
        # Taux de droits
        features['taux_droits'] = data.get('taux_droits', 0.0)
        
        # Classification tarifaire
        code_sh = data.get('code_sh', '')
        features['chapitre_tarifaire'] = code_sh[:2] if len(code_sh) >= 2 else '00'
        features['position_tarifaire'] = code_sh[:4] if len(code_sh) >= 4 else '0000'
        
        return features
    
    def _calculate_consistency_features(self, data: dict) -> dict:
        """Calcul des features de cohérence"""
        features = {}
        
        # Risque pays d'origine
        pays_origine = data.get('pays_origine', '')
        features['pays_origine_risque'] = self.risk_database.get_country_risk(pays_origine)
        
        # Risque importateur
        nom_importateur = data.get('nom_importateur', '')
        features['importateur_risque'] = self.risk_database.get_importer_risk(nom_importateur)
        
        # Historique de fraude
        features['historique_fraude_importateur'] = self.historical_data_provider.get_fraud_history(
            nom_importateur
        )
        
        # Cohérence bureau douane
        bureau_douane = data.get('bureau_douane', '')
        features['bureau_douane_risque'] = self.risk_database.get_customs_office_risk(bureau_douane)
        
        return features
    
    def _calculate_risk_features(self, data: dict) -> dict:
        """Calcul des features de risque"""
        features = {}
        
        # Saisonnalité
        date_declaration = data.get('date_declaration')
        if date_declaration:
            features['mois_declaration'] = date_declaration.month
            features['trimestre_declaration'] = (date_declaration.month - 1) // 3 + 1
            features['saisonnalite_risque'] = self._calculate_seasonality_risk(date_declaration)
        
        # Tendance des prix
        code_sh = data.get('code_sh', '')
        if code_sh:
            features['tendance_prix'] = self._calculate_price_trend(code_sh, data.get('valeur_declaree'))
        
        # Anomalies statistiques
        features['anomalie_statistique'] = self._detect_statistical_anomaly(data)
        
        return features
    
    def _calculate_temporal_features(self, data: dict) -> dict:
        """Calcul des features temporelles"""
        features = {}
        
        date_declaration = data.get('date_declaration')
        if date_declaration:
            # Features temporelles de base
            features['jour_semaine'] = date_declaration.weekday()
            features['jour_mois'] = date_declaration.day
            features['semaine_annee'] = date_declaration.isocalendar()[1]
            
            # Features de fin de période (risque accru)
            features['fin_mois'] = 1 if date_declaration.day > 25 else 0
            features['fin_trimestre'] = 1 if date_declaration.month in [3, 6, 9, 12] and date_declaration.day > 25 else 0
            features['fin_annee'] = 1 if date_declaration.month == 12 and date_declaration.day > 20 else 0
        
        return features
    
    def _calculate_statistical_features(self, data: dict) -> dict:
        """Calcul des features statistiques"""
        features = {}
        
        # Features de distribution
        valeur_declaree = data.get('valeur_declaree', 0)
        quantite = data.get('quantite', 1)
        
        if quantite > 0:
            features['prix_unitaire'] = valeur_declaree / quantite
            features['log_prix_unitaire'] = np.log(features['prix_unitaire'] + 1)
            features['sqrt_prix_unitaire'] = np.sqrt(features['prix_unitaire'])
        
        # Features de ratio
        if valeur_declaree > 0:
            features['ratio_quantite_valeur'] = quantite / valeur_declaree
        
        return features
```

## 4.2 Implémentation des Modèles ML

### 4.2.1 Architecture des Modèles

**Classe de base pour les modèles ML :**

```python
class BaseMLModel:
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.performance_metrics = {}
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """Entraînement du modèle"""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prédiction simple"""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Prédiction avec probabilités"""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Évaluation du modèle"""
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions),
            'recall': recall_score(y, predictions),
            'f1_score': f1_score(y, predictions),
            'auc': roc_auc_score(y, probabilities[:, 1]),
            'confusion_matrix': confusion_matrix(y, predictions).tolist()
        }
        
        return metrics
    
    def save_model(self, filepath: str) -> None:
        """Sauvegarde du modèle"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'performance_metrics': self.performance_metrics,
            'model_name': self.model_name,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """Chargement du modèle"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.performance_metrics = model_data['performance_metrics']
        self.is_trained = True
```

### 4.2.2 Implémentation CatBoost (Chapitre 30)

**Modèle CatBoost pour les médicaments :**

```python
class CatBoostFraudDetector(BaseMLModel):
    def __init__(self):
        super().__init__("CatBoost_Chap30", "CatBoost")
        self.model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=8,
            loss_function='Logloss',
            eval_metric='F1',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50
        )
        
        # Features spécifiques au chapitre 30
        self.categorical_features = [
            'pays_origine', 'bureau_douane', 'type_declaration',
            'chapitre_tarifaire', 'position_tarifaire'
        ]
        
        self.feature_names = [
            # Features tarifaires
            'valeur_declaree', 'quantite', 'prix_unitaire',
            'taux_droits', 'prix_vs_marche', 'ecart_prix_marche',
            
            # Features de cohérence
            'pays_origine_risque', 'importateur_risque',
            'bureau_douane_risque', 'historique_fraude_importateur',
            
            # Features de risque
            'saisonnalite_risque', 'tendance_prix', 'anomalie_statistique',
            
            # Features temporelles
            'mois_declaration', 'trimestre_declaration',
            'jour_semaine', 'fin_mois', 'fin_trimestre',
            
            # Features statistiques
            'log_prix_unitaire', 'sqrt_prix_unitaire', 'ratio_quantite_valeur'
        ]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """Entraînement du modèle CatBoost"""
        try:
            # Préparation des données
            train_pool = Pool(
                X_train, y_train,
                cat_features=self._get_categorical_indices()
            )
            
            val_pool = None
            if X_val is not None and y_val is not None:
                val_pool = Pool(
                    X_val, y_val,
                    cat_features=self._get_categorical_indices()
                )
            
            # Entraînement
            self.model.fit(
                train_pool,
                eval_set=val_pool,
                plot=False,
                verbose=100
            )
            
            self.is_trained = True
            
            # Évaluation
            if X_val is not None and y_val is not None:
                self.performance_metrics = self.evaluate(X_val, y_val)
            else:
                self.performance_metrics = self.evaluate(X_train, y_train)
            
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Erreur entraînement CatBoost: {e}")
            raise
    
    def predict_with_shap(self, X: np.ndarray) -> tuple:
        """Prédiction avec calcul SHAP"""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné")
        
        # Prédiction standard
        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        # Calcul SHAP
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        return probabilities, shap_values
    
    def _get_categorical_indices(self) -> list:
        """Obtention des indices des features catégorielles"""
        indices = []
        for cat_feature in self.categorical_features:
            if cat_feature in self.feature_names:
                indices.append(self.feature_names.index(cat_feature))
        return indices
    
    def get_feature_importance(self) -> dict:
        """Obtention de l'importance des features"""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné")
        
        importance = self.model.get_feature_importance()
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Tri par importance décroissante
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return sorted_importance
```

### 4.2.3 Implémentation XGBoost (Chapitres 84, 85)

**Modèle XGBoost pour les machines et électronique :**

```python
class XGBoostFraudDetector(BaseMLModel):
    def __init__(self, chapter: str):
        super().__init__(f"XGBoost_Chap{chapter}", "XGBoost")
        self.chapter = chapter
        
        self.model = XGBClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=50
        )
        
        # Features spécifiques selon le chapitre
        if chapter == "84":
            self.feature_names = self._get_chapter84_features()
        elif chapter == "85":
            self.feature_names = self._get_chapter85_features()
        else:
            self.feature_names = self._get_generic_features()
    
    def _get_chapter84_features(self) -> list:
        """Features spécifiques au chapitre 84 (Machines)"""
        return [
            # Features techniques
            'valeur_declaree', 'poids_net', 'poids_brut',
            'prix_unitaire', 'prix_vs_marche', 'ecart_prix_marche',
            
            # Features de cohérence
            'pays_origine_risque', 'importateur_risque',
            'bureau_douane_risque', 'historique_fraude_importateur',
            
            # Features spécifiques machines
            'classification_technique', 'marque_modele_risque',
            'poids_vs_valeur', 'technologie_risque',
            
            # Features temporelles
            'mois_declaration', 'trimestre_declaration',
            'jour_semaine', 'fin_mois', 'fin_trimestre',
            
            # Features statistiques
            'log_prix_unitaire', 'sqrt_prix_unitaire', 'ratio_quantite_valeur'
        ]
    
    def _get_chapter85_features(self) -> list:
        """Features spécifiques au chapitre 85 (Électronique)"""
        return [
            # Features électroniques
            'valeur_declaree', 'prix_unitaire', 'prix_vs_marche',
            'ecart_prix_marche', 'technologie_risque', 'generation_risque',
            
            # Features de cohérence
            'pays_origine_risque', 'importateur_risque',
            'bureau_douane_risque', 'historique_fraude_importateur',
            
            # Features spécifiques électronique
            'contrefacon_risque', 'reexportation_risque',
            'marque_risque', 'compatibilite_marche',
            
            # Features temporelles
            'mois_declaration', 'trimestre_declaration',
            'jour_semaine', 'fin_mois', 'fin_trimestre',
            
            # Features statistiques
            'log_prix_unitaire', 'sqrt_prix_unitaire', 'ratio_quantite_valeur'
        ]
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """Entraînement du modèle XGBoost"""
        try:
            # Préparation des données
            eval_set = []
            if X_val is not None and y_val is not None:
                eval_set = [(X_val, y_val)]
            
            # Entraînement
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=100
            )
            
            self.is_trained = True
            
            # Évaluation
            if X_val is not None and y_val is not None:
                self.performance_metrics = self.evaluate(X_val, y_val)
            else:
                self.performance_metrics = self.evaluate(X_train, y_train)
            
            return self.performance_metrics
            
        except Exception as e:
            logger.error(f"Erreur entraînement XGBoost: {e}")
            raise
    
    def predict_with_shap(self, X: np.ndarray) -> tuple:
        """Prédiction avec calcul SHAP"""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné")
        
        # Prédiction standard
        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        # Calcul SHAP
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        return probabilities, shap_values
    
    def get_feature_importance(self) -> dict:
        """Obtention de l'importance des features"""
        if not self.is_trained:
            raise ValueError("Modèle non entraîné")
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Tri par importance décroissante
        sorted_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return sorted_importance
```

### 4.2.4 Gestionnaire de Modèles

**Service de gestion des modèles :**

```python
class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_configs = self._load_model_configs()
        self.model_paths = self._setup_model_paths()
    
    def _load_model_configs(self) -> dict:
        """Chargement des configurations des modèles"""
        return {
            'chap30': {
                'model_class': CatBoostFraudDetector,
                'features': 25,
                'target_accuracy': 0.98
            },
            'chap84': {
                'model_class': XGBoostFraudDetector,
                'features': 28,
                'target_accuracy': 0.98
            },
            'chap85': {
                'model_class': XGBoostFraudDetector,
                'features': 26,
                'target_accuracy': 0.98
            }
        }
    
    def _setup_model_paths(self) -> dict:
        """Configuration des chemins des modèles"""
        base_path = "models/"
        return {
            'chap30': f"{base_path}catboost_chap30.pkl",
            'chap84': f"{base_path}xgboost_chap84.pkl",
            'chap85': f"{base_path}xgboost_chap85.pkl"
        }
    
    def load_models(self) -> None:
        """Chargement de tous les modèles"""
        for chapter, config in self.model_configs.items():
            try:
                model_path = self.model_paths[chapter]
                if os.path.exists(model_path):
                    model_class = config['model_class']
                    if chapter in ['chap84', 'chap85']:
                        model = model_class(chapter[-2:])  # "84" ou "85"
                    else:
                        model = model_class()
                    
                    model.load_model(model_path)
                    self.models[chapter] = model
                    logger.info(f"Modèle {chapter} chargé avec succès")
                else:
                    logger.warning(f"Modèle {chapter} non trouvé: {model_path}")
                    
            except Exception as e:
                logger.error(f"Erreur chargement modèle {chapter}: {e}")
    
    def get_model(self, chapter: str) -> BaseMLModel:
        """Obtention d'un modèle par chapitre"""
        if chapter not in self.models:
            raise ValueError(f"Modèle non disponible pour le chapitre {chapter}")
        return self.models[chapter]
    
    def predict(self, chapter: str, features: np.ndarray) -> dict:
        """Prédiction avec un modèle spécifique"""
        model = self.get_model(chapter)
        
        # Vérification de la compatibilité des features
        if len(features[0]) != len(model.feature_names):
            raise ValueError(f"Nombre de features incompatible: {len(features[0])} vs {len(model.feature_names)}")
        
        # Prédiction
        probabilities, shap_values = model.predict_with_shap(features)
        
        return {
            'probabilities': probabilities[0],
            'shap_values': shap_values[0],
            'model_name': model.model_name,
            'feature_names': model.feature_names
        }
    
    def retrain_model(self, chapter: str, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """Retraining d'un modèle"""
        try:
            config = self.model_configs[chapter]
            model_class = config['model_class']
            
            # Création d'une nouvelle instance
            if chapter in ['chap84', 'chap85']:
                model = model_class(chapter[-2:])
            else:
                model = model_class()
            
            # Entraînement
            performance = model.train(X_train, y_train, X_val, y_val)
            
            # Sauvegarde
            model_path = self.model_paths[chapter]
            model.save_model(model_path)
            
            # Mise à jour du modèle en mémoire
            self.models[chapter] = model
            
            logger.info(f"Modèle {chapter} retrainé avec succès")
            return performance
            
        except Exception as e:
            logger.error(f"Erreur retraining modèle {chapter}: {e}")
            raise
```

## 4.3 Système d'Apprentissage par Renforcement

### 4.3.1 Environnement RL

**Implémentation de l'environnement RL :**

```python
class FraudDetectionRLEnv:
    def __init__(self, ml_model: BaseMLModel, threshold_range: tuple = (0.1, 0.9)):
        self.ml_model = ml_model
        self.threshold_range = threshold_range
        self.current_threshold = 0.5
        self.performance_history = []
        self.feedback_history = []
        self.state_size = 10  # Taille de l'état
        self.action_size = 3  # Actions: diminuer, maintenir, augmenter
        
        # Paramètres de l'environnement
        self.max_episodes = 1000
        self.current_episode = 0
        self.episode_rewards = []
        
    def reset(self) -> np.ndarray:
        """Réinitialisation de l'environnement"""
        self.current_episode = 0
        self.episode_rewards = []
        self.current_threshold = 0.5
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """Obtention de l'état actuel"""
        state = np.zeros(self.state_size)
        
        # État actuel du seuil
        state[0] = self.current_threshold
        
        # Performance récente
        if len(self.performance_history) > 0:
            state[1] = np.mean(self.performance_history[-10:])  # Moyenne des 10 dernières
            state[2] = np.std(self.performance_history[-10:])   # Écart-type
        
        # Ratio de feedback
        if len(self.feedback_history) > 0:
            positive_feedback = sum(1 for f in self.feedback_history[-20:] if f > 0)
            state[3] = positive_feedback / len(self.feedback_history[-20:])
        
        # Score de drift
        state[4] = self.get_drift_score()
        
        # Temps depuis dernière mise à jour
        state[5] = self.get_time_since_update()
        
        # Tendance des performances
        if len(self.performance_history) > 5:
            recent_trend = np.polyfit(range(5), self.performance_history[-5:], 1)[0]
            state[6] = recent_trend
        
        # Stabilité du seuil
        state[7] = self.get_threshold_stability()
        
        # Charge de travail
        state[8] = self.get_workload()
        
        # Saisonnalité
        state[9] = self.get_seasonality_factor()
        
        return state
    
    def step(self, action: int) -> tuple:
        """Exécution d'une action"""
        # Conversion de l'action en ajustement de seuil
        if action == 0:  # Diminuer
            threshold_adjustment = -0.05
        elif action == 1:  # Maintenir
            threshold_adjustment = 0.0
        else:  # Augmenter
            threshold_adjustment = 0.05
        
        # Application de l'ajustement
        new_threshold = self.current_threshold + threshold_adjustment
        new_threshold = np.clip(new_threshold, *self.threshold_range)
        
        # Calcul de la récompense
        reward = self.calculate_reward(new_threshold)
        
        # Mise à jour de l'état
        self.current_threshold = new_threshold
        self.performance_history.append(reward)
        self.current_episode += 1
        
        # État suivant
        next_state = self.get_state()
        
        # Vérification de la fin d'épisode
        done = self.is_episode_done()
        
        return next_state, reward, done, {}
    
    def calculate_reward(self, threshold: float) -> float:
        """Calcul de la récompense"""
        # Performance ML avec le nouveau seuil
        ml_performance = self.ml_model.evaluate_with_threshold(threshold)
        
        # Feedback utilisateur récent
        user_feedback = self.get_recent_user_feedback()
        
        # Stabilité du seuil (éviter les oscillations)
        stability_penalty = -abs(threshold - 0.5) * 0.1
        
        # Récompense pour l'exploration
        exploration_bonus = 0.01 if np.random.random() < 0.1 else 0.0
        
        # Récompense combinée
        reward = (
            ml_performance * 0.6 +      # Performance ML (60%)
            user_feedback * 0.3 +       # Feedback utilisateur (30%)
            stability_penalty +         # Stabilité (10%)
            exploration_bonus           # Exploration (bonus)
        )
        
        return reward
    
    def get_recent_user_feedback(self) -> float:
        """Obtention du feedback utilisateur récent"""
        if len(self.feedback_history) == 0:
            return 0.0
        
        # Moyenne pondérée des feedbacks récents
        recent_feedback = self.feedback_history[-10:]
        weights = np.exp(np.linspace(-1, 0, len(recent_feedback)))
        weighted_feedback = np.average(recent_feedback, weights=weights)
        
        return weighted_feedback
    
    def get_drift_score(self) -> float:
        """Calcul du score de drift"""
        if len(self.performance_history) < 20:
            return 0.0
        
        # Comparaison des performances récentes vs historiques
        recent_performance = np.mean(self.performance_history[-10:])
        historical_performance = np.mean(self.performance_history[-20:-10])
        
        drift_score = abs(recent_performance - historical_performance)
        return drift_score
    
    def get_time_since_update(self) -> float:
        """Temps depuis la dernière mise à jour"""
        if len(self.performance_history) == 0:
            return 0.0
        
        # Normalisation du temps (0-1)
        time_factor = min(self.current_episode / 100, 1.0)
        return time_factor
    
    def get_threshold_stability(self) -> float:
        """Mesure de la stabilité du seuil"""
        if len(self.performance_history) < 5:
            return 0.0
        
        # Variance des ajustements récents
        recent_changes = np.diff(self.performance_history[-5:])
        stability = 1.0 / (1.0 + np.var(recent_changes))
        
        return stability
    
    def get_workload(self) -> float:
        """Charge de travail actuelle"""
        # Simulation basée sur l'heure et le jour
        now = datetime.now()
        hour_factor = np.sin(2 * np.pi * now.hour / 24)
        day_factor = 1.0 if now.weekday() < 5 else 0.5  # Moins de travail le weekend
        
        workload = (hour_factor + 1) / 2 * day_factor
        return workload
    
    def get_seasonality_factor(self) -> float:
        """Facteur de saisonnalité"""
        now = datetime.now()
        month_factor = np.sin(2 * np.pi * now.month / 12)
        return (month_factor + 1) / 2
    
    def is_episode_done(self) -> bool:
        """Vérification de la fin d'épisode"""
        return (
            self.current_episode >= self.max_episodes or
            len(self.performance_history) > 50 and np.std(self.performance_history[-20:]) < 0.01
        )
    
    def add_feedback(self, feedback: float) -> None:
        """Ajout de feedback utilisateur"""
        self.feedback_history.append(feedback)
        
        # Limitation de l'historique
        if len(self.feedback_history) > 100:
            self.feedback_history = self.feedback_history[-100:]
```

### 4.3.2 Agent Q-Learning

**Implémentation de l'agent Q-Learning :**

```python
class QLearningAgent:
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.1, 
                 gamma: float = 0.95, epsilon: float = 1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Paramètres d'exploration
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Table Q (état discretisé)
        self.q_table = np.zeros((state_size, action_size))
        
        # Mémoire pour l'apprentissage par batch
        self.memory = []
        self.batch_size = 32
        
        # Statistiques d'apprentissage
        self.episode_rewards = []
        self.episode_losses = []
        
    def discretize_state(self, state: np.ndarray) -> int:
        """Discrétisation de l'état continu"""
        # Discretisation simple en bins
        discretized = []
        for i, value in enumerate(state):
            # Normalisation et discrétisation
            normalized = (value - np.min(state)) / (np.max(state) - np.min(state) + 1e-8)
            bin_index = int(normalized * 9)  # 10 bins (0-9)
            discretized.append(bin_index)
        
        # Conversion en index unique
        state_index = 0
        for i, bin_val in enumerate(discretized):
            state_index += bin_val * (10 ** i)
        
        # Limitation de l'index
        state_index = min(state_index, self.state_size - 1)
        return state_index
    
    def act(self, state: np.ndarray) -> int:
        """Sélection d'une action selon la politique ε-greedy"""
        state_index = self.discretize_state(state)
        
        if np.random.random() <= self.epsilon:
            # Exploration
            action = np.random.choice(self.action_size)
        else:
            # Exploitation
            action = np.argmax(self.q_table[state_index])
        
        return action
    
    def learn(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> None:
        """Apprentissage Q-Learning"""
        state_index = self.discretize_state(state)
        next_state_index = self.discretize_state(next_state)
        
        # Valeur Q actuelle
        current_q = self.q_table[state_index, action]
        
        # Valeur Q maximale pour l'état suivant
        if done:
            next_max_q = 0
        else:
            next_max_q = np.max(self.q_table[next_state_index])
        
        # Nouvelle valeur Q (équation de Bellman)
        new_q = current_q + self.learning_rate * (
            reward + self.gamma * next_max_q - current_q
        )
        
        # Mise à jour de la table Q
        self.q_table[state_index, action] = new_q
        
        # Décroissance de l'exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def learn_batch(self, experiences: list) -> float:
        """Apprentissage par batch"""
        if len(experiences) < self.batch_size:
            return 0.0
        
        # Sélection d'un batch aléatoire
        batch = random.sample(experiences, self.batch_size)
        
        total_loss = 0.0
        for state, action, reward, next_state, done in batch:
            state_index = self.discretize_state(state)
            next_state_index = self.discretize_state(next_state)
            
            # Valeur Q actuelle
            current_q = self.q_table[state_index, action]
            
            # Valeur Q cible
            if done:
                target_q = reward
            else:
                target_q = reward + self.gamma * np.max(self.q_table[next_state_index])
            
            # Calcul de la perte
            loss = (target_q - current_q) ** 2
            total_loss += loss
            
            # Mise à jour Q-Learning
            new_q = current_q + self.learning_rate * (target_q - current_q)
            self.q_table[state_index, action] = new_q
        
        return total_loss / self.batch_size
    
    def save_model(self, filepath: str) -> None:
        """Sauvegarde du modèle"""
        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses
        }
        np.save(filepath, model_data)
    
    def load_model(self, filepath: str) -> None:
        """Chargement du modèle"""
        model_data = np.load(filepath, allow_pickle=True).item()
        self.q_table = model_data['q_table']
        self.epsilon = model_data['epsilon']
        self.learning_rate = model_data['learning_rate']
        self.gamma = model_data['gamma']
        self.episode_rewards = model_data.get('episode_rewards', [])
        self.episode_losses = model_data.get('episode_losses', [])
    
    def get_policy(self) -> np.ndarray:
        """Obtention de la politique actuelle"""
        return np.argmax(self.q_table, axis=1)
    
    def get_value_function(self) -> np.ndarray:
        """Obtention de la fonction de valeur"""
        return np.max(self.q_table, axis=1)
```

### 4.3.3 Entraînement RL

**Service d'entraînement RL :**

```python
class RLTrainer:
    def __init__(self, env: FraudDetectionRLEnv, agent: QLearningAgent):
        self.env = env
        self.agent = agent
        self.training_history = []
        
    def train(self, num_episodes: int = 1000, save_interval: int = 100) -> dict:
        """Entraînement de l'agent RL"""
        logger.info(f"Début de l'entraînement RL pour {num_episodes} épisodes")
        
        episode_rewards = []
        episode_losses = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            step_count = 0
            
            while True:
                # Sélection d'action
                action = self.agent.act(state)
                
                # Exécution de l'action
                next_state, reward, done, _ = self.env.step(action)
                
                # Apprentissage
                self.agent.learn(state, action, reward, next_state, done)
                
                # Mise à jour des statistiques
                episode_reward += reward
                step_count += 1
                
                # Transition vers l'état suivant
                state = next_state
                
                if done:
                    break
            
            # Enregistrement des statistiques
            episode_rewards.append(episode_reward)
            self.agent.episode_rewards.append(episode_reward)
            
            # Sauvegarde périodique
            if episode % save_interval == 0:
                self._save_checkpoint(episode)
                logger.info(f"Épisode {episode}: Reward={episode_reward:.3f}, Epsilon={self.agent.epsilon:.3f}")
        
        # Sauvegarde finale
        self._save_final_model()
        
        training_stats = {
            'episode_rewards': episode_rewards,
            'final_epsilon': self.agent.epsilon,
            'convergence_episode': self._find_convergence_episode(episode_rewards),
            'average_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        }
        
        logger.info(f"Entraînement terminé. Reward moyen final: {training_stats['average_reward']:.3f}")
        return training_stats
    
    def _save_checkpoint(self, episode: int) -> None:
        """Sauvegarde d'un checkpoint"""
        checkpoint_path = f"models/rl_checkpoint_episode_{episode}.npy"
        self.agent.save_model(checkpoint_path)

---

## 4.6 Interface Utilisateur (Flutter)

### 4.6.1 Architecture Frontend

**Développement de l'interface :**
- **Framework** : Flutter 3.x avec Dart
- **Architecture** : StatefulWidget avec ChangeNotifier
- **Navigation** : Named routes avec gestion des permissions par rôle
- **Services** : CompleteBackendService (98 endpoints backend centralisés)
- **État** : Gestion locale + rafraîchissement automatique (30s)
- **Thème** : Design system institutionnel (couleurs douanes sénégalaises)

### 4.6.2 Écrans par Rôle

**Inspecteur (`inspecteur`) :**
- **LoginScreen** : Authentification avec identifiants prédéfinis (`inspecteur` / `inspecteur123`)
- **HomeScreen** : Tableau de bord adaptatif avec actions contextuelles
- **UploadScreen** : Upload multi-formats (CSV, PDF, images) avec agrégation automatique et affichage SHAP
- **FraudAnalyticsScreen** : Analytics de fraude avec métriques et tendances
- **FeedbackScreen** : Saisie de feedback d'inspection avec confiance
- **PVScreen** : Génération de procès-verbaux
- **PVListScreen** : Liste des PV générés
- **PVDetailScreen** : Détails d'un PV spécifique

**Expert ML (`expert_ml`) :**
- **Toutes les pages Inspecteur +**
- **MLDashboardScreen** : Surveillance avancée des modèles ML avec détection de drift et recommandations de réentraînement
- **RLAnalyticsScreen** : Analytics du système RL avec profils d'inspecteurs et performance des bandits
- **BackendTestScreen** : Tests complets du système backend
- **PostgreSQLTestScreen** : Tests d'intégration base de données

**Chef de Service (`chef_service`) :**
- **DashboardScreen** : Vue d'ensemble avec KPIs globaux et tendances de fraude
- **FraudAnalyticsScreen** : Analytics de fraude pour supervision
- **BackendTestScreen** : Tests du système
- **PostgreSQLTestScreen** : Tests base de données

### 4.6.3 Services et Communication

**CompleteBackendService :**
- **98 endpoints backend centralisés** :
  - Router Principal (/predict) : 84 endpoints
  - ML Router (/ml) : 7 endpoints
  - PostgreSQL Router (/api/v2) : 7 endpoints
- **Upload avec sauvegarde PostgreSQL** : Support multipart avec persistance
- **Gestion d'erreurs** : Retry automatique et messages contextuels

**UserSessionService :**
- **Gestion des sessions** : Authentification et permissions
- **Stockage local** : SharedPreferences pour persistance
- **Rechargement automatique** : Mise à jour des permissions

### 4.6.4 Design System

**AppColors :**
- **Couleurs institutionnelles** : Vert douanes (0xFF2E7D32), Jaune doré (0xFFFFD700), Rouge discret (0xFFD32F2F)
- **Couleurs par chapitre** : Bleu pharmaceutique, Marron machines, Gris-bleu électrique
- **Couleurs d'état** : Vert conforme, Orange attention, Rouge fraude, Bleu information

**AppConfig :**
- **Configuration des chapitres** : Vraies métriques des modèles (F1, AUC, Precision, Recall)
- **Seuils optimaux** : Calculés scientifiquement par chapitre
- **Données réelles** : 487,230 déclarations (25,334 + 264,494 + 197,402)

### 4.6.5 Intégration Temps Réel

**Gestion des données :**
- **Rafraîchissement automatique** : Toutes les 30 secondes
- **Calculs réalistes** : Basés sur les données PostgreSQL
- **Animations** : Controllers pour transitions et feedback
- **Gestion d'état** : ChangeNotifier pour communication entre composants
    
    def _save_final_model(self) -> None:
        """Sauvegarde du modèle final"""
        final_path = "models/rl_agent_final.npy"
        self.agent.save_model(final_path)
    
    def _find_convergence_episode(self, rewards: list) -> int:
        """Trouve l'épisode de convergence"""
        if len(rewards) < 100:
            return len(rewards)
        
        # Calcul de la moyenne mobile
        window_size = 50
        moving_average = []
        for i in range(window_size, len(rewards)):
            moving_average.append(np.mean(rewards[i-window_size:i]))
        
        # Recherche de la convergence (variation < 1%)
        for i in range(1, len(moving_average)):
            if abs(moving_average[i] - moving_average[i-1]) / abs(moving_average[i-1]) < 0.01:
                return i + window_size
        
        return len(rewards)
    
    def evaluate(self, num_episodes: int = 100) -> dict:
        """Évaluation de l'agent entraîné"""
        logger.info(f"Évaluation de l'agent sur {num_episodes} épisodes")
        
        # Sauvegarde de l'epsilon actuel
        original_epsilon = self.agent.epsilon
        
        # Évaluation en mode exploitation pure
        self.agent.epsilon = 0.0
        
        evaluation_rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            
            while True:
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            evaluation_rewards.append(episode_reward)
        
        # Restauration de l'epsilon
        self.agent.epsilon = original_epsilon
        
        evaluation_stats = {
            'mean_reward': np.mean(evaluation_rewards),
            'std_reward': np.std(evaluation_rewards),
            'min_reward': np.min(evaluation_rewards),
            'max_reward': np.max(evaluation_rewards),
            'rewards': evaluation_rewards
        }
        
        logger.info(f"Évaluation terminée. Reward moyen: {evaluation_stats['mean_reward']:.3f}")
        return evaluation_stats
```

---

**Fin du Chapitre 4**