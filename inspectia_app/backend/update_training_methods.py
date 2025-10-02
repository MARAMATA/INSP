#!/usr/bin/env python3
"""
Script pour mettre à jour les méthodes d'entraînement dans tous les chapitres
pour utiliser correctement les données de validation
"""

import re
import os

def update_training_methods(file_path):
    """Mettre à jour les méthodes d'entraînement d'un fichier"""
    print(f"🔄 Mise à jour de {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Mettre à jour la signature de train_models
    old_signature = "def train_models(self, X_train, y_train):"
    new_signature = "def train_models(self, X_train, y_train, X_valid=None, y_valid=None):"
    content = content.replace(old_signature, new_signature)
    
    # 2. Mettre à jour la signature de train_all_models
    old_signature = "def train_all_models(self, X_train, y_train, models_config):"
    new_signature = "def train_all_models(self, X_train, y_train, models_config, X_valid=None, y_valid=None):"
    content = content.replace(old_signature, new_signature)
    
    # 3. Ajouter l'early stopping dans train_models
    early_stopping_code = '''
            # Utiliser les données de validation si disponibles pour l'early stopping
            if X_valid is not None and y_valid is not None:
                logger.info(f"   📊 Utilisation des données de validation pour {model_name}")
                # Pour les modèles qui supportent l'early stopping
                if model_name in ['LightGBM', 'XGBoost', 'CatBoost']:
                    try:
                        # Réentraîner avec validation set pour early stopping
                        if model_name == 'LightGBM':
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_valid, y_valid)],
                                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                            )
                        elif model_name == 'XGBoost':
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_valid, y_valid)],
                                early_stopping_rounds=50,
                                verbose=False
                            )
                        elif model_name == 'CatBoost':
                            model.fit(
                                X_train, y_train,
                                eval_set=(X_valid, y_valid),
                                early_stopping_rounds=50,
                                verbose=False
                            )
                        
                        # Recréer la pipeline avec le modèle optimisé
                        pipeline = self.build_model_pipeline(model)
                        pipeline.fit(X_train, y_train)
                        logger.info(f"   ✅ {model_name} optimisé avec early stopping")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Early stopping échoué pour {model_name}: {e}")
            '''
    
    # Insérer l'early stopping après pipeline.fit(X_train, y_train) dans train_models
    pattern = r'(pipeline\.fit\(X_train, y_train\)\s*\n\s*# Sauvegarder le modèle)'
    replacement = r'\1' + early_stopping_code
    content = re.sub(pattern, replacement, content)
    
    # 4. Ajouter l'early stopping dans train_all_models
    early_stopping_code_all = '''
            # Utiliser les données de validation si disponibles pour l'early stopping
            if X_valid is not None and y_valid is not None:
                logger.info(f"   📊 Utilisation des données de validation pour {model_name}")
                # Pour les modèles qui supportent l'early stopping
                if model_name in ['LightGBM', 'XGBoost', 'CatBoost']:
                    try:
                        # Extraire le modèle de la pipeline
                        model = pipeline.named_steps['classifier']
                        
                        # Réentraîner avec validation set pour early stopping
                        if model_name == 'LightGBM':
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_valid, y_valid)],
                                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                            )
                        elif model_name == 'XGBoost':
                            model.fit(
                                X_train, y_train,
                                eval_set=[(X_valid, y_valid)],
                                early_stopping_rounds=50,
                                verbose=False
                            )
                        elif model_name == 'CatBoost':
                            model.fit(
                                X_train, y_train,
                                eval_set=(X_valid, y_valid),
                                early_stopping_rounds=50,
                                verbose=False
                            )
                        
                        # Recréer la pipeline avec le modèle optimisé
                        pipeline = self.build_model_pipeline(model)
                        pipeline.fit(X_train, y_train)
                        logger.info(f"   ✅ {model_name} optimisé avec early stopping")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Early stopping échoué pour {model_name}: {e}")
            '''
    
    # Insérer l'early stopping après pipeline.fit(X_train, y_train) dans train_all_models
    pattern = r'(pipeline\.fit\(X_train, y_train\)\s*\n\s*# Sauvegarder dans self\.models)'
    replacement = r'\1' + early_stopping_code_all
    content = re.sub(pattern, replacement, content)
    
    # 5. Mettre à jour les appels dans run_complete_pipeline_robust
    content = content.replace(
        "self.train_models(X_train, y_train)",
        "self.train_models(X_train, y_train, X_valid, y_valid)"
    )
    
    content = content.replace(
        "self.train_all_models(X_train, y_train, models_config)",
        "self.train_all_models(X_train, y_train, models_config, X_valid, y_valid)"
    )
    
    # 6. Mettre à jour les descriptions
    content = content.replace(
        "Entraîner tous les modèles avec calibration comme chapitres 84/85",
        "Entraîner tous les modèles avec utilisation des données de validation"
    )
    
    content = content.replace(
        "Entraîner tous les modèles (version améliorée du chap84)",
        "Entraîner tous les modèles (version améliorée du chap84) avec validation"
    )
    
    # Sauvegarder le fichier modifié
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"   ✅ {file_path} mis à jour")

def main():
    """Fonction principale"""
    print("🚀 MISE À JOUR DES MÉTHODES D'ENTRAÎNEMENT")
    print("=" * 50)
    
    chapters = [
        "src/chapters/chap84/ml_model.py",
        "src/chapters/chap85/ml_model.py"
    ]
    
    for chapter in chapters:
        if os.path.exists(chapter):
            update_training_methods(chapter)
        else:
            print(f"   ⚠️ Fichier non trouvé: {chapter}")
    
    print("\n✅ Mise à jour terminée!")

if __name__ == "__main__":
    main()
