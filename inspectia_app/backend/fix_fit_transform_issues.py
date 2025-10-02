#!/usr/bin/env python3
"""
Script pour corriger tous les problèmes de fit/transform identifiés
"""

import re
import os

def fix_early_stopping(content):
    """Corriger l'early stopping pour respecter fit/transform"""
    
    # Pattern pour trouver l'early stopping problématique
    early_stopping_pattern = r'''            # Utiliser les données de validation si disponibles pour l'early stopping
            if X_valid is not None and y_valid is not None:
                logger\.info\(f"   📊 Utilisation des données de validation pour \{model_name\}"\)
                # Pour les modèles qui supportent l'early stopping
                if model_name in \['LightGBM', 'XGBoost', 'CatBoost'\]:
                    try:
                        # Réentraîner avec validation set pour early stopping
                        if model_name == 'LightGBM':
                            model\.fit\(
                                X_train, y_train,
                                eval_set=\[\(X_valid, y_valid\)\],
                                callbacks=\[lgb\.early_stopping\(stopping_rounds=50, verbose=False\)\]
                            \)
                        elif model_name == 'XGBoost':
                            model\.fit\(
                                X_train, y_train,
                                eval_set=\[\(X_valid, y_valid\)\],
                                early_stopping_rounds=50,
                                verbose=False
                            \)
                        elif model_name == 'CatBoost':
                            model\.fit\(
                                X_train, y_train,
                                eval_set=\(X_valid, y_valid\),
                                early_stopping_rounds=50,
                                verbose=False
                            \)
                        
                        # Recréer la pipeline avec le modèle optimisé
                        pipeline = self\.build_model_pipeline\(model\)
                        pipeline\.fit\(X_train, y_train\)
                        logger\.info\(f"   ✅ \{model_name\} optimisé avec early stopping"\)
                    except Exception as e:
                        logger\.warning\(f"   ⚠️ Early stopping échoué pour \{model_name\}: \{e\}"\)'''
    
    # Remplacement correct
    early_stopping_fix = '''            # Utiliser les données de validation si disponibles pour l'early stopping
            if X_valid is not None and y_valid is not None:
                logger.info(f"   📊 Utilisation des données de validation pour {model_name}")
                # Pour les modèles qui supportent l'early stopping
                if model_name in ['LightGBM', 'XGBoost', 'CatBoost']:
                    try:
                        # 1) Fit le preprocessor sur le TRAIN uniquement
                        preprocessor = self.build_preprocessing_pipeline()
                        preprocessor.fit(X_train, y_train)
                        
                        # 2) Transforme TRAIN et VALID avec CE preprocessor figé
                        Xt_train = preprocessor.transform(X_train)
                        Xt_valid = preprocessor.transform(X_valid)
                        
                        # 3) Fit le modèle (sans pipeline) avec early stopping sur Xt_*
                        if model_name == 'LightGBM':
                            model.fit(
                                Xt_train, y_train,
                                eval_set=[(Xt_valid, y_valid)],
                                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                            )
                        elif model_name == 'XGBoost':
                            model.fit(
                                Xt_train, y_train,
                                eval_set=[(Xt_valid, y_valid)],
                                early_stopping_rounds=50,
                                verbose=False
                            )
                        elif model_name == 'CatBoost':
                            model.fit(
                                Xt_train, y_train,
                                eval_set=(Xt_valid, y_valid),
                                early_stopping_rounds=50,
                                verbose=False
                            )
                        
                        # 4) Recompose une pipeline gelée pour la prod
                        final_pipeline = Pipeline([
                            ("preprocessor", preprocessor),   # déjà fit (ne plus refit)
                            ("classifier", model)             # déjà fit (ne plus refit)
                        ])
                        pipeline = final_pipeline
                        logger.info(f"   ✅ {model_name} optimisé avec early stopping")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Early stopping échoué pour {model_name}: {e}")'''
    
    return re.sub(early_stopping_pattern, early_stopping_fix, content, flags=re.DOTALL)

def fix_early_stopping_train_all_models(content):
    """Corriger l'early stopping dans train_all_models"""
    
    # Pattern pour train_all_models
    early_stopping_pattern_all = r'''            # Utiliser les données de validation si disponibles pour l'early stopping
            if X_valid is not None and y_valid is not None:
                logger\.info\(f"   📊 Utilisation des données de validation pour \{model_name\}"\)
                # Pour les modèles qui supportent l'early stopping
                if model_name in \['LightGBM', 'XGBoost', 'CatBoost'\]:
                    try:
                        # Extraire le modèle de la pipeline
                        model = pipeline\.named_steps\['classifier'\]
                        
                        # Réentraîner avec validation set pour early stopping
                        if model_name == 'LightGBM':
                            model\.fit\(
                                X_train, y_train,
                                eval_set=\[\(X_valid, y_valid\)\],
                                callbacks=\[lgb\.early_stopping\(stopping_rounds=50, verbose=False\)\]
                            \)
                        elif model_name == 'XGBoost':
                            model\.fit\(
                                X_train, y_train,
                                eval_set=\[\(X_valid, y_valid\)\],
                                early_stopping_rounds=50,
                                verbose=False
                            \)
                        elif model_name == 'CatBoost':
                            model\.fit\(
                                X_train, y_train,
                                eval_set=\(X_valid, y_valid\),
                                early_stopping_rounds=50,
                                verbose=False
                            \)
                        
                        # Recréer la pipeline avec le modèle optimisé
                        pipeline = self\.build_model_pipeline\(model\)
                        pipeline\.fit\(X_train, y_train\)
                        logger\.info\(f"   ✅ \{model_name\} optimisé avec early stopping"\)
                    except Exception as e:
                        logger\.warning\(f"   ⚠️ Early stopping échoué pour \{model_name\}: \{e\}"\)'''
    
    # Remplacement correct
    early_stopping_fix_all = '''            # Utiliser les données de validation si disponibles pour l'early stopping
            if X_valid is not None and y_valid is not None:
                logger.info(f"   📊 Utilisation des données de validation pour {model_name}")
                # Pour les modèles qui supportent l'early stopping
                if model_name in ['LightGBM', 'XGBoost', 'CatBoost']:
                    try:
                        # 1) Fit le preprocessor sur le TRAIN uniquement
                        preprocessor = self.build_preprocessing_pipeline()
                        preprocessor.fit(X_train, y_train)
                        
                        # 2) Transforme TRAIN et VALID avec CE preprocessor figé
                        Xt_train = preprocessor.transform(X_train)
                        Xt_valid = preprocessor.transform(X_valid)
                        
                        # 3) Fit le modèle (sans pipeline) avec early stopping sur Xt_*
                        if model_name == 'LightGBM':
                            model_config.fit(
                                Xt_train, y_train,
                                eval_set=[(Xt_valid, y_valid)],
                                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                            )
                        elif model_name == 'XGBoost':
                            model_config.fit(
                                Xt_train, y_train,
                                eval_set=[(Xt_valid, y_valid)],
                                early_stopping_rounds=50,
                                verbose=False
                            )
                        elif model_name == 'CatBoost':
                            model_config.fit(
                                Xt_train, y_train,
                                eval_set=(Xt_valid, y_valid),
                                early_stopping_rounds=50,
                                verbose=False
                            )
                        
                        # 4) Recompose une pipeline gelée pour la prod
                        final_pipeline = Pipeline([
                            ("preprocessor", preprocessor),   # déjà fit (ne plus refit)
                            ("classifier", model_config)      # déjà fit (ne plus refit)
                        ])
                        pipeline = final_pipeline
                        logger.info(f"   ✅ {model_name} optimisé avec early stopping")
                    except Exception as e:
                        logger.warning(f"   ⚠️ Early stopping échoué pour {model_name}: {e}")'''
    
    return re.sub(early_stopping_pattern_all, early_stopping_fix_all, content, flags=re.DOTALL)

def fix_calibrated_models_injection(content):
    """Corriger l'injection des modèles calibrés dans self.models"""
    
    # Pattern pour trouver où les modèles calibrés ne sont pas injectés
    # Chercher dans train_all_models_calibrated
    pattern = r'(calibrated\.fit\(X_valid_processed, y_valid\)\s*\n\s*calib_time = time\.time\(\) - calib_start\s*\n\s*logger\.info\(f"      ⏱️ Calibration terminée en \{calib_time:.2f\} secondes"\))'
    
    replacement = r'''\1
                
                # IMPORTANT: Injecter le modèle calibré dans self.models
                self.models[name] = calibrated'''
    
    return re.sub(pattern, replacement, content)

def fix_manual_preprocessing(content):
    """Supprimer ou corriger les méthodes de preprocessing manuel problématiques"""
    
    # Supprimer les appels à scale_features avec un seul argument
    content = re.sub(r'scale_features\(X_train\)', 'scale_features(X_train, X_test)', content)
    
    # Corriger _save_preprocessed_data pour utiliser les données transformées
    pattern = r'_save_preprocessed_data\(X_train_processed, X_valid, X_test, y_train, y_valid, y_test\)'
    replacement = '_save_preprocessed_data(X_train_processed, X_valid_processed, X_test_processed, y_train, y_valid, y_test)'
    content = re.sub(pattern, replacement, content)
    
    return content

def fix_data_leakage_check(content):
    """Corriger validate_no_data_leakage pour utiliser seulement train"""
    
    # Pattern pour trouver validate_no_data_leakage
    pattern = r'def validate_no_data_leakage\(self, df\):'
    replacement = 'def validate_no_data_leakage(self, df_train):'
    
    content = re.sub(pattern, replacement, content)
    
    # Remplacer les références à df par df_train dans la méthode
    pattern = r'correlations = df\.corr\(\)'
    replacement = 'correlations = df_train.corr()'
    content = re.sub(pattern, replacement, content)
    
    return content

def fix_onehot_encoder(content):
    """Corriger OneHotEncoder pour compatibilité"""
    
    # Remplacer sparse_output=False par sparse=False pour compatibilité
    content = re.sub(r'sparse_output=False', 'sparse=False', content)
    
    return content

def fix_split_order_consistency(content):
    """Corriger l'ordre des splits pour cohérence"""
    
    # time_based_split devrait retourner le même ordre que split_data_robust
    pattern = r'return X_train, X_test, X_valid, y_train, y_test, y_valid'
    replacement = 'return X_train, X_valid, X_test, y_train, y_valid, y_test'
    content = re.sub(pattern, replacement, content)
    
    return content

def update_file(file_path):
    """Mettre à jour un fichier avec tous les correctifs"""
    print(f"🔄 Correction de {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Appliquer tous les correctifs
    content = fix_early_stopping(content)
    content = fix_early_stopping_train_all_models(content)
    content = fix_calibrated_models_injection(content)
    content = fix_manual_preprocessing(content)
    content = fix_data_leakage_check(content)
    content = fix_onehot_encoder(content)
    content = fix_split_order_consistency(content)
    
    # Sauvegarder
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"   ✅ {file_path} corrigé")

def main():
    """Fonction principale"""
    print("🚀 CORRECTION DES PROBLÈMES FIT/TRANSFORM")
    print("=" * 50)
    
    chapters = [
        "src/chapters/chap30/ml_model.py",
        "src/chapters/chap84/ml_model.py",
        "src/chapters/chap85/ml_model.py"
    ]
    
    for chapter in chapters:
        if os.path.exists(chapter):
            update_file(chapter)
        else:
            print(f"   ⚠️ Fichier non trouvé: {chapter}")
    
    print("\n✅ Toutes les corrections appliquées!")

if __name__ == "__main__":
    main()
