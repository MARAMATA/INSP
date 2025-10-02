#!/usr/bin/env python3
"""
Script pour réorganiser les méthodes des chapitres dans le bon ordre
"""

# Ordre de référence optimal
REFERENCE_ORDER = [
    "__init__",
    "validate_no_data_leakage", 
    "load_data",
    "_capture_preprocessor",
    "time_based_split",
    "calibrate_prefit_model",
    "_evaluate_on_split",
    "_evaluate_single_model_on_test",
    "build_preprocessing_pipeline",
    "build_model_pipeline",
    "preprocess_data",
    "split_data_robust",
    "_splits_exist",
    "_load_splits",
    "_preprocessed_data_exist",
    "_load_preprocessed_data",
    "_save_preprocessed_data",
    "_save_splits",
    "scale_features",
    "_get_optimized_models_config",
    "cross_validate_with_regularization",
    "train_models",
    "train_all_models",
    "train_all_models_calibrated",
    "generate_report_robust",
    "evaluate_models",
    "generate_plots",
    "generate_shap_analysis",
    "get_feature_importance",
    "_get_feature_names_after_preprocessing",
    "display_results_summary",
    "_generate_comparison_plots",
    "generate_best_algorithm_results",
    "model_predict",
    "load_saved_models",
    "save_models",
    "generate_report",
    "show_existing_results",
    "run_complete_pipeline_robust"
]

def extract_methods_from_file(file_path):
    """Extraire toutes les méthodes d'un fichier"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    methods = {}
    lines = content.split('\n')
    current_method = None
    current_content = []
    
    for i, line in enumerate(lines):
        if line.strip().startswith('def ') and line.startswith('    def '):
            # Sauvegarder la méthode précédente
            if current_method:
                methods[current_method] = '\n'.join(current_content)
            
            # Nouvelle méthode
            method_name = line.strip().split('(')[0].replace('def ', '')
            current_method = method_name
            current_content = [line]
        elif current_method:
            current_content.append(line)
    
    # Sauvegarder la dernière méthode
    if current_method:
        methods[current_method] = '\n'.join(current_content)
    
    return methods

def reorder_file(input_file, output_file):
    """Réorganiser un fichier selon l'ordre de référence"""
    print(f"🔄 Réorganisation de {input_file}...")
    
    # Extraire les méthodes
    methods = extract_methods_from_file(input_file)
    
    # Lire le fichier original
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Trouver le début de la classe
    lines = content.split('\n')
    class_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('class '):
            class_start = i
            break
    
    # Trouver la fin de la classe (avant les fonctions globales)
    class_end = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() and not lines[i].startswith(' ') and not lines[i].startswith('\t'):
            if not lines[i].strip().startswith('#'):
                class_end = i
                break
    
    # Extraire l'en-tête de la classe
    header_lines = lines[:class_start]
    class_header = lines[class_start:class_start + 2]  # class + docstring
    
    # Extraire les imports et la fin
    imports_and_end = lines[class_end:]
    
    # Reconstruire le fichier dans le bon ordre
    new_content = []
    new_content.extend(header_lines)
    new_content.extend(class_header)
    new_content.append('')
    
    # Ajouter les méthodes dans l'ordre de référence
    for method_name in REFERENCE_ORDER:
        if method_name in methods:
            new_content.append(methods[method_name])
            new_content.append('')
    
    # Ajouter les méthodes non référencées (au cas où)
    for method_name, method_content in methods.items():
        if method_name not in REFERENCE_ORDER:
            print(f"   ⚠️ Méthode non référencée: {method_name}")
            new_content.append(method_content)
            new_content.append('')
    
    new_content.extend(imports_and_end)
    
    # Écrire le nouveau fichier
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_content))
    
    print(f"   ✅ Fichier réorganisé: {output_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python reorder_methods.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    reorder_file(input_file, output_file)
