import streamlit as st
import os
import zipfile
from pathlib import Path
import json
import shutil
from datetime import datetime
import base64
from PIL import Image
import io

# Configuration de la page
st.set_page_config(
    page_title="InspectIA - Visualisation et Téléchargement",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chapter-card {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .chapter-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .graph-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .download-btn {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        cursor: pointer;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .download-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .file-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem;
        border-bottom: 1px solid #eee;
        background: #f8f9fa;
        margin: 0.25rem 0;
        border-radius: 5px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown("""
<div class="main-header">
    <h1>📊 InspectIA - Visualisation et Téléchargement</h1>
    <p>Explorez et téléchargez tous les résultats, analyses et graphiques des modèles ML</p>
    <p><strong>🎯 Détection de Fraude Douanière - Chapitres 30, 84, 85</strong></p>
</div>
""", unsafe_allow_html=True)

# Chemins des dossiers
BASE_DIR = Path("/Users/macbook/Desktop/inspectia_app")
RESULTS_DIR = BASE_DIR / "backend" / "results"
ANALYSIS_DIR = BASE_DIR / "backend" / "analysis"
CALIBRATION_DIR = BASE_DIR / "backend"

def get_file_size(file_path):
    """Retourne la taille d'un fichier en format lisible"""
    if not file_path.exists():
        return "N/A"
    
    size = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def create_zip_file(files, zip_name):
    """Crée un fichier ZIP avec les fichiers sélectionnés"""
    zip_path = Path(f"/tmp/{zip_name}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files:
            if file_path.exists():
                arcname = file_path.relative_to(BASE_DIR)
                zipf.write(file_path, arcname)
    
    return zip_path

def get_chapter_info(chapter):
    """Retourne les informations d'un chapitre"""
    info = {
        "chap30": {
            "name": "Chapitre 30 - Produits Pharmaceutiques",
            "description": "Modèles ML pour la détection de fraude dans les produits pharmaceutiques",
            "icon": "💊",
            "color": "#e3f2fd",
            "gradient": "linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)"
        },
        "chap84": {
            "name": "Chapitre 84 - Machines et Appareils",
            "description": "Modèles ML pour la détection de fraude dans les machines et appareils",
            "icon": "⚙️",
            "color": "#f3e5f5",
            "gradient": "linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%)"
        },
        "chap85": {
            "name": "Chapitre 85 - Machines Électriques",
            "description": "Modèles ML pour la détection de fraude dans les machines électriques",
            "icon": "⚡",
            "color": "#e8f5e8",
            "gradient": "linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%)"
        }
    }
    return info.get(chapter, {"name": chapter, "description": "", "icon": "📊", "color": "#f5f5f5", "gradient": "linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%)"})

def load_chapter_metrics(chapter_dir):
    """Charge les métriques d'un chapitre depuis le rapport JSON"""
    report_path = chapter_dir / "ml_robust_report.json"
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Nettoyer les données pour éviter les erreurs de type
                return clean_metrics_data(data)
        except Exception as e:
            st.warning(f"⚠️ Erreur lors du chargement des métriques: {e}")
            return None
        return None
    
def clean_metrics_data(data):
    """Nettoie les données de métriques pour éviter les erreurs de type"""
    if not isinstance(data, dict):
        return {}
    
    cleaned_data = {}
    
    # Nettoyer model_performance
    if "model_performance" in data:
        model_perf = data["model_performance"]
        if isinstance(model_perf, dict):
            cleaned_data["model_performance"] = model_perf
        else:
            cleaned_data["model_performance"] = {}
    
    # Nettoyer best_model
    if "best_model" in data:
        best_model = data["best_model"]
        if isinstance(best_model, str):
            cleaned_data["best_model"] = best_model
        elif isinstance(best_model, dict) and "name" in best_model:
            cleaned_data["best_model"] = best_model["name"]
        else:
            cleaned_data["best_model"] = "N/A"
    else:
        cleaned_data["best_model"] = "N/A"
    
    return cleaned_data

def display_image_with_download(image_path, title, description="", unique_key=""):
    """Affiche une image avec option de téléchargement"""
    if image_path.exists():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"### {title}")
            if description:
                st.write(description)
            
            # Afficher l'image
            try:
                image = Image.open(image_path)
                st.image(image, use_container_width=True, caption=title)
            except Exception as e:
                st.error(f"Erreur lors du chargement de l'image: {e}")
        
        with col2:
            st.markdown("**Télécharger:**")
            with open(image_path, "rb") as f:
                st.download_button(
                    label=f"⬇️ {image_path.name}",
                    data=f.read(),
                    file_name=image_path.name,
                    mime="image/png",
                    key=f"download_{unique_key}_{image_path.name}"
                )

# Sidebar pour la navigation
st.sidebar.markdown("## 🎯 Navigation")
st.sidebar.markdown("---")

view_mode = st.sidebar.selectbox(
    "Mode d'affichage",
    ["📊 Visualisation par chapitre", "📈 Analyses détaillées", "🎯 Calibration", "📦 Téléchargement complet"]
)

# Lister les chapitres disponibles
chapters = []
if RESULTS_DIR.exists():
    for chapter_dir in RESULTS_DIR.iterdir():
        if chapter_dir.is_dir() and chapter_dir.name.startswith("chap"):
            chapters.append(chapter_dir.name)

if not chapters:
    st.error("❌ Aucun chapitre trouvé dans le dossier results")
    st.stop()

# Contenu principal
if view_mode == "📊 Visualisation par chapitre":
    st.header("📊 Visualisation des Résultats par Chapitre")
    
    # Sélecteur de chapitre
    selected_chapter = st.selectbox("Sélectionnez un chapitre:", chapters)
    chapter_info = get_chapter_info(selected_chapter)
    chapter_dir = RESULTS_DIR / selected_chapter
    
    # En-tête du chapitre
st.markdown(f"""
    <div class="chapter-card" style="background: {chapter_info['gradient']};">
        <h2>{chapter_info['icon']} {chapter_info['name']}</h2>
        <p>{chapter_info['description']}</p>
</div>
""", unsafe_allow_html=True)

    # Charger les métriques
    metrics_data = load_chapter_metrics(chapter_dir)
    
    if metrics_data:
        # Afficher les métriques principales
        st.subheader("📈 Métriques Principales")
        
        model_performance = metrics_data.get("model_performance", {})
        best_model = metrics_data.get("best_model", "N/A")
        
        # S'assurer que best_model est une chaîne de caractères
        if isinstance(best_model, dict):
            best_model = best_model.get("name", "N/A")
        elif not isinstance(best_model, str):
            best_model = str(best_model)
        
        if best_model in model_performance:
            best_metrics = model_performance[best_model]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 F1-Score</h3>
                    <h2 style="color: #28a745;">{best_metrics.get('f1_score', 0):.3f}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Précision</h3>
                    <h2 style="color: #007bff;">{best_metrics.get('precision', 0):.3f}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Rappel</h3>
                    <h2 style="color: #ffc107;">{best_metrics.get('recall', 0):.3f}</h2>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 AUC</h3>
                    <h2 style="color: #dc3545;">{best_metrics.get('auc', 0):.3f}</h2>
                </div>
                """, unsafe_allow_html=True)

    # Afficher les graphiques
    st.subheader("📊 Graphiques et Analyses")
    
    # Organiser les graphiques par catégorie
    graph_categories = {
        "🎯 Métriques": [
            ("metrics_best.png", "Métriques du Meilleur Modèle", "Performance du modèle optimal"),
            ("metrics_comparison_all.png", "Comparaison de Tous les Modèles", "Comparaison des performances de tous les algorithmes"),
            ("metrics_comparison_all_algorithms.png", "Comparaison des Algorithmes", "Analyse comparative des algorithmes ML")
        ],
        "📈 Courbes ROC": [
            ("roc_curve_best.png", "Courbe ROC - Meilleur Modèle", "Courbe ROC du modèle optimal"),
            ("roc_curves_all.png", "Courbes ROC - Tous les Modèles", "Courbes ROC de tous les modèles"),
            ("roc_comparison_all_algorithms.png", "Comparaison ROC", "Comparaison des courbes ROC")
        ],
        "📊 Précision-Rappel": [
            ("precision_recall_curve_best.png", "Courbe Précision-Rappel", "Courbe Précision-Rappel du meilleur modèle"),
            ("precision_recall_curves_all.png", "Courbes Précision-Rappel", "Courbes Précision-Rappel de tous les modèles")
        ],
        "🎯 Matrices de Confusion": [
            ("confusion_matrix_best.png", "Matrice de Confusion - Meilleur", "Matrice de confusion du modèle optimal"),
            ("confusion_matrices_all.png", "Matrices de Confusion - Tous", "Matrices de confusion de tous les modèles")
        ],
        "🔍 Analyse SHAP": [
            ("shap_summary_plot_20.png", "Résumé SHAP", "Analyse de l'importance des features avec SHAP"),
            ("shap_feature_importance_20.png", "Importance des Features", "Top 20 des features les plus importantes")
        ]
    }
    
    for category, graphs in graph_categories.items():
        st.markdown(f"### {category}")
        
        for filename, title, description in graphs:
            image_path = chapter_dir / filename
            if image_path.exists():
                display_image_with_download(image_path, title, description, f"{selected_chapter}_{filename}")
            else:
                st.info(f"📄 {title} - Fichier non trouvé")

elif view_mode == "📈 Analyses détaillées":
    st.header("📈 Analyses Détaillées par Chapitre")
    
    for chapter in sorted(chapters):
        chapter_info = get_chapter_info(chapter)
        analysis_dir = ANALYSIS_DIR / chapter
        
        if analysis_dir.exists():
            st.markdown(f"""
            <div class="chapter-card" style="background: {chapter_info['gradient']};">
                <h3>{chapter_info['icon']} {chapter_info['name']} - Analyses</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Afficher les graphiques d'analyse
            analysis_files = list(analysis_dir.glob("*.png"))
            if analysis_files:
                for analysis_file in sorted(analysis_files):
                    display_image_with_download(analysis_file, analysis_file.stem.replace("_", " ").title(), "", f"analysis_{chapter}_{analysis_file.name}")
            
            # Bouton de téléchargement ZIP
            col1, col2, col3 = st.columns([2, 1, 1])
            with col2:
                if st.button(f"📦 ZIP {chapter}", key=f"analysis_zip_{chapter}"):
                    files = list(analysis_dir.glob("*"))
                    zip_path = create_zip_file(files, f"{chapter}_analysis.zip")
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            label=f"⬇️ Télécharger",
                            data=f.read(),
                            file_name=f"{chapter}_analysis.zip",
                            mime="application/zip",
                            key=f"zip_download_{chapter}"
                        )
                    zip_path.unlink()
        else:
            st.info(f"📁 Aucun dossier d'analyse trouvé pour {chapter}")

elif view_mode == "🎯 Calibration":
    st.header("🎯 Analyses de Calibration")
    
    calibration_files = [
        ("calibration_analysis_chapitre_30_REAL_VALUES.png", "Calibration Chapitre 30", "Analyse de calibration pour les produits pharmaceutiques"),
        ("calibration_analysis_chapitre_84_REAL_VALUES.png", "Calibration Chapitre 84", "Analyse de calibration pour les machines et appareils"),
        ("calibration_analysis_chapitre_85_REAL_VALUES.png", "Calibration Chapitre 85", "Analyse de calibration pour les machines électriques"),
        ("calibration_comparison_ALL_CHAPTERS_REAL_VALUES.png", "Comparaison Calibration", "Comparaison de la calibration entre tous les chapitres"),
        ("calibration_curves_ALL_CHAPTERS_REAL_VALUES.png", "Courbes de Calibration", "Courbes de calibration pour tous les chapitres"),
        ("calibration_synthesis_FINAL_REAL_VALUES.png", "Synthèse Calibration", "Synthèse finale de l'analyse de calibration")
    ]
    
    for filename, title, description in calibration_files:
        image_path = CALIBRATION_DIR / filename
        if image_path.exists():
            display_image_with_download(image_path, title, description, f"calibration_{filename}")
    
    # Bouton de téléchargement ZIP pour calibration
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("📦 ZIP Calibration", key="calibration_zip"):
            available_files = [CALIBRATION_DIR / f[0] for f in calibration_files if (CALIBRATION_DIR / f[0]).exists()]
            zip_path = create_zip_file(available_files, "calibration_analysis.zip")
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="⬇️ Télécharger Calibration",
                    data=f.read(),
                    file_name="calibration_analysis.zip",
                    mime="application/zip",
                    key="calibration_zip_download"
                )
            zip_path.unlink()

elif view_mode == "�� Téléchargement complet":
    st.header("📦 Téléchargement Complet")
    st.write("Téléchargez tous les résultats, analyses et graphiques en un seul fichier ZIP.")
    
    # Statistiques des fichiers
    all_files = []
    if RESULTS_DIR.exists():
        for chapter_dir in RESULTS_DIR.iterdir():
            if chapter_dir.is_dir():
                all_files.extend(list(chapter_dir.glob("*")))
    
    if ANALYSIS_DIR.exists():
        for chapter_dir in ANALYSIS_DIR.iterdir():
            if chapter_dir.is_dir():
                all_files.extend(list(chapter_dir.glob("*")))
    
    calibration_files = [
        "calibration_analysis_chapitre_30_REAL_VALUES.png",
        "calibration_analysis_chapitre_84_REAL_VALUES.png", 
        "calibration_analysis_chapitre_85_REAL_VALUES.png",
        "calibration_comparison_ALL_CHAPTERS_REAL_VALUES.png",
        "calibration_curves_ALL_CHAPTERS_REAL_VALUES.png",
        "calibration_synthesis_FINAL_REAL_VALUES.png"
    ]
    
    for filename in calibration_files:
        file_path = CALIBRATION_DIR / filename
        if file_path.exists():
            all_files.append(file_path)
    
    # Afficher les statistiques
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Fichiers", len(all_files))
    
    with col2:
        total_size = sum(f.stat().st_size for f in all_files if f.exists())
        size_mb = total_size / (1024 * 1024)
        st.metric("💾 Taille Totale", f"{size_mb:.1f} MB")
    
    with col3:
        st.metric("📁 Chapitres", len(chapters))
    
    with col4:
        st.metric("🎯 Modèles", "3")
    
    # Bouton de téléchargement
    if st.button("🚀 Créer Archive Complète", key="create_full_archive"):
        with st.spinner("Création de l'archive complète..."):
            if all_files:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_name = f"inspectia_complete_results_{timestamp}.zip"
                zip_path = create_zip_file(all_files, zip_name)
                
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Télécharger Archive Complète",
                        data=f.read(),
                        file_name=zip_name,
                        mime="application/zip"
                    )
                zip_path.unlink()
                
                st.success(f"✅ Archive créée avec {len(all_files)} fichiers")
            else:
                st.error("❌ Aucun fichier trouvé à archiver")

# Pied de page
st.markdown("---")
        st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>📊 InspectIA - Système de Détection de Fraude Douanière</strong></p>
    <p>Généré le {}</p>
    <p>🎯 Chapitres 30, 84, 85 | 🤖 Modèles ML | 📈 Analyses Avancées</p>
        </div>
""".format(datetime.now().strftime("%d/%m/%Y à %H:%M")), unsafe_allow_html=True)