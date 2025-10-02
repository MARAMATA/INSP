import streamlit as st
import os
import zipfile
from pathlib import Path
import json
from datetime import datetime
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="InspectIA - Téléchargement des Résultats",
    page_icon="📊",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .chapter-card {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# En-tête
st.markdown("""
<div class="main-header">
    <h1>📊 InspectIA - Téléchargement des Résultats</h1>
    <p>Explorez et téléchargez tous les résultats, analyses et graphiques des modèles ML</p>
</div>
""", unsafe_allow_html=True)

# Chemins
BASE_DIR = Path("/Users/macbook/Desktop/inspectia_app")
RESULTS_DIR = BASE_DIR / "backend" / "results"
ANALYSIS_DIR = BASE_DIR / "backend" / "analysis"
CALIBRATION_DIR = BASE_DIR / "backend"

def get_file_size(file_path):
    """Retourne la taille d'un fichier"""
    if not file_path.exists():
        return "N/A"
    size = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"

def create_zip_file(files, zip_name):
    """Crée un fichier ZIP"""
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
        "chap30": {"name": "Chapitre 30 - Produits Pharmaceutiques", "icon": "💊"},
        "chap84": {"name": "Chapitre 84 - Machines et Appareils", "icon": "⚙️"},
        "chap85": {"name": "Chapitre 85 - Machines Électriques", "icon": "⚡"}
    }
    return info.get(chapter, {"name": chapter, "icon": "📊"})

def display_image_with_download(image_path, title, unique_key=""):
    """Affiche une image avec téléchargement"""
    if image_path.exists():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {title}")
            try:
                image = Image.open(image_path)
                st.image(image, use_container_width=True, caption=title)
            except Exception as e:
                st.error(f"Erreur: {e}")
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

# Navigation
st.sidebar.header("🎯 Navigation")
view_mode = st.sidebar.selectbox(
    "Mode d'affichage",
    ["📊 Visualisation par chapitre", "📈 Analyses détaillées", "🎯 Calibration", "📦 Téléchargement complet"]
)

# Lister les chapitres
chapters = []
if RESULTS_DIR.exists():
    for chapter_dir in RESULTS_DIR.iterdir():
        if chapter_dir.is_dir() and chapter_dir.name.startswith("chap"):
            chapters.append(chapter_dir.name)

if not chapters:
    st.error("❌ Aucun chapitre trouvé")
    st.stop()

# Contenu principal
if view_mode == "📊 Visualisation par chapitre":
    st.header("📊 Visualisation des Résultats par Chapitre")
    
    selected_chapter = st.selectbox("Sélectionnez un chapitre:", chapters)
    chapter_info = get_chapter_info(selected_chapter)
    chapter_dir = RESULTS_DIR / selected_chapter
    
    st.markdown(f"""
    <div class="chapter-card">
        <h2>{chapter_info['icon']} {chapter_info['name']}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Afficher les graphiques
    st.subheader("📊 Graphiques et Analyses")
    
    graph_categories = {
        "🎯 Métriques": [
            ("metrics_best.png", "Métriques du Meilleur Modèle"),
            ("metrics_comparison_all.png", "Comparaison de Tous les Modèles"),
        ],
        "📈 Courbes ROC": [
            ("roc_curve_best.png", "Courbe ROC - Meilleur Modèle"),
            ("roc_curves_all.png", "Courbes ROC - Tous les Modèles"),
        ],
        "📊 Précision-Rappel": [
            ("precision_recall_curve_best.png", "Courbe Précision-Rappel"),
            ("precision_recall_curves_all.png", "Courbes Précision-Rappel"),
        ],
        "🎯 Matrices de Confusion": [
            ("confusion_matrix_best.png", "Matrice de Confusion - Meilleur"),
            ("confusion_matrices_all.png", "Matrices de Confusion - Tous"),
        ],
        "🔍 Analyse SHAP": [
            ("shap_summary_plot_20.png", "Résumé SHAP"),
            ("shap_feature_importance_20.png", "Importance des Features"),
        ]
    }
    
    for category, graphs in graph_categories.items():
        st.markdown(f"### {category}")
        for filename, title in graphs:
            image_path = chapter_dir / filename
            if image_path.exists():
                display_image_with_download(image_path, title, f"{selected_chapter}_{filename}")
            else:
                st.info(f"📄 {title} - Fichier non trouvé")

elif view_mode == "📈 Analyses détaillées":
    st.header("📈 Analyses Détaillées par Chapitre")
    
    for chapter in sorted(chapters):
        chapter_info = get_chapter_info(chapter)
        analysis_dir = ANALYSIS_DIR / chapter
        
        if analysis_dir.exists():
            st.markdown(f"""
            <div class="chapter-card">
                <h3>{chapter_info['icon']} {chapter_info['name']} - Analyses</h3>
            </div>
            """, unsafe_allow_html=True)
            
            analysis_files = list(analysis_dir.glob("*.png"))
            if analysis_files:
                for analysis_file in sorted(analysis_files):
                    display_image_with_download(analysis_file, analysis_file.stem.replace("_", " ").title(), f"analysis_{chapter}_{analysis_file.name}")
            
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
        ("calibration_analysis_chapitre_30_REAL_VALUES.png", "Calibration Chapitre 30"),
        ("calibration_analysis_chapitre_84_REAL_VALUES.png", "Calibration Chapitre 84"),
        ("calibration_analysis_chapitre_85_REAL_VALUES.png", "Calibration Chapitre 85"),
        ("calibration_comparison_ALL_CHAPTERS_REAL_VALUES.png", "Comparaison Calibration"),
        ("calibration_curves_ALL_CHAPTERS_REAL_VALUES.png", "Courbes de Calibration"),
        ("calibration_synthesis_FINAL_REAL_VALUES.png", "Synthèse Calibration")
    ]
    
    for filename, title in calibration_files:
        image_path = CALIBRATION_DIR / filename
        if image_path.exists():
            display_image_with_download(image_path, title, f"calibration_{filename}")
    
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

elif view_mode == "📦 Téléchargement complet":
    st.header("📦 Téléchargement Complet")
    st.write("Téléchargez tous les résultats, analyses et graphiques en un seul fichier ZIP.")
    
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
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>📊 InspectIA - Système de Détection de Fraude Douanière</strong></p>
    <p>Généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")}</p>
</div>
""", unsafe_allow_html=True)

