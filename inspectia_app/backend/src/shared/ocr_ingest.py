# backend/src/shared/ocr_ingest.py
"""
Module OCR_INGEST - Extraction et traitement des données de déclarations douanières
Intégré au nouveau système ML-RL hybride avec contrat de communication standardisé
"""
from pathlib import Path
import hashlib, shutil, csv, re, json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Imports optionnels avec gestion d'erreur
try:
    from pdf2image import convert_from_path  # type: ignore
except ImportError:
    convert_from_path = None

try:
    import PyPDF2  # type: ignore
except ImportError:
    PyPDF2 = None

try:
    import fitz  # type: ignore # PyMuPDF
except ImportError:
    fitz = None

try:
    import pandas as pd
except Exception:
    pd = None

logger = logging.getLogger(__name__)

def check_dependencies() -> Dict[str, bool]:
    """Vérifier les dépendances optionnelles pour le traitement des documents"""
    dependencies = {
        "pdf2image": convert_from_path is not None,
        "PyPDF2": PyPDF2 is not None,
        "PyMuPDF": fitz is not None,
        "pandas": pd is not None
    }
    
    missing = [name for name, available in dependencies.items() if not available]
    if missing:
        logger.warning(f"Dépendances manquantes: {', '.join(missing)}")
        logger.info("Pour installer les dépendances manquantes:")
        if "pdf2image" in missing:
            logger.info("  pip install pdf2image poppler-utils")
        if "PyPDF2" in missing:
            logger.info("  pip install PyPDF2")
        if "PyMuPDF" in missing:
            logger.info("  pip install PyMuPDF")
        if "pandas" in missing:
            logger.info("  pip install pandas")
    
    return dependencies

# Configuration des chemins pour le nouveau système
INBOX = Path(__file__).resolve().parents[2] / "data" / "ocr_inbox"
DATA  = Path(__file__).resolve().parents[2] / "data" / "ocr_dataset"
MANIFEST = DATA / "manifest.csv"

# Assurer que les répertoires existent
INBOX.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# -------------------------------
# MAPPING DES CHAMPS DE DÉCLARATIONS RÉELLES
# -------------------------------

# Mapping complet des champs CSV vers les features ML
FIELD_MAPPING = {
    # === CHAMPS D'IDENTIFICATION ===
    "declaration_id": ["N° Déclaration", "Nº Déclaration", "N° Répertoire", "N° Ninea", "DECLARATION_ID"],
    "reference_declaration": ["Référence Déclaration", "REFERENCE_DECLARATION"],
    "ninea": ["N° Ninea", "NINEA"],
    "ppm": ["PPM"],
    "annee": ["ANNEE", "Année"],
    "bureau": ["BUREAU", "Bureau", "Bureau Frontière"],
    "numero": ["NUMERO", "NUMERO_DECLARATION", "NUMERO_REPERTOIRE"],
    
    # === CHAMPS FINANCIERS ===
    "valeur_caf": ["VALEUR_CAF", "Valeur en Douane", "Valeur Douane", "Valeur Fob", "valeur_euro"],
    "valeur_fob": ["VALEUR_FOB", "Valeur Fob"],
    "valeur_douane": ["VALEUR_DOUANE", "Valeur en Douane", "Valeur Douane"],
    "montant_liquidation": ["MONTANT_LIQUIDATION", "Montant Liquidé"],
    "assurance": ["Assurance"],
    "fret": ["Frêt", "Fret"],
    "facture": ["Facture"],
    
    # === CHAMPS PHYSIQUES ===
    "poids_net": ["POIDS_NET", "Poids Net", "POIDS_NET_KG"],
    "poids_brut": ["POIDS_BRUT", "Poids Brut"],
    "nombre_colis": ["NOMBRE_COLIS", "Nombre de colis"],
    "quantite_complement": ["QTTE_COMPLEMENTAIRE", "QUANTITE_COMPLEMENT", "Qtité complémentaire"],
    "quantite_mercuriale": ["QUANTITE_MERCURIALE", "Qtité mercuriale"],
    "valeur_unitaire_par_kg": ["VALEUR_UNITAIRE_PAR_KG", "Valeur unitaire par kg"],
    "valeur_par_colis": ["VALEUR_PAR_COLIS", "Valeur par colis"],
    
    # === CHAMPS DE CLASSIFICATION ===
    "code_sh_complet": ["NOMENCLATURE_COMPLETE", "CODE_SH_COMPLET", "Nomenclature", "code_produit"],
    "code_sh": ["CODE_SH", "Nomenclature"],
    "libelle_tarif": ["LIBELLE_TARIF", "LIBELLE DU TARIF"],
    "description_commerciale": ["DESCRIPTION_COMMERCIALE", "DESCRIPTION COMMERCIALE"],
    "categorie_produit": ["CATEGORIE_PRODUIT", "Catégorie produit"],
    "alerte_mots_cles": ["ALERTE_MOTS_CLES", "Alerte mots clés"],
    
    # === CHAMPS GÉOGRAPHIQUES ===
    "pays_origine": ["PAYS_ORIGINE", "CODE_PAYS_ORIGINE", "Origine"],
    "pays_provenance": ["PAYS_PROVENANCE", "CODE_PAYS_PROVENANCE", "Provenance"],
    "destination": ["DESTINATION", "Destination"],
    "bureau_frontiere": ["BUREAU_FRONTIERE", "Bureau Frontière"],
    
    # === CHAMPS DE RÉGIME ===
    "regime_complet": ["REGIME", "REGIME_COMPLET", "Régime"],
    "regime_fiscal": ["REGIME_FISCAL", "Régime Fiscal"],
    "type_regime": ["TYPE_REGIME", "Mode Réglement"],
    "regime_douanier": ["REGIME_DOUANIER", "Régime douanier"],
    "regime_fiscal_code": ["REGIME_FISCAL_CODE", "Régime Fiscal Code"],
    
    # === CHAMPS DE TAUX ET DROITS ===
    "taux_droits_percent": ["TAUX_DROITS_PERCENT", "Taux droits percent"],
    "taux": ["TAUX", "Taux"],
    "montant": ["MONTANT", "Montant"],
    "code_taxe": ["CODE_TAXE", "Code Taxe"],
    "libelle_taxe": ["LIBELLE_TAXE", "Libellé"],
    "base_taxable": ["BASE_TAXABLE", "Base Taxable"],
    
    # === CHAMPS DE TRANSPORT ===
    "nom_navire": ["NOM_NAVIRE", "NOM NAVIRE"],
    "date_arrivee": ["DATE_ARRIVEE", "DATE ARRIVEE"],
    "date_embarquement": ["DATE_EMBARQUEMENT", "DT EMBT"],
    "date_enregistrement": ["DATE_ENREGISTREMENT", "Date Enregistrement"],
    "date_manifeste": ["DATE_MANIFESTE", "DATE ENREG ART.MANIF"],
    "transport_par": ["TRANSPORT_PAR", "TRANSPORT PAR"],
    
    # === CHAMPS DE CONTRÔLE ===
    "statut_bae": ["STATUT_BAE", "APUREMENT Manifeste"],
    "circuit_controle": ["CIRCUIT_CONTROLE", "Circuit vert-Contrôle ducuments", "Circuit bleu-BAE automatique"],
    "nombre_conteneur": ["NOMBRE_CONTENEUR", "Nombre conteneur"],
    "conteneur_id": ["CONTENEUR_ID", "TCNU", "BSIU", "LHV", "RTM", "CN"],
    
    # === CHAMPS D'ARTICLES ===
    "art": ["ART", "ART"],
    "article_manifeste": ["ARTICLE_MANIFESTE", "N° Article Manifeste/T. précédent"],
    "numero_article": ["NUMERO_ARTICLE", "N° Article"],
    "soumission": ["SOUMISSION", "Soumission"],
    "nb_article": ["NB_ARTICLE", "Nb. Article"],
    
    # === CHAMPS DE DOCUMENTS ===
    "dpi": ["DPI", "DPI"],
    "code_pieces_jointes": ["CODE_PIECES_JOINTES", "CODE PIECES JOINTES PARTICULIERES"],
    "na": ["NA", "NA"],
    "precision_uemoa": ["PRECISION_UEMOA", "Précision UEMOA"],
    "date_declaration": ["DATE_DECLARATION", "Date Déclaration"],
    
    # === CHAMPS DE CRÉDIT ET AGRÉMENT ===
    "credit": ["CREDIT", "N° Crédit"],
    "agrement": ["AGREMENT", "N° Agrément"],
    "code_ppm_declarant": ["CODE_PPM_DECLARANT", "CODE_DECLARANT", "Code PPM Déclarant"],
    "code_ppm_destinataire": ["CODE_PPM_DESTINATAIRE", "CODE_DESTINATAIRE", "Code PPM Destinataire"],
    
    # === CHAMPS DE DÉCLARANT ===
    "declarant": ["DECLARANT", "Déclarant"],
    "expediteur": ["EXPEDITEUR", "Expéditeur ou Destinataire réel"]
}

# Mapping complet des champs CSV vers les features ML (clés de sortie)
CSV_TO_ML_MAPPING = {
    # === CHAMPS D'IDENTIFICATION ===
    "ANNEE": "ANNEE",
    "BUREAU": "BUREAU", 
    "NUMERO": "NUMERO",
    "DECLARATION_ID": "DECLARATION_ID",
    "REFERENCE_DECLARATION": "REFERENCE_DECLARATION",
    "NINEA": "NINEA",
    "PPM": "PPM",
    
    # === CHAMPS FINANCIERS ===
    "VALEUR_CAF": "VALEUR_CAF",
    "VALEUR_FOB": "VALEUR_FOB",
    "VALEUR_DOUANE": "VALEUR_DOUANE",
    "MONTANT_LIQUIDATION": "MONTANT_LIQUIDATION",
    "VALEUR_UNITAIRE_PAR_KG": "VALEUR_UNITAIRE_PAR_KG",
    "VALEUR_PAR_COLIS": "VALEUR_PAR_COLIS",
    "ASSURANCE": "ASSURANCE",
    "FRET": "FRET",
    "FACTURE": "FACTURE",
    
    # === CHAMPS PHYSIQUES ===
    "POIDS_NET": "POIDS_NET_KG",
    "POIDS_BRUT": "POIDS_BRUT",
    "NOMBRE_COLIS": "NOMBRE_COLIS",
    "QTTE_COMPLEMENTAIRE": "QUANTITE_COMPLEMENT",
    "QUANTITE_MERCURIALE": "QUANTITE_MERCURIALE",
    "TAUX_DROITS_PERCENT": "TAUX_DROITS_PERCENT",
    
    # === CHAMPS DE CLASSIFICATION ===
    "NOMENCLATURE_COMPLETE": "CODE_PRODUIT_STR",
    "CODE_SH": "CODE_SH",
    "LIBELLE_TARIF": "LIBELLE_TARIF",
    "DESCRIPTION_COMMERCIALE": "DESCRIPTION_COMMERCIALE",
    "CATEGORIE_PRODUIT": "CATEGORIE_PRODUIT",
    "ALERTE_MOTS_CLES": "ALERTE_MOTS_CLES",
    
    # === CHAMPS GÉOGRAPHIQUES ===
    "PAYS_ORIGINE": "PAYS_ORIGINE_STR",
    "PAYS_PROVENANCE": "PAYS_PROVENANCE_STR",
    "DESTINATION": "DESTINATION",
    "BUREAU_FRONTIERE": "BUREAU_FRONTIERE",
    
    # === CHAMPS DE RÉGIME ===
    "REGIME": "REGIME_FISCAL",
    "TYPE_REGIME": "TYPE_REGIME",
    "REGIME_DOUANIER": "REGIME_DOUANIER",
    "REGIME_FISCAL": "REGIME_FISCAL",
    "REGIME_FISCAL_CODE": "REGIME_FISCAL_CODE",
    "STATUT_BAE": "STATUT_BAE",
    
    # === CHAMPS DE TAUX ET DROITS ===
    "TAUX": "TAUX",
    "MONTANT": "MONTANT",
    "CODE_TAXE": "CODE_TAXE",
    "LIBELLE_TAXE": "LIBELLE_TAXE",
    "BASE_TAXABLE": "BASE_TAXABLE",
    
    # === CHAMPS DE TRANSPORT ===
    "NOM_NAVIRE": "NOM_NAVIRE",
    "DATE_ARRIVEE": "DATE_ARRIVEE",
    "DATE_EMBARQUEMENT": "DATE_EMBARQUEMENT",
    "DATE_ENREGISTREMENT": "DATE_ENREGISTREMENT",
    "DATE_MANIFESTE": "DATE_MANIFESTE",
    "TRANSPORT_PAR": "TRANSPORT_PAR",
    
    # === CHAMPS DE CONTRÔLE ===
    "CIRCUIT_CONTROLE": "CIRCUIT_CONTROLE",
    "NOMBRE_CONTENEUR": "NOMBRE_CONTENEUR",
    "CONTENEUR_ID": "CONTENEUR_ID",
    
    # === CHAMPS D'ARTICLES ===
    "ART": "ART",
    "ARTICLE_MANIFESTE": "ARTICLE_MANIFESTE",
    "NUMERO_ARTICLE": "NUMERO_ARTICLE_STR",
    "SOUMISSION": "SOUMISSION",
    "NB_ARTICLE": "NB_ARTICLE",
    
    # === CHAMPS DE DOCUMENTS ===
    "DPI": "DPI",
    "CODE_PIECES_JOINTES": "CODE_PIECES_JOINTES",
    "NA": "NA",
    "PRECISION_UEMOA": "PRECISION_UEMOA_STR",
    "DATE_DECLARATION": "DATE_DECLARATION_STR",
    
    # === CHAMPS DE CRÉDIT ET AGRÉMENT ===
    "CREDIT": "CREDIT",
    "AGREMENT": "AGREMENT",
    "CODE_PPM_DECLARANT": "CODE_PPM_DECLARANT",
    "CODE_PPM_DESTINATAIRE": "CODE_PPM_DESTINATAIRE",
    
    # === CHAMPS DE DÉCLARANT ===
    "DECLARANT": "DECLARANT",
    "EXPEDITEUR": "EXPEDITEUR"
}

# Patterns de validation pour les champs extraits
VALIDATION_PATTERNS = {
    "ninea": r"^\d{9}$",  # 9 chiffres
    "code_sh_complet": r"^\d{6}\s\d{2}\s\d{2}$",  # Format: 300490 90 00
    "pays_origine": r"^[A-Z]{2}$",  # 2 lettres majuscules
    "regime_complet": r"^[A-Z]\d+$",  # Format: C1, S110, etc.
    "date_arrivee": r"^\d{2}/\d{2}/\d{4}$",  # Format: DD/MM/YYYY
    "valeur_caf": r"^\d+(\.\d+)?$",  # Nombre entier ou décimal
    "VALEUR_CAF": r"^\d+(\.\d+)?$",  # Nombre entier ou décimal
    "poids_net": r"^\d+(\.\d+)?$",  # Nombre entier ou décimal
    "POIDS_NET_KG": r"^\d+(\.\d+)?$",  # Nombre entier ou décimal
    "nombre_colis": r"^\d+$",  # Nombre entier
    "NOMBRE_COLIS": r"^\d+$",  # Nombre entier
    "quantite_complement": r"^\d+$",  # Nombre entier
    "QUANTITE_COMPLEMENT": r"^\d+$",  # Nombre entier
    "taux_droits_percent": r"^\d+(\.\d+)?$",  # Nombre entier ou décimal
    "DECLARATION_ID": r".*"  # Accepter n'importe quel DECLARATION_ID
}

# -------------------------------
# INTERFACE DE COMMUNICATION OCR_INGEST ↔ OCR_PIPELINE
# -------------------------------

class OCRDataContract:
    """Contrat de données standardisé pour la communication entre OCR_INGEST et OCR_PIPELINE"""
    
    @staticmethod
    def create_ingest_result(
        validation_status: str,
        fields_extracted: int,
        advanced_features_count: int,
        ml_ready: bool,
        fraud_detection_enabled: bool,
        advanced_context: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Créer un résultat standardisé d'OCR_INGEST pour OCR_PIPELINE
        
        Args:
            validation_status: "success" ou "error"
            fields_extracted: Nombre de champs extraits
            advanced_features_count: Nombre de features avancées créées
            ml_ready: Prêt pour ML
            fraud_detection_enabled: Détection de fraude activée
            advanced_context: Contexte avancé avec toutes les features
            metadata: Métadonnées additionnelles
        
        Returns:
            Dictionnaire standardisé pour OCR_PIPELINE
        """
        return {
            "validation_status": validation_status,
            "fields_extracted": fields_extracted,
            "advanced_features_count": advanced_features_count,
            "ml_ready": ml_ready,
            "fraud_detection_enabled": fraud_detection_enabled,
            "advanced_context": advanced_context,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0"
        }
    
    @staticmethod
    def validate_ingest_result(result: Dict[str, Any]) -> bool:
        """Valider qu'un résultat d'OCR_INGEST est conforme au contrat"""
        required_fields = [
            "validation_status", "fields_extracted", "advanced_features_count",
            "ml_ready", "fraud_detection_enabled", "advanced_context", "metadata"
        ]
        return all(field in result for field in required_fields)
    
    @staticmethod
    def extract_pipeline_input(ingest_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extraire les données nécessaires pour OCR_PIPELINE depuis le résultat d'OCR_INGEST
        
        Args:
            ingest_result: Résultat d'OCR_INGEST
        
        Returns:
            Données formatées pour OCR_PIPELINE
        """
        if not OCRDataContract.validate_ingest_result(ingest_result):
            raise ValueError("Résultat OCR_INGEST invalide")
        
        return {
            "context": ingest_result["advanced_context"],
            "ml_ready": ingest_result["ml_ready"],
            "fraud_detection_enabled": ingest_result["fraud_detection_enabled"],
            "features_count": ingest_result["advanced_features_count"],
            "metadata": ingest_result["metadata"]
        }

def apply_field_mapping(data: Dict[str, Any], mapping_type: str = "csv_to_ml") -> Dict[str, Any]:
    """
    Appliquer le mapping des champs selon le type spécifié
    
    Args:
        data: Dictionnaire de données à mapper
        mapping_type: Type de mapping ("csv_to_ml", "ocr_to_standard", "standard_to_ml")
    
    Returns:
        Dictionnaire avec les champs mappés
    """
    try:
        mapped_data = {}
        
        if mapping_type == "csv_to_ml":
            # Mapping des champs CSV vers les features ML
            for csv_key, ml_key in CSV_TO_ML_MAPPING.items():
                if csv_key in data:
                    mapped_data[ml_key] = data[csv_key]
                else:
                    # Essayer des variantes
                    for variant in [csv_key.lower(), csv_key.upper(), csv_key.replace('_', ' ')]:
                        if variant in data:
                            mapped_data[ml_key] = data[variant]
                            break
            
            # Ajouter les champs non mappés directement
            for key, value in data.items():
                if key not in CSV_TO_ML_MAPPING:
                    mapped_data[key] = value
        
        elif mapping_type == "ocr_to_standard":
            # Mapping des champs OCR vers les champs standard
            for standard_key, ocr_variants in FIELD_MAPPING.items():
                for variant in ocr_variants:
                    if variant in data:
                        mapped_data[standard_key] = data[variant]
                        break
            
            # Ajouter les champs non mappés
            for key, value in data.items():
                if not any(key in variants for variants in FIELD_MAPPING.values()):
                    mapped_data[key] = value
        
        elif mapping_type == "standard_to_ml":
            # Mapping des champs standard vers les features ML
            standard_to_ml = {
                "poids_net": "POIDS_NET_KG",
                "quantite_complement": "QUANTITE_COMPLEMENT", 
                "code_sh_complet": "CODE_PRODUIT_STR",
                "pays_origine": "PAYS_ORIGINE_STR",
                "pays_provenance": "PAYS_PROVENANCE_STR",
                "regime_complet": "REGIME_FISCAL",
                "numero_article": "NUMERO_ARTICLE_STR",
                "precision_uemoa": "PRECISION_UEMOA_STR",
                "date_declaration": "DATE_DECLARATION_STR"
            }
            
            for standard_key, ml_key in standard_to_ml.items():
                if standard_key in data:
                    mapped_data[ml_key] = data[standard_key]
            
            # Ajouter les champs non mappés
            for key, value in data.items():
                if key not in standard_to_ml:
                    mapped_data[key] = value
        
        logger.info(f"Mapping {mapping_type} appliqué: {len(mapped_data)} champs")
        return mapped_data
        
    except Exception as e:
        logger.error(f"Erreur application mapping {mapping_type}: {e}")
        return data

# Mapping des codes de pays
COUNTRY_MAPPING = {
    "FR": "France", "CN": "Chine", "IN": "Inde", "NL": "Pays-Bas",
    "MY": "Malaisie", "PT": "Portugal", "IT": "Italie", "GB": "Royaume-Uni",
    "MA": "Maroc", "SN": "Sénégal", "EG": "Égypte", "AE": "Émirats Arabes Unis",
    "US": "États-Unis", "ES": "Espagne", "HK": "Hong Kong", "NO": "Norvège"
}

# Mapping des régimes douaniers
REGIME_MAPPING = {
    "C1": "Consommation", "C100": "Consommation normale", "C111": "Consommation avec franchise",
    "C131": "Consommation avec réduction", "S110": "Suspensif admission temporaire",
    "S300": "Suspensif entrepôt", "S510": "Suspensif transit", "E100": "Exportation", "00": "Régime normal"
}

def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def _ensure_dirs(chapter: str):
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / chapter / "raw").mkdir(parents=True, exist_ok=True)
    (INBOX / chapter).mkdir(parents=True, exist_ok=True)  # <— important
    if not MANIFEST.exists():
        MANIFEST.write_text(
            "chapter,DECLARATION_ID,article,page,src_name,stored_path,sha256,ingested_at\n",
            encoding="utf-8"
        )

def _explode_pdf(pdf: Path, out_dir: Path) -> List[Path]:
    if convert_from_path is None:
        raise RuntimeError("pdf2image non installé : pip install pdf2image poppler-utils")
    pages = convert_from_path(str(pdf), dpi=300)
    out = []
    base = pdf.stem
    for i, im in enumerate(pages, start=1):
        out_path = out_dir / f"{base}__page-{i:03d}.png"
        im.save(out_path, "PNG")
        out.append(out_path)
    return out

def extract_fields_from_text(text: str) -> Dict[str, Any]:
    """Extraire les champs des déclarations à partir du texte OCR"""
    extracted_data = {}
    
    # Patterns de recherche pour chaque champ
    patterns = {
        'declaration_id': [
            r'N[°º]\s*Déclaration[:\s]*(\d{4}\s+\d{2}[A-Z]\s+\d+)',
            r'N[°º]\s*Répertoire[:\s]*(\d+[A-Z]\d+)',
            r'(\d{4}\s+\d{2}[A-Z]\s+\d+)'
        ],
        'ninea': [
            r'N[°º]\s*Ninea[:\s]*(\d{9})',
            r'NINEA[:\s]*(\d{9})'
        ],
        'valeur_caf': [
            r'Valeur\s+en\s+Douane[:\s]*(\d+)',
            r'Valeur\s+Douane[:\s]*(\d+)',
            r'Valeur\s+Fob[:\s]*(\d+)'
        ],
        'poids_net': [
            r'Poids\s+Net[:\s]*(\d+\.\d{3})',
            r'Poids\s+Net[:\s]*(\d+)'
        ],
        'code_sh_complet': [
            r'Nomenclature[:\s]*(\d{6}\s+\d{2}\s+\d{2})',
            r'(\d{6}\s+\d{2}\s+\d{2})'
        ],
        'pays_origine': [
            r'Origine[:\s]*([A-Z]{2})',
            r'Provenance[:\s]*([A-Z]{2})'
        ],
        'regime_complet': [
            r'Régime[:\s]*([A-Z]\d+)',
            r'Régime\s+Fiscal[:\s]*(\d{2})'
        ],
        'nombre_colis': [
            r'Nombre\s+de\s+colis[:\s]*(\d+)',
            r'Nombre\s+de\s+colis[:\s]*(\d+)'
        ],
        'quantite_complement': [
            r'Qtité\s+complémentaire[:\s]*(\d+)',
            r'Qtité\s+complémentaire[:\s]*(\d+)'
        ],
        'date_arrivee': [
            r'DATE\s+ARRIVEE[:\s]*(\d{2}/\d{2}/\d{4})',
            r'Date\s+Arrivée[:\s]*(\d{2}/\d{2}/\d{4})'
        ],
        'description_commerciale': [
            r'DESCRIPTION\s+COMMERCIALE[:\s]*([^\n]+)',
            r'Description\s+Commerciale[:\s]*([^\n]+)'
        ]
    }
    
    # Extraire chaque champ
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted_data[field] = match.group(1).strip()
                break
    
    return extracted_data

def process_pdf_declaration(pdf_path: str) -> Dict[str, Any]:
    """Traiter un PDF de déclaration douanière - Version optimisée pour le nouveau système"""
    try:
        # Essayer d'abord avec PyMuPDF (meilleur pour le texte)
        if fitz:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
        elif PyPDF2:
            # Fallback avec PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
        else:
            # Si aucune bibliothèque PDF disponible, extraire depuis le nom de fichier
            file_path = Path(pdf_path)
            filename = file_path.stem
            
            # Parser le nom de fichier pour extraire des informations
            extracted_data = {}
            patterns = {
                'declaration_id': r'DECL[_-]?(\w+)',
                'ninea': r'NINEA[_-]?(\d{9})',
                'annee': r'(\d{4})',
                'bureau': r'(\d{2}[A-Z])',
                'numero': r'(\d+)'
            }
            
            for field, pattern in patterns.items():
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    extracted_data[field] = match.group(1)
            
            if not extracted_data:
                extracted_data = {
                    'declaration_id': f'PDF_{file_path.stem}',
                    'source_file': filename
                }
            
            normalized_data = normalize_ocr_data(extracted_data)
            logger.info(f"PDF traité depuis nom de fichier: {len(extracted_data)} champs extraits")
            return normalized_data
        
        # Extraire les champs du texte
        extracted_data = extract_fields_from_text(text)
        
        # Si aucune donnée extraite du texte, essayer depuis le nom de fichier
        if not extracted_data:
            file_path = Path(pdf_path)
            filename = file_path.stem
            
            patterns = {
                'declaration_id': r'DECL[_-]?(\w+)',
                'ninea': r'NINEA[_-]?(\d{9})',
                'annee': r'(\d{4})',
                'bureau': r'(\d{2}[A-Z])',
                'numero': r'(\d+)'
            }
            
            for field, pattern in patterns.items():
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    extracted_data[field] = match.group(1)
            
            if not extracted_data:
                extracted_data = {
                    'declaration_id': f'PDF_{file_path.stem}',
                    'source_file': filename
                }
        
        # Normaliser les données
        normalized_data = normalize_ocr_data(extracted_data)
        
        logger.info(f"PDF traité: {len(extracted_data)} champs extraits")
        return normalized_data
        
    except Exception as e:
        logger.error(f"Erreur traitement PDF {pdf_path}: {e}")
        return {}

def aggregate_csv_by_declaration(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Agréger les données CSV par DECLARATION_ID (ANNEE/BUREAU/NUMERO)
    """
    try:
        # Créer DECLARATION_ID si pas présent
        if 'DECLARATION_ID' not in df.columns:
            # Essayer différentes variantes de colonnes NUMERO
            numero_col = None
            for col in ['NUMERO', 'NUMERO_DECLARATION', 'NUMERO_REPERTOIRE']:
                if col in df.columns:
                    numero_col = col
                    break
            
            if all(col in df.columns for col in ['ANNEE', 'BUREAU']) and numero_col:
                df['DECLARATION_ID'] = df['ANNEE'].astype(str) + '/' + df['BUREAU'].astype(str) + '/' + df[numero_col].astype(str)
            else:
                # Si pas de colonnes ANNEE/BUREAU/NUMERO, créer un ID unique
                df['DECLARATION_ID'] = 'DECL_' + df.index.astype(str)
        
        # Colonnes numériques à agréger (CORRECTION: utiliser les vraies clés du CSV)
        numeric_cols = [
            'VALEUR_CAF', 'VALEUR_DOUANE', 'MONTANT_LIQUIDATION', 'POIDS_NET',  # CORRECTION: POIDS_NET
            'NOMBRE_COLIS', 'QTTE_COMPLEMENTAIRE', 'VALEUR_UNITAIRE_PAR_KG',  # CORRECTION: QTTE_COMPLEMENTAIRE
            'VALEUR_PAR_COLIS', 'TAUX_DROITS_PERCENT'
        ]
        
        # Colonnes catégorielles à prendre en premier (CORRECTION: utiliser les vraies clés du CSV)
        categorical_cols = [
            'STATUT_BAE', 'TYPE_REGIME', 'REGIME_DOUANIER', 'REGIME_FISCAL',
            'REGIME', 'NOMENCLATURE_COMPLETE', 'ALERTE_MOTS_CLES',  # CORRECTION: REGIME et NOMENCLATURE_COMPLETE
            'CATEGORIE_PRODUIT', 'PAYS_ORIGINE', 'PAYS_PROVENANCE',  # CORRECTION: PAYS_ORIGINE et PAYS_PROVENANCE
            'CODE_PPM_DECLARANT', 'CODE_PPM_DESTINATAIRE'  # CORRECTION: CODE_PPM_DECLARANT et CODE_PPM_DESTINATAIRE
        ]
        
        # Filtrer les colonnes existantes
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        # Agrégation
        agg_dict = {}
        
        # Numériques : somme
        for col in numeric_cols:
            agg_dict[col] = 'sum'
        
        # Catégorielles : premier
        for col in categorical_cols:
            agg_dict[col] = 'first'
        
        # Vérifier si on a des colonnes à agréger
        if not agg_dict:
            # Pas de colonnes numériques/catégorielles, retourner les données telles quelles
            declarations = df.to_dict('records')
        else:
            # Agrégation par DECLARATION_ID
            df_agg = df.groupby('DECLARATION_ID').agg(agg_dict).reset_index()
            
            # Convertir en liste de dictionnaires
            declarations = df_agg.to_dict('records')
        
        # CORRECTION: Mapper les clés du CSV vers les clés attendues par le système
        for decl in declarations:
            # Mapping des clés numériques
            if 'POIDS_NET' in decl:
                decl['POIDS_NET_KG'] = decl.pop('POIDS_NET')
            if 'QTTE_COMPLEMENTAIRE' in decl:
                decl['QUANTITE_COMPLEMENT'] = decl.pop('QTTE_COMPLEMENTAIRE')
            
            # Mapping des clés catégorielles
            if 'NOMENCLATURE_COMPLETE' in decl:
                decl['CODE_SH_COMPLET'] = decl.pop('NOMENCLATURE_COMPLETE')
            if 'PAYS_ORIGINE' in decl:
                decl['CODE_PAYS_ORIGINE'] = decl.pop('PAYS_ORIGINE')
            if 'PAYS_PROVENANCE' in decl:
                decl['CODE_PAYS_PROVENANCE'] = decl.pop('PAYS_PROVENANCE')
            if 'REGIME' in decl:
                decl['REGIME_COMPLET'] = decl.pop('REGIME')
            if 'CODE_PPM_DECLARANT' in decl:
                decl['CODE_DECLARANT'] = decl.pop('CODE_PPM_DECLARANT')
            if 'CODE_PPM_DESTINATAIRE' in decl:
                decl['CODE_DESTINATAIRE'] = decl.pop('CODE_PPM_DESTINATAIRE')
        
        logger.info(f"✅ CSV agrégé: {len(declarations)} déclarations uniques")
        return declarations
        
    except Exception as e:
        logger.error(f"Erreur agrégation CSV: {e}")
        return []

def process_csv_declaration(csv_path: str) -> Dict[str, Any]:
    """Traiter un fichier CSV de déclaration avec agrégation par DECLARATION_ID"""
    try:
        if not pd:
            raise RuntimeError("pandas non disponible")
        
        # Lire le CSV
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return {"error": "CSV vide"}
        
        # Agrégation par DECLARATION_ID si nécessaire
        aggregated_data = aggregate_csv_by_declaration(df)
        
        # Prendre la première déclaration pour l'analyse
        if aggregated_data:
            first_declaration = aggregated_data[0]
            
            # Normaliser les données
            normalized_data = normalize_ocr_data(first_declaration)
            
            return {
                "extracted_data": normalized_data,
                "total_declarations": len(aggregated_data),
                "source_type": "csv"
            }
        else:
            return {"error": "Aucune donnée valide trouvée dans le CSV"}
        
    except Exception as e:
        logger.error(f"Erreur traitement CSV {csv_path}: {e}")
        return {}

def process_image_declaration(image_path: str) -> Dict[str, Any]:
    """Traiter une image de déclaration avec OCR - Version simplifiée pour le nouveau système"""
    try:
        # Pour le nouveau système, on utilise une extraction basique
        # Le traitement OCR avancé est géré par OCR_PIPELINE
        
        # Extraire les champs basiques depuis le nom du fichier
        file_path = Path(image_path)
        filename = file_path.stem
        
        # Parser le nom de fichier pour extraire des informations
        extracted_data = {}
        
        # Patterns de base pour extraire des informations du nom de fichier
        patterns = {
            'declaration_id': r'DECL[_-]?(\w+)',
            'ninea': r'NINEA[_-]?(\d{9})',
            'annee': r'(\d{4})',
            'bureau': r'(\d{2}[A-Z])',
            'numero': r'(\d+)'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                extracted_data[field] = match.group(1)
        
        # Si aucune donnée extraite du nom, créer des données par défaut
        if not extracted_data:
            extracted_data = {
                'declaration_id': f'IMG_{file_path.stem}',
                'source_file': filename
            }
        
        # Normaliser les données
        normalized_data = normalize_ocr_data(extracted_data)
        
        logger.info(f"Image traitée: {len(normalized_data)} champs extraits depuis {filename}")
        return normalized_data
        
    except Exception as e:
        logger.error(f"Erreur traitement image {image_path}: {e}")
        return {}

def normalize_ocr_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normaliser les données OCR extraites"""
    normalized = {}
    
    for key, value in data.items():
        if value is None or value == "":
            continue
            
        # Nettoyer la valeur
        if isinstance(value, str):
            value = value.strip()
        
        # Conversion spécifique par type de champ
        if key in ["valeur_caf", "valeur_fob", "valeur_douane", "poids_net", "poids_brut", "VALEUR_CAF", "POIDS_NET_KG", "valeur_caf"]:
            try:
                normalized[key] = float(str(value).replace(",", "."))
            except:
                normalized[key] = 0.0
        elif key in ["nombre_colis", "quantite_complement", "quantite_mercuriale", "NOMBRE_COLIS", "QUANTITE_COMPLEMENT", "nombre_colis"]:
            try:
                normalized[key] = int(str(value))
            except:
                normalized[key] = 0
        elif key in ["DECLARATION_ID"]:
            # Préserver le DECLARATION_ID tel quel
            normalized[key] = str(value)
        else:
            normalized[key] = str(value)
    
    return normalized

def validate_extracted_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Valider et nettoyer les données extraites"""
    validated_data = {}
    
    for field, value in data.items():
        if field in VALIDATION_PATTERNS:
            pattern = VALIDATION_PATTERNS[field]
            if re.match(pattern, str(value)):
                validated_data[field] = value
            else:
                logger.warning(f"Champ {field} invalide: {value}")
                # Essayer de nettoyer la valeur
                cleaned_value = clean_field_value(field, value)
                if cleaned_value:
                    validated_data[field] = cleaned_value
        else:
            validated_data[field] = value
    
    return validated_data

def clean_field_value(field: str, value: str) -> str:
    """Nettoyer une valeur de champ"""
    if not value:
        return ""
    
    value = str(value).strip()
    
    # Nettoyage spécifique par type de champ
    if field == "ninea":
        # Garder seulement les chiffres
        cleaned = re.sub(r'\D', '', value)
        return cleaned if len(cleaned) == 9 else ""
    
    elif field == "code_sh_complet":
        # Nettoyer le format du code SH
        cleaned = re.sub(r'[^\d\s]', '', value)
        parts = cleaned.split()
        if len(parts) >= 3:
            return f"{parts[0]:0>6} {parts[1]:0>2} {parts[2]:0>2}"
        return cleaned
    
    elif field == "pays_origine":
        # Garder seulement les lettres majuscules
        cleaned = re.sub(r'[^A-Z]', '', value.upper())
        return cleaned if len(cleaned) == 2 else ""
    
    elif field == "valeur_caf":
        # Garder seulement les chiffres
        return re.sub(r'\D', '', value)
    
    elif field == "poids_net":
        # Nettoyer le format du poids
        cleaned = re.sub(r'[^\d.]', '', value)
        if '.' in cleaned:
            parts = cleaned.split('.')
            if len(parts) == 2:
                return f"{parts[0]}.{parts[1][:3]:0<3}"
        return cleaned
    
    return value

def _parse_meta(name: str) -> Dict[str, str]:
    # Essaie d'inférer DECLARATION_ID / article / page depuis le nom (best effort)
    decl = None
    m = re.search(r"(\d{4}\s+\d+[A-Z]+\s+\d+|DECL[_-]?\w+)", name, re.I)
    if m: decl = m.group(1)

    art = None
    m = re.search(r"article[-_]?(\d+)", name, re.I)
    if m: art = m.group(1)

    page = None
    m = re.search(r"page[-_]?(\d+)", name, re.I)
    if m: page = m.group(1)

    return {"DECLARATION_ID": decl or "", "article": art or "", "page": page or ""}

def ingest(chapter: str, paths: List[str]) -> Dict[str, int]:
    """
    Ingest images/PDF pour un chapitre, déduplique, met à jour le manifest.
    """
    _ensure_dirs(chapter)
    inbox_dir = INBOX / chapter
    stored_dir = DATA / chapter / "raw"
    stored_dir.mkdir(parents=True, exist_ok=True)

    added, skipped = 0, 0
    now = datetime.utcnow().isoformat()

    # Collecte des fichiers à traiter
    files: List[Path] = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_dir():
            files.extend([*p.glob("*.png"), *p.glob("*.jpg"), *p.glob("*.jpeg"), *p.glob("*.pdf")])
        elif p.exists():
            files.append(p)

    # Explosion des PDFs
    expanded: List[Path] = []
    for f in files:
        if f.suffix.lower() == ".pdf":
            expanded += _explode_pdf(f, inbox_dir)
        else:
            expanded.append(f)

    # Ingestion + dédup
    seen_hashes = set()
    if MANIFEST.exists():
        import csv as _csv
        with MANIFEST.open("r", encoding="utf-8", newline="") as mf:
            reader = _csv.DictReader(mf)
            for row in reader:
                sha = row.get("sha256")
                if sha: seen_hashes.add(sha)

    with MANIFEST.open("a", encoding="utf-8", newline="") as mf:
        writer = csv.writer(mf)
        for f in expanded:
            if f.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
            sha = _hash_file(f)
            if sha in seen_hashes:
                skipped += 1
                try:
                    if f.parent == inbox_dir:
                        f.unlink()  # Supprimer le doublon d'INBOX
                except Exception:
                    pass
                continue
            meta = _parse_meta(f.name)
            dest = stored_dir / f.name
            if f.parent != stored_dir:
                try:
                    shutil.copy2(f, dest)
                except Exception:
                    dest = stored_dir / f"{sha[:16]}_{f.name}"
                    shutil.copy2(f, dest)
            writer.writerow([
                chapter,
                meta["DECLARATION_ID"], meta["article"], meta["page"],
                f.name, str(dest.relative_to(DATA)), sha, now
            ])
            added += 1
            
            # Nettoyer les fichiers temporaires d'INBOX après copie
            try:
                if f.parent == inbox_dir:
                    f.unlink()  # Supprimer le fichier temporaire d'INBOX
            except Exception:
                pass  # Ignorer les erreurs de suppression
    
    return {"added": added, "skipped": skipped}

def create_advanced_context_from_ocr_data(ocr_data: Dict[str, Any], chapter: str) -> Dict[str, Any]:
    """
    Créer un contexte avancé à partir des données OCR avec le mapping complet
    """
    try:
        # ÉTAPE 1: Appliquer le mapping CSV vers ML
        mapped_data = apply_field_mapping(ocr_data, 'csv_to_ml')
        
        # ÉTAPE 2: Créer le contexte de base avec toutes les features mappées
        context = {}
        
        # Features numériques avec conversion sécurisée
        numeric_features = [
            'POIDS_NET_KG', 'NOMBRE_COLIS', 'QUANTITE_COMPLEMENT', 'TAUX_DROITS_PERCENT',
            'VALEUR_CAF', 'VALEUR_UNITAIRE_PAR_KG', 'VALEUR_DOUANE', 'MONTANT_LIQUIDATION',
            'POIDS_BRUT', 'QUANTITE_MERCURIALE', 'VALEUR_FOB', 'VALEUR_PAR_COLIS',
            'ASSURANCE', 'FRET', 'TAUX', 'MONTANT', 'BASE_TAXABLE', 'NOMBRE_CONTENEUR'
        ]
        
        for feature in numeric_features:
            value = mapped_data.get(feature, 0)
            try:
                context[feature] = float(value) if value is not None else 0.0
            except (ValueError, TypeError):
                context[feature] = 0.0
        
        # Features string avec conversion sécurisée
        string_features = [
            'CODE_PRODUIT_STR', 'PAYS_ORIGINE_STR', 'PAYS_PROVENANCE_STR', 'BUREAU',
            'REGIME_FISCAL', 'NUMERO_ARTICLE_STR', 'PRECISION_UEMOA_STR', 'DATE_DECLARATION_STR',
            'CODE_SH', 'LIBELLE_TARIF', 'DESCRIPTION_COMMERCIALE', 'CATEGORIE_PRODUIT',
            'ALERTE_MOTS_CLES', 'DESTINATION', 'BUREAU_FRONTIERE', 'TYPE_REGIME',
            'REGIME_DOUANIER', 'REGIME_FISCAL_CODE', 'STATUT_BAE', 'CODE_TAXE',
            'LIBELLE_TAXE', 'NOM_NAVIRE', 'DATE_ARRIVEE', 'DATE_EMBARQUEMENT',
            'DATE_ENREGISTREMENT', 'DATE_MANIFESTE', 'TRANSPORT_PAR', 'CIRCUIT_CONTROLE',
            'CONTENEUR_ID', 'ART', 'ARTICLE_MANIFESTE', 'SOUMISSION', 'NB_ARTICLE',
            'DPI', 'CODE_PIECES_JOINTES', 'NA', 'CREDIT', 'AGREMENT',
            'CODE_PPM_DECLARANT', 'CODE_PPM_DESTINATAIRE', 'DECLARANT', 'EXPEDITEUR',
            'FACTURE', 'ANNEE', 'NUMERO', 'DECLARATION_ID', 'REFERENCE_DECLARATION',
            'NINEA', 'PPM'
        ]
        
        for feature in string_features:
            value = mapped_data.get(feature, '')
            context[feature] = str(value) if value is not None else ''
        
        # ÉTAPE 3: Calculer les ratios et features dérivées
        context['VALEUR_UNITAIRE_KG'] = 0.0
        context['RATIO_DOUANE_CAF'] = 0.0
        
        if context['POIDS_NET_KG'] > 0:
            context['VALEUR_UNITAIRE_KG'] = context['VALEUR_CAF'] / context['POIDS_NET_KG']
        
        if context['VALEUR_CAF'] > 0:
            context['RATIO_DOUANE_CAF'] = context['TAUX_DROITS_PERCENT'] / 100.0
        
        # Ajouter les features business spécifiques au chapitre
        context.update(_create_chapter_specific_business_features(context, chapter))
        
        # Ajouter les scores de détection de fraude avancée (simulés pour l'OCR)
        context.update(_create_advanced_fraud_scores(context, chapter))
        
        logger.info(f"Contexte avancé créé pour {chapter}: {len(context)} features")
        return context
        
    except Exception as e:
        logger.error(f"Erreur création contexte avancé pour {chapter}: {e}")
        return {}

def _create_chapter_specific_business_features(context: Dict[str, Any], chapter: str) -> Dict[str, Any]:
    """Créer les features business spécifiques à chaque chapitre"""
    features = {}
    
    if chapter == "chap30":
        # Features spécifiques au chapitre 30 (Produits pharmaceutiques)
        code_produit = context.get('CODE_PRODUIT_STR', '')
        pays_origine = context.get('PAYS_ORIGINE_STR', '')
        
        features.update({
            'BUSINESS_IS_MEDICAMENT': 1 if code_produit.startswith('30') else 0,
            'BUSINESS_POIDS_ELEVE': 1 if context.get('POIDS_NET_KG', 0) > 100 else 0,
            'BUSINESS_IS_ANTIPALUDEEN': 1 if 'antipalud' in code_produit.lower() else 0,
            'BUSINESS_GLISSEMENT_COSMETIQUE': 1 if code_produit.startswith('33') else 0,
            'BUSINESS_GLISSEMENT_PAYS_COSMETIQUES': 1 if pays_origine in ['FR', 'IT', 'DE'] else 0,
            'BUSINESS_DROITS_ELEVES': 1 if context.get('TAUX_DROITS_PERCENT', 0) > 20 else 0,
            'BUSINESS_RATIO_DOUANE_CAF': context.get('RATIO_DOUANE_CAF', 0),
            'BUSINESS_ARTICLES_MULTIPLES': 1 if context.get('NOMBRE_COLIS', 0) > 1 else 0,
            'BUSINESS_IS_PRECISION_UEMOA': 1 if context.get('PRECISION_UEMOA_STR', '') else 0,
            'BUSINESS_BUREAU_RISQUE': 1 if context.get('BUREAU', '') in ['19C', '19D'] else 0,
        })
        
    elif chapter == "chap84":
        # Features spécifiques au chapitre 84 (Machines et équipements)
        code_produit = context.get('CODE_PRODUIT_STR', '')
        
        features.update({
            'BUSINESS_IS_MACHINE': 1 if code_produit.startswith('84') else 0,
            'BUSINESS_IS_ELECTRONIQUE': 1 if code_produit.startswith(('8471', '8473', '8474', '8475', '8476', '8477', '8478', '8479')) else 0,
            'BUSINESS_POIDS_ELEVE': 1 if context.get('POIDS_NET_KG', 0) > 1000 else 0,
            'BUSINESS_DROITS_ELEVES': 1 if context.get('TAUX_DROITS_PERCENT', 0) > 20 else 0,
            'BUSINESS_RATIO_DOUANE_CAF': context.get('RATIO_DOUANE_CAF', 0),
            'BUSINESS_ARTICLES_MULTIPLES': 1 if context.get('NOMBRE_COLIS', 0) > 1 else 0,
            'BUSINESS_IS_PRECISION_UEMOA': 1 if context.get('PRECISION_UEMOA_STR', '') else 0,
            'BUSINESS_BUREAU_RISQUE': 1 if context.get('BUREAU', '') in ['19C', '19D'] else 0,
        })
        
    elif chapter == "chap85":
        # Features spécifiques au chapitre 85 (Appareils électriques)
        code_produit = context.get('CODE_PRODUIT_STR', '')
        
        features.update({
            'BUSINESS_IS_ELECTRONIQUE': 1 if code_produit.startswith('85') else 0,
            'BUSINESS_IS_TELEPHONE': 1 if code_produit.startswith(('8517', '8525', '8526', '8527', '8528', '8529')) else 0,
            'BUSINESS_POIDS_FAIBLE': 1 if context.get('POIDS_NET_KG', 0) < 10 else 0,
            'BUSINESS_DROITS_ELEVES': 1 if context.get('TAUX_DROITS_PERCENT', 0) > 20 else 0,
            'BUSINESS_RATIO_DOUANE_CAF': context.get('RATIO_DOUANE_CAF', 0),
            'BUSINESS_ARTICLES_MULTIPLES': 1 if context.get('NOMBRE_COLIS', 0) > 1 else 0,
            'BUSINESS_IS_PRECISION_UEMOA': 1 if context.get('PRECISION_UEMOA_STR', '') else 0,
            'BUSINESS_BUREAU_RISQUE': 1 if context.get('BUREAU', '') in ['19C', '19D'] else 0,
        })
    
    return features

def _create_advanced_fraud_scores(context: Dict[str, Any], chapter: str) -> Dict[str, Any]:
    """
    Créer des scores de fraude basiques pour OCR_INGEST
    NOTE: Les vrais scores de fraude sont calculés dans le preprocessing et utilisés par les modèles ML.
    Cette fonction fournit des scores basiques pour la compatibilité avec le nouveau système.
    """
    # Scores basiques par défaut (seront remplacés par les vrais scores dans OCR_PIPELINE)
    scores = {
        'BIENAYME_CHEBYCHEV_SCORE': 0.1,
        'MIRROR_TEI_SCORE': 0.1,
        'SPECTRAL_CLUSTER_SCORE': 0.1,
        'HIERARCHICAL_CLUSTER_SCORE': 0.1,
        'ADMIN_VALUES_SCORE': 0.1,
        'COMPOSITE_FRAUD_SCORE': 0.1,
    }
    
    # Ajustements basiques basés sur les patterns business (sans recalcul complexe)
    if context.get('BUSINESS_DROITS_ELEVES', 0) == 1:
        scores['BIENAYME_CHEBYCHEV_SCORE'] += 0.1
        scores['MIRROR_TEI_SCORE'] += 0.1
    
    if context.get('BUSINESS_ARTICLES_MULTIPLES', 0) == 1:
        scores['MIRROR_TEI_SCORE'] += 0.1
        scores['SPECTRAL_CLUSTER_SCORE'] += 0.1
    
    if context.get('BUSINESS_BUREAU_RISQUE', 0) == 1:
        scores['SPECTRAL_CLUSTER_SCORE'] += 0.1
        scores['HIERARCHICAL_CLUSTER_SCORE'] += 0.1
    
    # Normaliser les scores
    for key in scores:
        if key != 'COMPOSITE_FRAUD_SCORE':
            scores[key] = min(1.0, max(0.0, scores[key]))
    
    # Calculer le score composite
    score_values = [v for k, v in scores.items() if k != 'COMPOSITE_FRAUD_SCORE']
    scores['COMPOSITE_FRAUD_SCORE'] = sum(score_values) / len(score_values) if score_values else 0.1
    
    logger.info(f"Scores de fraude basiques créés pour {chapter}: {scores['COMPOSITE_FRAUD_SCORE']:.3f}")
    return scores

def process_declaration_file(file_path: str, chapter: str = None) -> Dict[str, Any]:
    """
    Traiter un fichier de déclaration (PDF, CSV ou Image) avec le mapping complet et les nouvelles features avancées
    """
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
        
        # Déterminer le type de fichier
        file_ext = file_path.suffix.lower()
        if file_ext == '.pdf':
            processing_result = process_pdf_declaration(str(file_path))
            source_type = "pdf"
        elif file_ext == '.csv':
            processing_result = process_csv_declaration(str(file_path))
            source_type = "csv"
        elif file_ext in ['.png', '.jpg', '.jpeg']:
            processing_result = process_image_declaration(str(file_path))
            source_type = "image"
        else:
            raise ValueError(f"Type de fichier non supporté: {file_ext}")
        
        # Extraire les données et métadonnées
        if isinstance(processing_result, dict) and "extracted_data" in processing_result:
            # Cas où la fonction retourne un dict avec métadonnées
            extracted_data = processing_result["extracted_data"]
            total_declarations = processing_result.get("total_declarations", 1)
        else:
            # Cas où la fonction retourne directement les données
            extracted_data = processing_result
            total_declarations = 1
        
        # Valider les données extraites
        validated_data = validate_extracted_data(extracted_data)
        
        # Déterminer le chapitre si non fourni
        if not chapter:
            from .ocr_pipeline import extract_chapter_from_code_sh
            # Essayer les deux formats de clés possibles
            code_sh = validated_data.get('code_sh_complet', '') or validated_data.get('CODE_SH_COMPLET', '')
            chapter = extract_chapter_from_code_sh(code_sh)
        
        # Créer le contexte avancé avec les nouvelles features
        advanced_context = create_advanced_context_from_ocr_data(validated_data, chapter)
        
        # Créer le résultat standardisé avec le contrat de communication
        metadata = {
            "file_path": str(file_path),
            "chapter": chapter,
            "extracted_data": advanced_context,
            "source_type": source_type,
            "total_declarations": total_declarations,
            "processing_timestamp": datetime.now().isoformat()
        }
        
        result = OCRDataContract.create_ingest_result(
            validation_status="success",
            fields_extracted=len(validated_data),
            advanced_features_count=len(advanced_context),
            ml_ready=True,
            fraud_detection_enabled=True,
            advanced_context=advanced_context,
            metadata=metadata
        )
        
        logger.info(f"Déclaration traitée avec features avancées: {file_path.name} -> {chapter} ({len(validated_data)} champs, {len(advanced_context)} features avancées)")
        return result
        
    except Exception as e:
        logger.error(f"Erreur traitement déclaration {file_path}: {e}")
        return {
            "file_path": str(file_path),
            "chapter": chapter or "unknown",
            "error": str(e),
            "processing_timestamp": datetime.now().isoformat(),
            "validation_status": "error",
            "ml_ready": False,
            "fraud_detection_enabled": False
        }

# -------------------------------
# FONCTIONS UTILITAIRES POUR LE NOUVEAU SYSTÈME
# -------------------------------

def get_supported_file_types() -> List[str]:
    """Retourne la liste des types de fichiers supportés"""
    return ['.csv', '.pdf', '.png', '.jpg', '.jpeg']

def validate_file_type(file_path: str) -> bool:
    """Valide si le type de fichier est supporté"""
    file_ext = Path(file_path).suffix.lower()
    return file_ext in get_supported_file_types()

def get_extraction_statistics() -> Dict[str, Any]:
    """Retourne les statistiques d'extraction du système"""
    return {
        "supported_file_types": get_supported_file_types(),
        "field_mappings_count": len(FIELD_MAPPING),
        "csv_ml_mappings_count": len(CSV_TO_ML_MAPPING),
        "validation_patterns_count": len(VALIDATION_PATTERNS),
        "dependencies": check_dependencies()
    }

def create_test_declaration_data(chapter: str = "chap30") -> Dict[str, Any]:
    """Crée des données de test pour un chapitre donné"""
    base_data = {
        'DECLARATION_ID': f'TEST_{chapter}_001',
        'VALEUR_CAF': 1000.0,
        'POIDS_NET_KG': 10.5,
        'NOMBRE_COLIS': 1,
        'QUANTITE_COMPLEMENT': 0,
        'TAUX_DROITS_PERCENT': 5.0
    }
    
    # Ajouter des données spécifiques au chapitre
    if chapter == "chap30":
        base_data.update({
            'CODE_PRODUIT_STR': '3004909000',
            'PAYS_ORIGINE_STR': 'FR',
            'BUSINESS_IS_MEDICAMENT': 1
        })
    elif chapter == "chap84":
        base_data.update({
            'CODE_PRODUIT_STR': '8471300000',
            'PAYS_ORIGINE_STR': 'CN',
            'BUSINESS_IS_MACHINE': 1
        })
    elif chapter == "chap85":
        base_data.update({
            'CODE_PRODUIT_STR': '8517120000',
            'PAYS_ORIGINE_STR': 'KR',
            'BUSINESS_IS_ELECTRONIQUE': 1
        })
    
    return base_data

def get_module_info() -> Dict[str, Any]:
    """Retourne les informations sur le module OCR_INGEST"""
    return {
        "module_name": "OCR_INGEST",
        "version": "2.0.0",
        "description": "Module d'extraction et traitement des données de déclarations douanières",
        "integrated_system": "ML-RL Hybride",
        "communication_contract": "OCRDataContract",
        "supported_chapters": ["chap30", "chap84", "chap85"],
        "main_functions": [
            "process_declaration_file",
            "process_csv_declaration", 
            "process_pdf_declaration",
            "process_image_declaration",
            "aggregate_csv_by_declaration",
            "create_advanced_context_from_ocr_data"
        ]
    }


