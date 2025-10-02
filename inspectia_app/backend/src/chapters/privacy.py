#!/usr/bin/env python3
"""
Module de gestion de la confidentialité des données pour respecter la conformité légale.

🔒 CONFORMITÉ LÉGALE:
- Loi sénégalaise n°2008-12 sur la protection des données personnelles
- Inspiration GDPR (droit à l'oubli, minimisation des données)
- Anonymisation irréversible par défaut (SHA-256)
- Option réversible temporaire (Fernet, 14j max)
- Suppression automatique des mappings après expiration

📊 USAGE ML:
- Hachage déterministe → suivi longitudinal possible
- Colonnes originales supprimées → aucune fuite
- Compatible avec tous les algorithmes ML
"""

import pandas as pd
import hashlib
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Configuration du logging
logger = logging.getLogger(__name__)

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    logger.warning("⚠️ Module cryptography non installé → fallback SHA-256 uniquement")

class DataPrivacyManager:
    """
    Gestion de l'anonymisation/chiffrement pour respecter la loi sénégalaise n°2008-12
    et s'inspirer du GDPR.
    
    🔒 CONFORMITÉ LÉGALE:
    - Loi sénégalaise n°2008-12 sur la protection des données personnelles
    - Inspiration GDPR (droit à l'oubli, minimisation des données)
    - Anonymisation irréversible par défaut (SHA-256)
    - Option réversible temporaire (Fernet, 14j max)
    - Suppression automatique des mappings après expiration
    
    📊 USAGE ML:
    - Hachage déterministe → suivi longitudinal possible
    - Colonnes originales supprimées → aucune fuite
    - Compatible avec tous les algorithmes ML
    """

    def __init__(self, mapping_dir="../../../data/mappings_secure", retention_days=14):
        self.mapping_dir = Path(mapping_dir)
        self.mapping_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.key_path = self.mapping_dir / "fernet.key"
        self.mapping_path = self.mapping_dir / "mapping.json"
        self.fernet = self._load_or_create_key()

    def _load_or_create_key(self):
        """Crée ou charge la clé Fernet (rotation possible après 14 jours)."""
        if not HAS_CRYPTO:
            return None
        if self.key_path.exists():
            with open(self.key_path, "rb") as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_path, "wb") as f:
                f.write(key)
        return Fernet(key)

    def _hash_value(self, value: str) -> str:
        """Hachage SHA-256 irréversible (fallback légal minimal)."""
        if pd.isna(value):
            return None
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()

    def _encrypt_value(self, value: str) -> str:
        """Chiffrement Fernet (réversible temporairement)."""
        if not HAS_CRYPTO or pd.isna(value):
            return self._hash_value(value)
        return self.fernet.encrypt(str(value).encode("utf-8")).decode("utf-8")

    def anonymize_dataframe(self, df: pd.DataFrame, sensitive_cols=None, reversible=False):
        """
        Transforme les colonnes sensibles (CODE_IMPORTATEUR, DECLARANT, BUREAU...).
        - Hachage SHA-256 si irreversible
        - Chiffrement Fernet si reversible=True
        """
        if sensitive_cols is None:
            sensitive_cols = [
                "CODE_IMPORTATEUR", "NOM_DECLARANT", "NOM_DESTINATAIRE", 
                "CODE_DESTINATAIRE", "CODE_AGENT", "REFERENCE_COMPLETE",
                "DATE_DECLARATION"
                # EXCLU: NUMERO_DECLARATION (utilisé pour DECLARATION_ID)
                # EXCLU: ANNEE (utilisé pour DECLARATION_ID)
                # EXCLU: LIBELLE_PRODUIT (nécessaire pour TF-IDF)
                # EXCLU: DESCRIPTION_COMMERCIALE (nécessaire pour TF-IDF)
            ]

        mapping = {}
        for col in sensitive_cols:
            if col in df.columns:
                logger.info(f"🔒 Anonymisation de la colonne {col} ({'réversible' if reversible else 'irréversible'})")
                new_col = f"{col}_HASHED"
                df[new_col] = df[col].apply(lambda x: self._encrypt_value(x) if reversible else self._hash_value(x))
                mapping[col] = new_col
                df.drop(columns=[col], inplace=True)

        # Sauvegarde du mapping si réversible
        if reversible and HAS_CRYPTO:
            payload = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "mapping": mapping
            }
            with open(self.mapping_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info(f"📁 Mapping temporaire sauvegardé: {self.mapping_path}")
        return df

    def cleanup_old_mappings(self):
        """Supprime les mappings après la durée légale (14 jours par défaut)."""
        if not self.mapping_path.exists():
            return
        try:
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            created_at = datetime.fromisoformat(payload.get("created_at"))
            # Assurer que les deux datetimes ont le même fuseau horaire
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - created_at > timedelta(days=self.retention_days):
                os.remove(self.mapping_path)
                logger.info("🗑️ Mapping supprimé (expiration délai légal)")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Erreur lors du nettoyage des mappings: {e}")

def apply_privacy_protection(df: pd.DataFrame, mode="production", sensitive_cols=None):
    """
    Fonction utilitaire pour appliquer la protection des données.
    
    Args:
        df: DataFrame à anonymiser
        mode: "production" (irréversible) ou "development" (réversible 14j)
        sensitive_cols: Liste des colonnes sensibles
    
    Returns:
        DataFrame anonymisé
    """
    privacy = DataPrivacyManager()
    
    if sensitive_cols is None:
        sensitive_cols = [
            "CODE_IMPORTATEUR", "NOM_DECLARANT", "NOM_DESTINATAIRE", 
            "CODE_DESTINATAIRE", "CODE_AGENT", "REFERENCE_COMPLETE",
            "DATE_DECLARATION"
            # EXCLU: NUMERO_DECLARATION (utilisé pour DECLARATION_ID)
            # EXCLU: ANNEE (utilisé pour DECLARATION_ID)
            # EXCLU: LIBELLE_PRODUIT (nécessaire pour TF-IDF)
            # EXCLU: DESCRIPTION_COMMERCIALE (nécessaire pour TF-IDF)
        ]
    
    reversible = (mode == "development")
    df = privacy.anonymize_dataframe(df, sensitive_cols=sensitive_cols, reversible=reversible)
    privacy.cleanup_old_mappings()
    
    return df
