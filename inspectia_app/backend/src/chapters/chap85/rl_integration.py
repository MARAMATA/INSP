# backend/src/chapters/chap85/rl_integration.py
from __future__ import annotations
from typing import Any, Dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
from shared.advanced_reinforcement_learning import AdvancedRLManager
from shared.ml_retraining_system import get_retraining_system

# Configuration RL adaptée au chapitre 85 (Appareils électriques)
# Intégration avec XGBoost - Validation F1: 0.9808 ⭐ (critère de sélection)
# Test: F1=0.9808, AUC=0.9993, Precision=0.9894, Recall=0.9723
# Seuils optimaux calculés: seuil=0.200, confiance_haute=0.900
# 3 niveaux = stratégies différentes optimisées pour les nouveaux modèles
# - basic   -> epsilon-greedy avec seuils optimaux (0.200)
# - advanced-> UCB avec confiance élevée (0.900)
# - expert  -> hybrid avec seuils optimaux (0.200)
_MANAGERS = {
    "basic":   AdvancedRLManager(chapter="chap85", epsilon=0.09, strategy="epsilon_greedy"),
    "advanced":AdvancedRLManager(chapter="chap85", epsilon=0.04, strategy="ucb"),
    "expert":  AdvancedRLManager(chapter="chap85", epsilon=0.02, strategy="hybrid"),
}

def get_manager(level: str = "basic") -> AdvancedRLManager:
    return _MANAGERS.get(level, _MANAGERS["basic"])

def rl_predict(
    context: Dict[str, Any],
    level: str = "basic",
    ml_probability: float | None = None,
    ml_threshold: float | None = None,
) -> Dict[str, Any]:
    return get_manager(level).predict(
        context,
        ml_probability=ml_probability,
        ml_threshold=ml_threshold,
    )

def rl_add_feedback(
    level: str,
    *,
    declaration_id: str,
    predicted_fraud: bool,
    predicted_probability: float,
    inspector_decision: bool,
    inspector_confidence: float,
    inspector_id: str,
    context: Dict[str, Any] | None = None,
    notes: str | None = None,
    exploration_used: bool | None = None,
    review_time_seconds: int | None = None,
    feedback_category: str = "regular",
) -> Dict[str, Any]:
    """Ajoute un feedback avec retraining automatique des modèles ML"""
    # Ajouter le feedback au système RL
    result = get_manager(level).add_feedback(
        declaration_id=declaration_id,
        predicted_fraud=predicted_fraud,
        predicted_probability=predicted_probability,
        inspector_decision=inspector_decision,
        inspector_confidence=inspector_confidence,
        inspector_id=inspector_id,
        context=context,
        notes=notes,
        exploration_used=exploration_used,
        review_time_seconds=review_time_seconds,
        feedback_category=feedback_category,
    )
    
    # 🚀 NOUVEAU: Déclencher le retraining automatique des modèles ML
    try:
        retraining_system = get_retraining_system()
        if retraining_system.should_retrain("chap85"):
            retraining_result = retraining_system.retrain_model("chap85")
            result["ml_retraining"] = retraining_result
        else:
            result["ml_retraining"] = {"success": False, "reason": "Conditions non remplies"}
    except Exception as e:
        result["ml_retraining"] = {"success": False, "error": str(e)}
    
    return result

def rl_performance(level: str = "basic") -> Dict[str, Any]:
    return get_manager(level).get_performance_summary()

def rl_analytics(level: str = "basic") -> Dict[str, Any]:
    return get_manager(level).get_feedback_analytics()

def rl_get_inspector_profiles() -> Dict[str, Any]:
    """Récupère tous les profils d'inspecteurs pour le chapitre 85"""
    try:
        manager = get_manager("basic")
        profiles = manager.store.get_inspector_profiles()
        return {
            "status": "success",
            "chapter": "chap85",
            "profiles": profiles,
            "count": len(profiles)
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def rl_retraining_status() -> Dict[str, Any]:
    """Récupère le statut de retraining pour le chapitre 85"""
    try:
        retraining_system = get_retraining_system()
        return {
            "chapter": "chap85",
            "should_retrain": retraining_system.should_retrain("chap85"),
            "feedback_count": retraining_system._get_feedback_count("chap85"),
            "feedback_quality": retraining_system._get_feedback_quality("chap85"),
            "min_feedbacks_required": retraining_system.min_feedbacks_for_retraining,
            "retraining_interval_hours": retraining_system.retraining_interval / 3600
        }
    except Exception as e:
        return {"error": str(e)}

def rl_trigger_retraining() -> Dict[str, Any]:
    """Déclenche manuellement le retraining pour le chapitre 85"""
    try:
        retraining_system = get_retraining_system()
        return retraining_system.retrain_model("chap85")
    except Exception as e:
        return {"success": False, "error": str(e)}