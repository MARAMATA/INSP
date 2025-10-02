# backend/src/shared/advanced_reinforcement_learning.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import json
import math
import os
import random
import psycopg2
import psycopg2.extras
import threading
import numpy as np
from collections import defaultdict, deque
import logging
import joblib
import pandas as pd

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_decision_thresholds(chapter: str) -> Dict[str, float]:
    """Charger les seuils de décision optimaux calculés basés sur les performances réelles des modèles ML"""
    try:
        from pathlib import Path
        import json
        
        # Charger les seuils optimaux calculés depuis les performances réelles
        thresholds_file = Path(__file__).resolve().parents[2] / "results" / chapter / "optimal_thresholds.json"
        if thresholds_file.exists():
            with open(thresholds_file, 'r') as f:
                thresholds = json.load(f)
            logger.info(f"✅ Seuils optimaux chargés pour {chapter}: seuil={thresholds.get('optimal_threshold', 0.5):.3f}")
            return thresholds
        else:
            logger.warning(f"Fichier de seuils optimaux non trouvé pour {chapter}: {thresholds_file}")
            # Seuils par défaut si fichier non trouvé
            return {
                "conforme": 0.2,
                "fraude": 0.8,
                "optimal_threshold": 0.4,
                "ml_threshold": 0.4,
                "decision_threshold": 0.4,
                "rl_threshold": 0.4,
                "confidence_high": 0.8,
                "confidence_medium": 0.6,
                "auc": 0.5,
                "f1": 0.5,
                "precision": 0.5,
                "recall": 0.5
            }
    except Exception as e:
        logger.error(f"Erreur chargement seuils optimaux pour {chapter}: {e}")
        return {
            "conforme": 0.2,
            "fraude": 0.8,
            "optimal_threshold": 0.4,
            "ml_threshold": 0.4,
            "decision_threshold": 0.4,
            "rl_threshold": 0.4,
            "confidence_high": 0.8,
            "confidence_medium": 0.6,
            "auc": 0.5,
            "f1": 0.5,
            "precision": 0.5,
            "recall": 0.5
        }

# ============
# Modèles de données étendus
# ============

@dataclass
class AdvancedDecisionRecord:
    """Record de décision enrichi avec contexte étendu"""
    chapter: str
    declaration_id: str
    timestamp: str
    context_key: str
    context_json: str
    action: str
    model_probability: Optional[float]
    rl_probability: float
    exploration_used: bool
    decision_source: str
    confidence_score: float  # Nouveau: score de confiance
    context_complexity: int  # Nouveau: complexité du contexte
    seasonal_factor: float   # Nouveau: facteur saisonnier
    bureau_risk_score: float # Nouveau: score de risque bureau
    extra_json: Optional[str] = None

@dataclass
class AdvancedFeedbackRecord:
    """Record de feedback enrichi avec métadonnées étendues"""
    chapter: str
    declaration_id: str
    timestamp: str
    inspector_id: str
    inspector_decision: bool
    inspector_confidence: float
    predicted_fraud: bool
    predicted_probability: float
    predicted_action: str
    notes: Optional[str] = None
    exploration_used: Optional[bool] = None
    context_key: Optional[str] = None
    context_json: Optional[str] = None
    # Nouveaux champs
    feedback_quality_score: float = 0.8  # Qualité estimée du feedback
    inspector_expertise_level: str = "standard"  # junior, standard, senior, expert
    review_time_seconds: Optional[int] = None
    feedback_category: str = "regular"  # regular, urgent, review, audit
    similar_cases_count: int = 0  # Nombre de cas similaires vus par l'inspecteur

@dataclass
class InspectorProfile:
    """Profil d'inspecteur avec historique de performance"""
    inspector_id: str
    name: str
    expertise_level: str
    specialization: List[str]  # Chapitres de spécialisation
    total_reviews: int
    accuracy_rate: float
    avg_confidence: float
    avg_review_time: float
    created_at: str
    last_active: str
    performance_trend: str  # improving, stable, declining

# ============
# Algorithmes RL avancés
# ============

class AdvancedEpsilonGreedyBandit:
    """
    Bandit epsilon-greedy avancé avec:
    - Decay adaptatif de epsilon
    - Upper Confidence Bounds (UCB)
    - Thompson Sampling
    - Contextual features weighting
    """
    
    def __init__(self, 
                 epsilon: float = 0.3,
                 min_epsilon: float = 0.05,
                 epsilon_decay: float = 0.95,  # Décroissance plus rapide
                 ucb_c: float = 2.0,
                 min_visits_for_exploit: int = 5,  # Moins de visites nécessaires
                 context_weight_decay: float = 0.9):
        
        self.epsilon = float(epsilon)
        self.initial_epsilon = float(epsilon)
        self.min_epsilon = float(min_epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.ucb_c = float(ucb_c)
        self.min_visits_for_exploit = int(min_visits_for_exploit)
        self.context_weight_decay = float(context_weight_decay)
        
        # Statistiques étendues: stats[(ctx, action)] = {"n": int, "q": float, "variance": float, "last_update": timestamp}
        self.stats: Dict[Tuple[str, str], Dict[str, Union[float, str]]] = {}
        
        # Historique des récompenses pour Thompson Sampling
        self.reward_history: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        
        # Poids des features contextuelles
        self.context_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Compteur global pour epsilon decay
        self.global_step = 0
        
        self._lock = threading.Lock()

    def _key(self, ctx: str, action: str) -> Tuple[str, str]:
        return (ctx, action)

    def _ensure(self, ctx: str, action: str):
        k = self._key(ctx, action)
        if k not in self.stats:
            self.stats[k] = {
                "n": 0.0, 
                "q": 0.0, 
                "variance": 0.0,
                "last_update": datetime.utcnow().isoformat()
            }
            self.reward_history[k] = []

    def _calculate_ucb_score(self, ctx: str, action: str) -> float:
        """Calcule le score Upper Confidence Bound"""
        k = self._key(ctx, action)
        stats = self.stats[k]
        
        if stats["n"] == 0:
            return float('inf')  # Favorise l'exploration des actions non testées
        
        # Total des visites pour ce contexte
        total_visits = sum(
            self.stats[self._key(ctx, a)]["n"] 
            for a in ["flag", "pass"] 
            if self._key(ctx, a) in self.stats
        )
        
        if total_visits <= 0:
            return float('inf')
        
        # UCB formula: Q(a) + c * sqrt(ln(N) / n(a))
        confidence_bonus = self.ucb_c * math.sqrt(math.log(total_visits) / stats["n"])
        return stats["q"] + confidence_bonus

    def _thompson_sampling_choice(self, ctx: str) -> str:
        """Choix par Thompson Sampling (échantillonnage bayésien)"""
        samples = {}
        
        for action in ["flag", "pass"]:
            k = self._key(ctx, action)
            rewards = self.reward_history[k]
            
            if len(rewards) < 2:
                # Pas assez de données, utiliser des priors optimistes
                samples[action] = random.gauss(0.5, 0.3)
            else:
                # Échantillonner depuis la distribution des récompenses
                if len(rewards) > 0:
                    mean_reward = np.mean(rewards)
                    std_reward = max(np.std(rewards), 0.1)  # Minimum de variance
                    samples[action] = random.gauss(mean_reward, std_reward / math.sqrt(len(rewards)))
                else:
                    # Fallback si pas de récompenses
                    samples[action] = random.gauss(0.5, 0.3)
        
        # Retourner l'action avec le plus haut échantillon
        return max(samples.keys(), key=lambda a: samples[a])

    def _update_context_weights(self, ctx: str, reward: float):
        """Met à jour les poids des features contextuelles"""
        # Extraction simple des features du contexte
        if "|" in ctx:
            features = ctx.split("|")
            for feature in features:
                if ":" in feature:
                    feature_name = feature.split(":")[0]
                    # Ajuste le poids selon la récompense
                    if reward > 0.5:
                        self.context_weights[feature_name] *= 1.05  # Augmente légèrement
                    else:
                        self.context_weights[feature_name] *= 0.98  # Diminue légèrement
                    
                    # Bornes des poids
                    self.context_weights[feature_name] = max(0.1, min(2.0, self.context_weights[feature_name]))

    def choose(self, ctx: str, strategy: str = "epsilon_greedy") -> Tuple[str, bool, Dict[str, float]]:
        """
        Choix d'action avec stratégies multiples
        
        Args:
            ctx: Contexte de décision
            strategy: "epsilon_greedy", "ucb", "thompson", "hybrid"
            
        Returns:
            (action, exploration_used, strategy_info)
        """
        with self._lock:
            # Assurer que les actions existent
            for action in ("flag", "pass"):
                self._ensure(ctx, action)

            # Décroissance de epsilon
            current_epsilon = max(
                self.min_epsilon, 
                self.epsilon * (self.epsilon_decay ** self.global_step)
            )
            self.global_step += 1

            # Informations de stratégie
            strategy_info = {
                "current_epsilon": current_epsilon,
                "global_step": self.global_step,
                "strategy_used": strategy
            }

            n_flag = self.stats[(ctx, "flag")]["n"]
            n_pass = self.stats[(ctx, "pass")]["n"]
            total_n = n_flag + n_pass

            # Stratégies de choix
            if strategy == "ucb" and total_n >= self.min_visits_for_exploit:
                # Upper Confidence Bounds
                ucb_flag = self._calculate_ucb_score(ctx, "flag")
                ucb_pass = self._calculate_ucb_score(ctx, "pass")
                
                strategy_info.update({
                    "ucb_flag": ucb_flag,
                    "ucb_pass": ucb_pass
                })
                
                action = "flag" if ucb_flag >= ucb_pass else "pass"
                return (action, False, strategy_info)

            elif strategy == "thompson" and total_n >= self.min_visits_for_exploit:
                # Thompson Sampling
                action = self._thompson_sampling_choice(ctx)
                return (action, False, strategy_info)

            elif strategy == "hybrid":
                # Stratégie hybride: UCB pour exploration, epsilon-greedy pour exploitation
                explore = (random.random() < current_epsilon and 
                          current_epsilon > self.min_epsilon and
                          total_n < self.min_visits_for_exploit)
                
                if explore:
                    # Phase d'exploration: utiliser UCB
                    if total_n >= 2:
                        ucb_flag = self._calculate_ucb_score(ctx, "flag")
                        ucb_pass = self._calculate_ucb_score(ctx, "pass")
                        action = "flag" if ucb_flag >= ucb_pass else "pass"
                    else:
                        action = random.choice(["flag", "pass"])
                    return (action, True, strategy_info)
                else:
                    # Phase d'exploitation: epsilon-greedy classique
                    q_flag = self.stats[(ctx, "flag")]["q"]
                    q_pass = self.stats[(ctx, "pass")]["q"]
                    action = "flag" if q_flag >= q_pass else "pass"
                    return (action, False, strategy_info)

            else:  # epsilon_greedy (défaut)
                # Exploration basée sur epsilon uniquement
                explore = random.random() < current_epsilon
                
                if explore:
                    action = random.choice(["flag", "pass"])
                    return (action, True, strategy_info)
                else:
                    q_flag = self.stats[(ctx, "flag")]["q"]
                    q_pass = self.stats[(ctx, "pass")]["q"]
                    action = "flag" if q_flag >= q_pass else "pass"
                    return (action, False, strategy_info)

    def update(self, ctx: str, action: str, reward: float, **kwargs) -> Dict[str, Any]:
        """
        Met à jour avec statistiques étendues
        
        Args:
            ctx: Contexte
            action: Action prise
            reward: Récompense reçue
            **kwargs: Métadonnées supplémentaires (confidence, inspector_expertise, etc.)
            
        Returns:
            Statistiques de mise à jour
        """
        reward = float(reward)
        
        with self._lock:
            self._ensure(ctx, action)
            
            k = self._key(ctx, action)
            stats = self.stats[k]
            
            # S'assurer que toutes les valeurs sont des float
            stats["n"] = float(stats["n"])
            stats["q"] = float(stats["q"])
            stats["variance"] = float(stats["variance"])
            
            # Mise à jour des statistiques de base
            n = stats["n"] + 1.0
            old_q = stats["q"]
            new_q = old_q + (reward - old_q) / n
            
            # Mise à jour de la variance (pour Thompson Sampling)
            if n > 1:
                old_variance = float(stats["variance"])  # Convertir en float
                new_variance = ((n - 2) * old_variance + (reward - old_q) * (reward - new_q)) / (n - 1)
                stats["variance"] = new_variance
            else:
                stats["variance"] = 0.1  # Variance initiale
            
            # Mise à jour des valeurs
            stats["n"] = n
            stats["q"] = new_q
            stats["last_update"] = datetime.utcnow().isoformat()
            
            # Décrémenter epsilon pour réduire l'exploration au fil du temps
            self.global_step += 1
            if self.global_step % 10 == 0:  # Décrémenter tous les 10 updates
                self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Historique des récompenses (limité pour éviter la croissance infinie)
            self.reward_history[k].append(reward)
            if len(self.reward_history[k]) > 1000:  # Garde seulement les 1000 dernières
                self.reward_history[k] = self.reward_history[k][-1000:]
            
            # Mise à jour des poids contextuels
            self._update_context_weights(ctx, reward)
            
            # Pondération par expertise de l'inspecteur
            inspector_expertise = kwargs.get("inspector_expertise", "standard")
            expertise_weights = {
                "junior": 0.7,
                "standard": 1.0, 
                "senior": 1.3,
                "expert": 1.5
            }
            
            weighted_reward = reward * expertise_weights.get(inspector_expertise, 1.0)
            
            return {
                "updated_q": new_q,
                "n_visits": n,
                "variance": stats["variance"],
                "weighted_reward": weighted_reward,
                "context_weights_updated": len(self.context_weights),
                "improvement": new_q - old_q
            }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retourne des métriques de performance détaillées"""
        with self._lock:
            total_actions = sum(stats["n"] for stats in self.stats.values())
            
            if total_actions == 0:
                return {"error": "Aucune action enregistrée"}
            
            # Répartition des actions
            flag_actions = sum(stats["n"] for (ctx, action), stats in self.stats.items() if action == "flag")
            pass_actions = sum(stats["n"] for (ctx, action), stats in self.stats.items() if action == "pass")
            
            # Q-values moyennes
            flag_q_values = [stats["q"] for (ctx, action), stats in self.stats.items() if action == "flag" and stats["n"] > 0]
            pass_q_values = [stats["q"] for (ctx, action), stats in self.stats.items() if action == "pass" and stats["n"] > 0]
            
            avg_q_flag = np.mean(flag_q_values) if flag_q_values else 0.0
            avg_q_pass = np.mean(pass_q_values) if pass_q_values else 0.0
            
            # Contextes les plus actifs
            context_activity = defaultdict(int)
            for (ctx, action), stats in self.stats.items():
                context_activity[ctx] += stats["n"]
            
            top_contexts = sorted(context_activity.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "total_actions": total_actions,
                "flag_percentage": (flag_actions / total_actions) * 100 if total_actions > 0 else 0,
                "pass_percentage": (pass_actions / total_actions) * 100 if total_actions > 0 else 0,
                "avg_q_flag": float(avg_q_flag) if not np.isnan(avg_q_flag) else 0,
                "avg_q_pass": float(avg_q_pass) if not np.isnan(avg_q_pass) else 0,
                "unique_contexts": len(set(ctx for ctx, action in self.stats.keys())),
                "current_epsilon": max(self.min_epsilon, self.epsilon * (self.epsilon_decay ** self.global_step)),
                "top_contexts": [(ctx, int(count)) for ctx, count in top_contexts],
                "context_weights_count": len(self.context_weights),
                "global_step": self.global_step
            }

    def export(self) -> Dict[str, Any]:
        """Export complet avec toutes les données"""
        with self._lock:
            return {
                "stats": {f"{ctx}|{act}": dict(stats) for (ctx, act), stats in self.stats.items()},
                "reward_history": {f"{ctx}|{act}": rewards for (ctx, act), rewards in self.reward_history.items()},
                "context_weights": dict(self.context_weights),
                "global_step": self.global_step,
                "current_epsilon": max(self.min_epsilon, self.epsilon * (self.epsilon_decay ** self.global_step)),
                "hyperparameters": {
                    "epsilon": self.epsilon,
                    "min_epsilon": self.min_epsilon,
                    "epsilon_decay": self.epsilon_decay,
                    "ucb_c": self.ucb_c,
                    "min_visits_for_exploit": self.min_visits_for_exploit
                }
            }

    def import_(self, payload: Dict[str, Any]) -> None:
        """Import complet avec toutes les données"""
        with self._lock:
            self.stats.clear()
            self.reward_history.clear()
            self.context_weights.clear()
            
            # Import des statistiques
            for k, v in payload.get("stats", {}).items():
                try:
                    ctx, act = k.split("|", 1)
                    # Convertir tous les nombres en float pour éviter les erreurs decimal
                    if isinstance(v, dict):
                        converted_v = {}
                        for key, value in v.items():
                            if isinstance(value, (int, float, str)) and str(value).replace('.', '').replace('-', '').isdigit():
                                converted_v[key] = float(value)
                            else:
                                converted_v[key] = value
                        self.stats[(ctx, act)] = converted_v
                    else:
                        self.stats[(ctx, act)] = v
                except Exception as e:
                    logger.warning(f"Erreur import stats pour {k}: {e}")
            
            # Import de l'historique des récompenses
            for k, rewards in payload.get("reward_history", {}).items():
                try:
                    ctx, act = k.split("|", 1)
                    self.reward_history[(ctx, act)] = list(rewards)
                except Exception as e:
                    logger.warning(f"Erreur import reward_history pour {k}: {e}")
            
            # Import des poids contextuels
            self.context_weights.update(payload.get("context_weights", {}))
            
            # Import des paramètres globaux
            self.global_step = payload.get("global_step", 0)
            
            # Import des hyperparamètres si présents
            hyperparams = payload.get("hyperparameters", {})
            if hyperparams:
                self.epsilon = hyperparams.get("epsilon", self.epsilon)
                self.min_epsilon = hyperparams.get("min_epsilon", self.min_epsilon)
                self.epsilon_decay = hyperparams.get("epsilon_decay", self.epsilon_decay)
                self.ucb_c = hyperparams.get("ucb_c", self.ucb_c)
                self.min_visits_for_exploit = hyperparams.get("min_visits_for_exploit", self.min_visits_for_exploit)

    def reset(self) -> None:
        """Remet à zéro toutes les statistiques du bandit"""
        with self._lock:
            self.stats.clear()
            self.reward_history.clear()
            self.context_weights.clear()
            self.global_step = 0
            self.epsilon = self.initial_epsilon
            logger.info("Bandit réinitialisé complètement")

    @property
    def total_pulls(self) -> int:
        """Retourne le nombre total de pulls effectués"""
        with self._lock:
            return sum(stats["n"] for stats in self.stats.values())


# ============
# Stockage avancé avec métriques de performance
# ============

class AdvancedRLStore:
    """Store RL avancé avec tables étendues et analytics - PostgreSQL"""
    
    def __init__(self, chapter: str):
        self.chapter = chapter
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'user': 'maramata',
            'password': 'maramata',
            'database': 'INSPECT_IA'
        }
        self.connection = None
        self._init_db()

    def _connect(self):
        """Obtient une connexion PostgreSQL"""
        if self.connection is None or self.connection.closed:
            self.connection = psycopg2.connect(**self.db_config)
        return self.connection

    def _init_db(self):
        """Initialise les tables étendues PostgreSQL"""
        con = self._connect()
        cur = con.cursor()
        
        # Table des décisions (étendue) - PostgreSQL
        cur.execute("""
        CREATE TABLE IF NOT EXISTS advanced_decisions(
            decision_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            chapter_id VARCHAR(10) NOT NULL,
            declaration_id VARCHAR(100),
            ts VARCHAR(50),
            context_key VARCHAR(255),
            context_json JSONB,
            action VARCHAR(50),
            model_proba DECIMAL(8,6),
            rl_proba DECIMAL(8,6),
            exploration BOOLEAN,
            decision_source VARCHAR(50),
            confidence_score DECIMAL(8,6) DEFAULT 0.5,
            context_complexity INTEGER DEFAULT 1,
            seasonal_factor DECIMAL(8,6) DEFAULT 1.0,
            bureau_risk_score DECIMAL(8,6) DEFAULT 0.5,
            extra_json JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Table des feedbacks (étendue) - PostgreSQL
        cur.execute("""
        CREATE TABLE IF NOT EXISTS advanced_feedbacks(
            feedback_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            chapter_id VARCHAR(10) NOT NULL,
            declaration_id VARCHAR(100),
            ts VARCHAR(50),
            inspector_id VARCHAR(100),
            inspector_decision INTEGER,
            inspector_confidence DECIMAL(8,6),
            predicted_fraud INTEGER,
            predicted_probability DECIMAL(8,6),
            predicted_action VARCHAR(50),
            notes TEXT,
            exploration_used BOOLEAN,
            context_key VARCHAR(255),
            context_json JSONB,
            feedback_quality_score DECIMAL(8,6) DEFAULT 0.8,
            inspector_expertise_level VARCHAR(50) DEFAULT 'standard',
            review_time_seconds INTEGER,
            feedback_category VARCHAR(50) DEFAULT 'regular',
            similar_cases_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Table des profils d'inspecteurs - PostgreSQL
        cur.execute("""
        CREATE TABLE IF NOT EXISTS inspector_profiles(
            inspector_id VARCHAR(100) PRIMARY KEY,
            chapter_id VARCHAR(10) NOT NULL,
            name VARCHAR(255),
            expertise_level VARCHAR(50) DEFAULT 'standard',
            specialization JSONB,  -- JSON array of chapters
            total_reviews INTEGER DEFAULT 0,
            accuracy_rate DECIMAL(8,6) DEFAULT 0.0,
            avg_confidence DECIMAL(8,6) DEFAULT 0.5,
            avg_review_time DECIMAL(8,6) DEFAULT 300.0,
            performance_trend VARCHAR(50) DEFAULT 'stable',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Table des politiques (étendue) - PostgreSQL
        cur.execute("""
        CREATE TABLE IF NOT EXISTS advanced_policies(
            k VARCHAR(255) PRIMARY KEY,
            chapter_id VARCHAR(10) NOT NULL,
            json_blob JSONB,
            version INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Table des métriques de performance RL (séparée de la table principale)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS rl_performance_metrics(
            metric_id SERIAL PRIMARY KEY,
            chapter TEXT,
            metric_name TEXT,
            metric_value REAL,
            metric_metadata TEXT,  -- JSON with additional context
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Index pour optimiser les requêtes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ts ON advanced_decisions(ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedbacks_inspector ON advanced_feedbacks(inspector_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_feedbacks_ts ON advanced_feedbacks(ts);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON rl_performance_metrics(metric_name, recorded_at);")
        
        con.commit()
        con.close()

    def save_advanced_decision(self, d: AdvancedDecisionRecord) -> None:
        """Sauvegarde une décision avec contexte étendu - PostgreSQL"""
        con = self._connect()
        cur = con.cursor()
        cur.execute("""
          INSERT INTO advanced_decisions(
            chapter_id, declaration_id, ts, context_key, context_json, action, model_proba, 
            rl_proba, exploration, decision_source, confidence_score, 
            context_complexity, seasonal_factor, bureau_risk_score, extra_json
          )
          VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            self.chapter, d.declaration_id, d.timestamp, d.context_key, 
            json.dumps(d.context_json) if d.context_json else None, d.action,
            d.model_probability, d.rl_probability, d.exploration_used, d.decision_source,
            d.confidence_score, d.context_complexity, d.seasonal_factor, d.bureau_risk_score, 
            json.dumps(d.extra_json) if d.extra_json else None
        ))
        con.commit()
        cur.close()

    def save_advanced_feedback(self, f: AdvancedFeedbackRecord) -> None:
        """Sauvegarde un feedback avec métadonnées étendues - PostgreSQL"""
        con = self._connect()
        cur = con.cursor()
        cur.execute("""
          INSERT INTO advanced_feedbacks(
            chapter_id, declaration_id, ts, inspector_id, inspector_decision, inspector_confidence,
            predicted_fraud, predicted_probability, predicted_action, notes, exploration_used,
            context_key, context_json, feedback_quality_score, inspector_expertise_level,
            review_time_seconds, feedback_category, similar_cases_count
          )
          VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            self.chapter, f.declaration_id, f.timestamp, f.inspector_id, int(f.inspector_decision), 
            float(f.inspector_confidence), int(f.predicted_fraud), float(f.predicted_probability), 
            f.predicted_action, f.notes, f.exploration_used,
            f.context_key, json.dumps(f.context_json) if f.context_json else None, 
            f.feedback_quality_score, f.inspector_expertise_level,
            f.review_time_seconds, f.feedback_category, f.similar_cases_count
        ))
        con.commit()
        cur.close()

    def save_or_update_inspector_profile(self, profile: InspectorProfile) -> None:
        """Sauvegarde ou met à jour un profil d'inspecteur - PostgreSQL"""
        con = self._connect()
        cur = con.cursor()
        
        specialization_json = json.dumps(profile.specialization)
        
        cur.execute("""
        INSERT INTO inspector_profiles(
            profile_id, inspector_id, chapter_id, name, expertise_level, specialization, total_reviews,
            accuracy_rate, avg_confidence, avg_review_time, performance_trend, last_active
        )
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (inspector_id, chapter_id) DO UPDATE SET
            name = EXCLUDED.name,
            expertise_level = EXCLUDED.expertise_level,
            specialization = EXCLUDED.specialization,
            total_reviews = EXCLUDED.total_reviews,
            accuracy_rate = EXCLUDED.accuracy_rate,
            avg_confidence = EXCLUDED.avg_confidence,
            avg_review_time = EXCLUDED.avg_review_time,
            performance_trend = EXCLUDED.performance_trend,
            last_active = EXCLUDED.last_active
        """, (
            f"{profile.inspector_id}_{self.chapter}", profile.inspector_id, self.chapter, profile.name, profile.expertise_level, specialization_json,
            profile.total_reviews, profile.accuracy_rate, profile.avg_confidence,
            profile.avg_review_time, profile.performance_trend, profile.last_active
        ))
        con.commit()
        cur.close()

    def get_inspector_profile(self, inspector_id: str) -> Optional[InspectorProfile]:
        """Récupère le profil d'un inspecteur - PostgreSQL"""
        con = self._connect()
        cur = con.cursor()
        cur.execute("""
        SELECT inspector_id, name, expertise_level, specialization, total_reviews,
               accuracy_rate, avg_confidence, avg_review_time, performance_trend,
               created_at, last_active
        FROM inspector_profiles WHERE inspector_id = %s AND chapter_id = %s
        """, (inspector_id, self.chapter))
        
        row = cur.fetchone()
        cur.close()
        
        if not row:
            return None
            
        return InspectorProfile(
            inspector_id=row[0],
            name=row[1] or f"Inspector {row[0]}",
            expertise_level=row[2],
            specialization=json.loads(row[3]) if row[3] else [],
            total_reviews=int(row[4]) if row[4] else 0,
            accuracy_rate=float(row[5]) if row[5] else 0.0,
            avg_confidence=float(row[6]) if row[6] else 0.0,
            avg_review_time=float(row[7]) if row[7] else 0.0,
            created_at=row[8],
            last_active=row[9],
            performance_trend=row[10]
        )

    def record_performance_metric(self, metric_name: str, value: float, metadata: Dict[str, Any] = None) -> None:
        """Enregistre une métrique de performance"""
        con = self._connect()
        cur = con.cursor()
        
        metadata_json = json.dumps(metadata or {})
        
        cur.execute("""
        INSERT INTO rl_performance_metrics(chapter, metric_name, metric_value, metric_metadata)
        VALUES(%s,%s,%s,%s)
        """, (self.chapter, metric_name, value, metadata_json))
        
        con.commit()
        con.close()

    def get_performance_trends(self, metric_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Récupère les tendances d'une métrique sur N jours"""
        con = self._connect()
        cur = con.cursor()
        
        since_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        cur.execute("""
        SELECT metric_value, metric_metadata, recorded_at
        FROM rl_performance_metrics 
        WHERE chapter = %s AND metric_name = %s AND recorded_at >= %s
        ORDER BY recorded_at ASC
        """, (self.chapter, metric_name, since_date))
        
        rows = cur.fetchall()
        con.close()
        
        return [
            {
                "value": row[0],
                "metadata": json.loads(row[1]) if row[1] else {},
                "timestamp": row[2]
            }
            for row in rows
        ]

    def get_advanced_analytics(self) -> Dict[str, Any]:
        """Analytics avancés avec drill-down par inspecteur, période, etc."""
        con = self._connect()
        cur = con.cursor()
        
        analytics = {}
        
        # Performance par inspecteur (limité à 10)
        cur.execute("""
        SELECT 
            inspector_id,
            COUNT(*) as total_feedbacks,
            AVG(CASE WHEN inspector_decision = predicted_fraud THEN 1.0 ELSE 0.0 END) as accuracy,
            AVG(inspector_confidence) as avg_confidence,
            AVG(review_time_seconds) as avg_review_time,
            inspector_expertise_level
        FROM advanced_feedbacks
        WHERE review_time_seconds IS NOT NULL
        GROUP BY inspector_id, inspector_expertise_level
        HAVING COUNT(*) >= 5
        ORDER BY accuracy DESC
        LIMIT 10
        """)
        
        inspector_stats = []
        for row in cur.fetchall():
            inspector_stats.append({
                "inspector_id": row[0],
                "total_feedbacks": row[1],
                "accuracy": round(row[2], 3),
                "avg_confidence": round(row[3], 3),
                "avg_review_time": round(row[4], 1) if row[4] else None,
                "expertise_level": row[5]
            })
        
        analytics["inspector_performance"] = inspector_stats
        
        # Tendances temporelles (derniers 7 jours seulement)
        cur.execute("""
        SELECT 
            DATE(ts::timestamp) as date,
            COUNT(*) as decisions_count,
            AVG(rl_proba) as avg_fraud_probability,
            SUM(CASE WHEN action = 'flag' THEN 1 ELSE 0 END) as flags_count,
            AVG(confidence_score) as avg_confidence
        FROM advanced_decisions
        WHERE ts::timestamp >= NOW() - INTERVAL '7 days'
        GROUP BY DATE(ts::timestamp)
        ORDER BY date ASC
        LIMIT 7
        """)
        
        temporal_trends = []
        for row in cur.fetchall():
            temporal_trends.append({
                "date": row[0],
                "decisions_count": row[1],
                "avg_fraud_probability": round(row[2], 3),
                "flags_count": row[3],
                "flag_rate": round(row[3] / row[1], 3) if row[1] > 0 else 0,
                "avg_confidence": round(row[4], 3)
            })
        
        analytics["temporal_trends"] = temporal_trends
        
        # Performance par contexte (limité à 10)
        cur.execute("""
        SELECT 
            context_key,
            COUNT(*) as decisions_count,
            AVG(rl_proba) as avg_fraud_prob,
            AVG(confidence_score) as avg_confidence,
            SUM(CASE WHEN exploration = true THEN 1 ELSE 0 END) as exploration_count
        FROM advanced_decisions
        GROUP BY context_key
        HAVING COUNT(*) >= 10
        ORDER BY COUNT(*) DESC
        LIMIT 10
        """)
        
        context_stats = []
        for row in cur.fetchall():
            context_stats.append({
                "context_key": row[0],
                "decisions_count": row[1],
                "avg_fraud_probability": round(row[2], 3),
                "avg_confidence": round(row[3], 3),
                "exploration_rate": round(row[4] / row[1], 3) if row[1] > 0 else 0
            })
        
        analytics["context_performance"] = context_stats
        
        con.close()
        return analytics
    
    def _record_performance_metric(self, metric_name: str, metric_value: float, context: Dict[str, Any] = None):
        """Enregistre une métrique de performance"""
        try:
            con = self._connect()
            cur = con.cursor()
            
            context_json = json.dumps(context or {})
            
            cur.execute("""
                INSERT INTO rl_performance_metrics(
                    chapter_id, metric_name, metric_value, context_json
                ) VALUES (%s, %s, %s, %s)
            """, (self.chapter, metric_name, metric_value, context_json))
            
            con.commit()
            cur.close()
            con.close()
            
        except Exception as e:
            logger.error(f"Erreur enregistrement métrique {metric_name}: {e}")
    
    def get_inspector_profiles(self) -> List[Dict[str, Any]]:
        """Récupère tous les profils d'inspecteurs pour le chapitre"""
        try:
            con = self._connect()
            cur = con.cursor()
            
            cur.execute("""
                SELECT inspector_id, name, expertise_level, specialization, 
                       total_reviews, accuracy_rate, avg_confidence, 
                       avg_review_time, performance_trend, created_at, last_active
                FROM inspector_profiles 
                WHERE chapter_id = %s
                ORDER BY total_reviews DESC, accuracy_rate DESC
            """, (self.chapter,))
            
            profiles = []
            for row in cur.fetchall():
                profiles.append({
                    "inspector_id": row[0],
                    "name": row[1],
                    "expertise_level": row[2],
                    "specialization": json.loads(row[3]) if row[3] else [],
                    "total_reviews": row[4],
                    "accuracy_rate": float(row[5]) if row[5] else 0.0,
                    "avg_confidence": float(row[6]) if row[6] else 0.0,
                    "avg_review_time": float(row[7]) if row[7] else 0.0,
                    "performance_trend": row[8],
                    "created_at": row[9].isoformat() if row[9] else None,
                    "last_active": row[10].isoformat() if row[10] else None
                })
            
            cur.close()
            con.close()
            
            return profiles
            
        except Exception as e:
            logger.error(f"Erreur récupération profils inspecteurs: {e}")
            return []

# ============
# Gestionnaire RL avancé
# ============

class AdvancedRLManager:
    """
    Gestionnaire RL avancé avec:
    - Stratégies multiples (epsilon-greedy, UCB, Thompson Sampling)
    - Gestion des profils d'inspecteurs
    - Analytics avancés
    - Adaptation contextuelle
    - Métriques de performance en temps réel
    """
    
    def __init__(self, chapter: str, epsilon: float = 0.1, strategy: str = "hybrid"):
        self.chapter = chapter
        self.strategy = strategy
        self.bandit = AdvancedEpsilonGreedyBandit(epsilon=epsilon)
        self.store = AdvancedRLStore(chapter)
        self._load_bandit()
        
        # Charger le modèle ML entraîné
        self.ml_model = self._load_ml_model()
        
        # Cache des profils d'inspecteurs
        self.inspector_cache: Dict[str, InspectorProfile] = {}
        self.cache_ttl = 3600  # 1 heure
        self.last_cache_update = 0
        
        # Métriques en temps réel
        self.session_metrics = {
            "predictions_count": 0,
            "feedbacks_count": 0,
            "avg_confidence": 0.0,
            "exploration_rate": 0.0,
            "session_start": datetime.utcnow().isoformat()
        }

    def _load_ml_model(self):
        """Charge le meilleur modèle ML avancé pour le chapitre"""
        try:
            # Déterminer le meilleur modèle selon F1 de validation
            model_priority = {
                "chap30": "xgboost",   # XGBoost - Validation F1: 0.9821 ⭐ Test: F1=0.9811, AUC=0.9997
                "chap84": "xgboost",   # XGBoost - Validation F1: 0.9891 ⭐ Test: F1=0.9888, AUC=0.9997
                "chap85": "xgboost"    # XGBoost - Validation F1: 0.9781 ⭐ Test: F1=0.9808, AUC=0.9993
            }
            
            best_model = model_priority.get(self.chapter, "lightgbm")
            
            # Utiliser les nouveaux modèles ML avancés
            models_dir = Path(__file__).resolve().parents[2] / "results" / self.chapter / "models"
            model_path = models_dir / f"{best_model}_model.pkl"
            
            if model_path.exists():
                model = joblib.load(model_path)
                # Vérifier que le modèle est bien un pipeline complet
                model_type = str(type(model))
                if 'Pipeline' in model_type:
                    logger.info(f"✅ Modèle ML avancé chargé pour {self.chapter}: {best_model}")
                else:
                    logger.warning(f"Modèle {self.chapter} n'est pas un pipeline complet: {model_type}")
                return model
            else:
                logger.warning(f"Modèle ML avancé non trouvé pour {self.chapter}: {model_path}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur chargement modèle ML pour {self.chapter}: {e}")
            return None

    def _get_required_features(self) -> List[str]:
        """Retourne la liste des features requises pour le modèle ML du chapitre"""
        # Features communes à tous les chapitres
        common_features = [
            'VALEUR_CAF', 'VALEUR_DOUANE', 'MONTANT_LIQUIDATION', 'POIDS_NET',
            'VALEUR_UNITAIRE_KG', 'TAUX_DROITS_PERCENT', 'RATIO_DOUANE_CAF',
            'NUMERO_ARTICLE', 'PRECISION_UEMOA',
            'BIENAYME_CHEBYCHEV_SCORE', 'TEI_CALCULE', 'MIRROR_TEI_SCORE', 
            'MIRROR_TEI_DEVIATION', 'SPECTRAL_CLUSTER_SCORE', 
            'HIERARCHICAL_CLUSTER_SCORE', 'ADMIN_VALUES_SCORE', 
            'ADMIN_VALUES_DEVIATION', 'COMPOSITE_FRAUD_SCORE', 'RATIO_POIDS_VALEUR',
            'CODE_PRODUIT_STR', 'PAYS_ORIGINE_STR', 'PAYS_PROVENANCE_STR',
            'BUREAU', 'REGIME_FISCAL', 'NUMERO_DPI'
        ]
        
        # Features spécifiques par chapitre
        chapter_specific = {
            "chap30": [
                'BUSINESS_GLISSEMENT_COSMETIQUE', 'BUSINESS_GLISSEMENT_PAYS_COSMETIQUES',
                'BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'BUSINESS_RISK_PAYS_HIGH',
                'BUSINESS_ORIGINE_DIFF_PROVENANCE', 'BUSINESS_REGIME_PREFERENTIEL',
                'BUSINESS_REGIME_NORMAL', 'BUSINESS_VALEUR_ELEVEE',
                'BUSINESS_VALEUR_EXCEPTIONNELLE', 'BUSINESS_POIDS_ELEVE',
                'BUSINESS_DROITS_ELEVES', 'BUSINESS_RATIO_LIQUIDATION_CAF',
                'BUSINESS_RATIO_DOUANE_CAF', 'BUSINESS_IS_MEDICAMENT',
                'BUSINESS_IS_ANTIPALUDEEN', 'BUSINESS_IS_PRECISION_UEMOA',
                'BUSINESS_ARTICLES_MULTIPLES', 'BUSINESS_AVEC_DPI'
            ],
            "chap84": [
                'BUSINESS_GLISSEMENT_MACHINE', 'BUSINESS_GLISSEMENT_PAYS_MACHINES',
                'BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'BUSINESS_RISK_PAYS_HIGH',
                'BUSINESS_ORIGINE_DIFF_PROVENANCE', 'BUSINESS_REGIME_PREFERENTIEL',
                'BUSINESS_REGIME_NORMAL', 'BUSINESS_VALEUR_ELEVEE',
                'BUSINESS_VALEUR_EXCEPTIONNELLE', 'BUSINESS_POIDS_ELEVE',
                'BUSINESS_DROITS_ELEVES', 'BUSINESS_RATIO_LIQUIDATION_CAF',
                'BUSINESS_RATIO_DOUANE_CAF', 'BUSINESS_IS_MACHINE',
                'BUSINESS_IS_ELECTRONIQUE', 'BUSINESS_IS_PRECISION_UEMOA',
                'BUSINESS_ARTICLES_MULTIPLES', 'BUSINESS_AVEC_DPI'
            ],
            "chap85": [
                'BUSINESS_GLISSEMENT_ELECTRONIQUE', 'BUSINESS_GLISSEMENT_PAYS_ELECTRONIQUES',
                'BUSINESS_GLISSEMENT_RATIO_SUSPECT', 'BUSINESS_RISK_PAYS_HIGH',
                'BUSINESS_ORIGINE_DIFF_PROVENANCE', 'BUSINESS_REGIME_PREFERENTIEL',
                'BUSINESS_REGIME_NORMAL', 'BUSINESS_VALEUR_ELEVEE',
                'BUSINESS_VALEUR_EXCEPTIONNELLE', 'BUSINESS_POIDS_FAIBLE',
                'BUSINESS_DROITS_ELEVES', 'BUSINESS_RATIO_LIQUIDATION_CAF',
                'BUSINESS_RATIO_DOUANE_CAF', 'BUSINESS_IS_ELECTRONIQUE',
                'BUSINESS_IS_TELEPHONE', 'BUSINESS_IS_PRECISION_UEMOA',
                'BUSINESS_ARTICLES_MULTIPLES', 'BUSINESS_AVEC_DPI'
            ]
        }
        
        return common_features + chapter_specific.get(self.chapter, [])

    def _load_bandit(self):
        """Charge l'état du bandit depuis le stockage"""
        try:
            con = self.store._connect()
            cur = con.cursor()
            cur.execute("SELECT json_blob FROM advanced_policies WHERE k = 'bandit_state' ORDER BY version DESC LIMIT 1")
            row = cur.fetchone()
            con.close()
            
            if row and row[0]:
                state = json.loads(row[0])
                self.bandit.import_(state)
                logger.info(f"État RL chargé pour {self.chapter} (version {state.get('version', 1)})")
        except Exception as e:
            logger.warning(f"Impossible de charger l'état RL pour {self.chapter}: {e}")

    def _persist_bandit(self):
        """Sauvegarde l'état du bandit avec versioning"""
        try:
            state = self.bandit.export()
            state["version"] = int(datetime.utcnow().timestamp())
            state["chapter"] = self.chapter
            
            con = self.store._connect()
            cur = con.cursor()
            
            # Utiliser INSERT OR REPLACE pour éviter les conflits
            cur.execute("""
            INSERT INTO advanced_policies(k, json_blob, version, updated_at)
            VALUES('bandit_state', %s, %s, %s)
            ON CONFLICT (k) DO UPDATE SET
                json_blob = EXCLUDED.json_blob,
                version = EXCLUDED.version,
                updated_at = EXCLUDED.updated_at
            """, (json.dumps(state), state["version"], datetime.utcnow().isoformat()))
            con.commit()
            con.close()
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde état RL pour {self.chapter}: {e}")

    def _get_inspector_profile(self, inspector_id: str) -> InspectorProfile:
        """Récupère ou crée un profil d'inspecteur avec cache"""
        current_time = datetime.utcnow().timestamp()
        
        # Vérifier le cache
        if (inspector_id in self.inspector_cache and 
            current_time - self.last_cache_update < self.cache_ttl):
            return self.inspector_cache[inspector_id]
        
        # Charger depuis la base de données
        profile = self.store.get_inspector_profile(inspector_id)
        
        if not profile:
            # Créer un nouveau profil
            profile = InspectorProfile(
                inspector_id=inspector_id,
                name=f"Inspector {inspector_id}",
                expertise_level="standard",
                specialization=[self.chapter],
                total_reviews=0,
                accuracy_rate=0.0,
                avg_confidence=0.5,
                avg_review_time=300.0,
                created_at=datetime.utcnow().isoformat(),
                last_active=datetime.utcnow().isoformat(),
                performance_trend="new"
            )
            self.store.save_or_update_inspector_profile(profile)
        
        # Mettre en cache
        self.inspector_cache[inspector_id] = profile
        self.last_cache_update = current_time
        
        return profile

    def _calculate_context_complexity(self, context: Dict[str, Any]) -> int:
        """Calcule la complexité du contexte de décision avec les nouvelles features"""
        complexity = 1
        
        # Facteurs de complexité basés sur les nouvelles métriques
        if context.get("VALEUR_CAF", 0) > 100000000:  # > 100M FCFA
            complexity += 2
        
        if context.get("NUMERO_ARTICLE", 1) > 20:  # Articles multiples
            complexity += 1
        
        if context.get("PAYS_ORIGINE_STR") != context.get("PAYS_PROVENANCE_STR"):
            complexity += 1
        
        # Complexité basée sur les scores de fraude
        composite_score = context.get("COMPOSITE_FRAUD_SCORE", 0)
        if composite_score > 0.3:
            complexity += 2
        elif composite_score > 0.15:
            complexity += 1
        
        # Complexité basée sur les features business
        if self.chapter == "chap30":
            if context.get("BUSINESS_GLISSEMENT_COSMETIQUE", 0) > 0.5:
                complexity += 1
        elif self.chapter == "chap84":
            if context.get("BUSINESS_GLISSEMENT_MACHINE", 0) > 0.5:
                complexity += 1
        elif self.chapter == "chap85":
            if context.get("BUSINESS_GLISSEMENT_ELECTRONIQUE", 0) > 0.5:
                complexity += 1
        
        return min(complexity, 5)  # Max 5

    def _calculate_seasonal_factor(self, context: Dict[str, Any]) -> float:
        """Calcule un facteur saisonnier basé sur la date et le type de produit"""
        try:
            # Utiliser le mois actuel si pas de date dans le contexte
            mois = context.get("MOIS", datetime.utcnow().month)
            
            # Facteurs saisonniers par chapitre
            seasonal_factors = {
                "chap30": {  # Pharma: pics en hiver et début d'année
                    1: 1.2, 2: 1.1, 3: 1.0, 4: 0.9, 5: 0.9, 6: 0.9,
                    7: 0.8, 8: 0.8, 9: 0.9, 10: 1.0, 11: 1.1, 12: 1.2
                },
                "chap84": {  # Machines: pics en début et fin d'année
                    1: 1.1, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
                    7: 0.9, 8: 0.9, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.2
                },
                "chap85": {  # Électronique: pics avant fêtes et rentrées
                    1: 1.1, 2: 0.9, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0,
                    7: 1.0, 8: 1.1, 9: 1.2, 10: 1.1, 11: 1.1, 12: 1.3
                }
            }
            
            chapter_factors = seasonal_factors.get(self.chapter, {})
            return chapter_factors.get(mois, 1.0)
            
        except Exception:
            return 1.0

    def _calculate_bureau_risk_score(self, context: Dict[str, Any]) -> float:
        """Calcule un score de risque pour le bureau avec les nouvelles features"""
        bureau = context.get("BUREAU", "")
        
        # Utiliser la feature business si disponible
        bureau_risque = context.get("BUSINESS_BUREAU_RISQUE", 0)
        if bureau_risque > 0:
            return float(bureau_risque)
        
        # Scores de risque par bureau (à adapter selon les données réelles)
        risk_scores = {
            # Bureaux à haut risque
            "19C": 0.8, "20D": 0.7, "18N": 0.7,
            # Bureaux à risque modéré  
            "11C": 0.6, "12D": 0.5, "DKR": 0.5,
            # Bureaux à faible risque
            "01A": 0.3, "02B": 0.4
        }
        
        return risk_scores.get(bureau, 0.5)  # Défaut: risque moyen

    def _enhanced_context_key(self, context: Dict[str, Any]) -> str:
        """Clé de contexte enrichie avec les nouvelles features"""
        parts = [
            f"B:{str(context.get('BUREAU','UNK')).upper()}",
            f"R:{str(context.get('REGIME_FISCAL','UNK')).upper()}",
            f"O:{str(context.get('PAYS_ORIGINE_STR','UNK')).upper()}",
            f"P:{str(context.get('PAYS_PROVENANCE_STR','UNK')).upper()}",
        ]
        
        # Ajouter des features spécifiques au chapitre avec les nouvelles features business
        if self.chapter == "chap30":
            glissement = context.get("BUSINESS_GLISSEMENT_COSMETIQUE", 0)
            parts.append(f"GLISS:{glissement:.1f}")
        elif self.chapter == "chap84":
            glissement = context.get("BUSINESS_GLISSEMENT_MACHINE", 0)
            parts.append(f"GLISS:{glissement:.1f}")
        elif self.chapter == "chap85":
            glissement = context.get("BUSINESS_GLISSEMENT_ELECTRONIQUE", 0)
            parts.append(f"GLISS:{glissement:.1f}")
        
        # Valeur CAF en tranches
        valeur_caf = context.get("VALEUR_CAF", 0)
        if valeur_caf > 100000000:  # > 100M
            parts.append("VAL:VERY_HIGH")
        elif valeur_caf > 50000000:  # > 50M
            parts.append("VAL:HIGH")
        elif valeur_caf > 10000000:  # > 10M
            parts.append("VAL:MEDIUM")
        else:
            parts.append("VAL:LOW")
        
        # Mois pour la saisonnalité
        mois = context.get("MOIS", datetime.utcnow().month)
        parts.append(f"M:{mois}")
        
        return "|".join(parts)

    def predict(self, context: Dict[str, Any], ml_probability: float = None, threshold: float = None, **kwargs) -> Dict[str, Any]:
        """
        Prédiction RL avancée avec intégration du meilleur modèle ML
        
        Args:
            context: Contexte de la déclaration
            ml_probability: Probabilité du meilleur modèle ML (optionnel)
            threshold: Seuil de décision (optionnel)
            **kwargs: Arguments additionnels (ml_threshold, decision_threshold, etc.)
        """
        # Charger les seuils centralisés pour ce chapitre
        thresholds = load_decision_thresholds(self.chapter)
        
        # Utiliser le seuil optimal calculé basé sur les performances réelles
        optimal_threshold = thresholds.get("optimal_threshold", 0.5)
        
        # Backcompat: accepter ml_threshold passé par l'API d'inférence (mais privilégier le seuil optimal)
        if threshold is None:
            threshold = kwargs.pop("ml_threshold", None)
        if threshold is None:
            threshold = kwargs.pop("decision_threshold", None)
        if threshold is None:
            threshold = optimal_threshold  # Utiliser le seuil optimal par défaut
        ts = datetime.utcnow().isoformat()
        decl_id = str(context.get("DECLARATION_ID", "UNKNOWN"))
        
        # Enrichir le contexte
        ctx_key = self._enhanced_context_key(context)
        ctx_json = json.dumps(context, ensure_ascii=False, default=str)
        
        # Calculer les facteurs contextuels
        complexity = self._calculate_context_complexity(context)
        seasonal_factor = self._calculate_seasonal_factor(context)
        bureau_risk_score = self._calculate_bureau_risk_score(context)
        
        # INTÉGRATION ML-RL : Utiliser le modèle ML
        if ml_probability is not None:
            # Utiliser la probabilité ML fournie (déjà parfaitement calibrée)
            base_prob = float(ml_probability)
            ml_used = True
            logger.debug(f"Probabilité ML utilisée pour {self.chapter}: {base_prob:.3f}")
        elif self.ml_model is not None:
            # Utiliser le modèle ML avancé chargé pour prédire
            try:
                # Préparer les données pour le modèle ML avancé avec toutes les features
                df = pd.DataFrame([context])
                
                # S'assurer que toutes les features nécessaires sont présentes
                required_features = self._get_required_features()
                for feature in required_features:
                    if feature not in df.columns:
                        df[feature] = 0.0  # Valeur par défaut
                
                # Utiliser le pipeline ML complet (preprocessing + modèle)
                prediction = self.ml_model.predict_proba(df)[0][1]  # Probabilité de fraude
                base_prob = float(prediction)
                ml_used = True
                logger.debug(f"Prédiction ML avancée pour {self.chapter}: {base_prob:.3f}")
            except Exception as e:
                logger.warning(f"Erreur prédiction ML avancée pour {self.chapter}: {e}")
                base_prob = self._calculate_base_probability(context)
                ml_used = False
        else:
            # Fallback sur le calcul RL traditionnel
            base_prob = self._calculate_base_probability(context)
            ml_used = False
        
        # Choix d'action avec stratégie avancée
        action, explored, strategy_info = self.bandit.choose(ctx_key, strategy=self.strategy)
        
        # Ajustement de la probabilité basé sur les facteurs contextuels
        context_adjustment = (
            (complexity - 1) * 0.05 +  # +5% par niveau de complexité
            (seasonal_factor - 1) * 0.1 +  # Ajustement saisonnier
            (bureau_risk_score - 0.5) * 0.2  # Ajustement bureau
        )
        
        # INTÉGRATION ML-RL : Ajustement minimal si ML utilisé
        if ml_used:
            # Ajustement très conservateur car les modèles sont parfaitement calibrés
            # Les probabilités ML sont déjà optimales, on fait juste un micro-ajustement contextuel
            final_prob = max(0.01, min(0.99, base_prob + context_adjustment * 0.1))
        else:
            # Ajustement normal pour le RL pur
            final_prob = max(0.01, min(0.99, base_prob + context_adjustment))
        
        # Score de confiance basé sur l'expérience du contexte
        q_values = [
            self.bandit.stats.get((ctx_key, "flag"), {"n": 0, "q": 0.0}),
            self.bandit.stats.get((ctx_key, "pass"), {"n": 0, "q": 0.0})
        ]
        total_experience = sum(q["n"] for q in q_values)
        confidence_score = min(0.95, 0.3 + (total_experience / 100) * 0.6)
        
        # INTÉGRATION ML-RL : Augmenter significativement la confiance si ML utilisé
        if ml_used:
            # Les modèles sont parfaitement calibrés, confiance élevée
            confidence_score = min(0.95, confidence_score + 0.2)
        
        # Décision finale avec seuil optimal calculé
        # Utiliser le seuil optimal basé sur les performances réelles du modèle
        optimal_threshold = thresholds.get("optimal_threshold", threshold)
        predicted = (action == "flag") or (final_prob >= optimal_threshold)
        
        # Log de la décision avec seuil optimal
        logger.debug(f"Décision RL pour {self.chapter}: prob={final_prob:.3f}, seuil_optimal={optimal_threshold:.3f}, décision={predicted}")
        
        # Enregistrer la décision avec intégration ML
        decision_record = AdvancedDecisionRecord(
            chapter=self.chapter,
            declaration_id=decl_id,
            timestamp=ts,
            context_key=ctx_key,
            context_json=ctx_json,
            action=action,
            model_probability=ml_probability,  # Sauvegarder la probabilité ML
            rl_probability=final_prob,
            exploration_used=explored,
            decision_source="ml_rl_integrated" if ml_used else "advanced_rl",
            confidence_score=confidence_score,
            context_complexity=complexity,
            seasonal_factor=seasonal_factor,
            bureau_risk_score=bureau_risk_score,
            extra_json=json.dumps(strategy_info)
        )
        
        self.store.save_advanced_decision(decision_record)
        
        # Mettre à jour les métriques de session
        self.session_metrics["predictions_count"] += 1
        self.session_metrics["avg_confidence"] = (
            (self.session_metrics["avg_confidence"] * (self.session_metrics["predictions_count"] - 1) + 
             confidence_score) / self.session_metrics["predictions_count"]
        )
        if explored:
            self.session_metrics["exploration_rate"] = (
                sum(1 for i in range(self.session_metrics["predictions_count"]) if i % 10 == 0) / 
                self.session_metrics["predictions_count"]
            )
        
        # Enregistrer métriques de performance avec info ML
        self.store.record_performance_metric("prediction_confidence", confidence_score, {
            "context_complexity": complexity,
            "seasonal_factor": seasonal_factor,
            "bureau_risk_score": bureau_risk_score,
            "exploration_used": explored,
            "ml_integration_used": ml_used,
            "ml_probability": ml_probability
        })
        
        return {
            "predicted_fraud": bool(predicted),
            "fraud_probability": float(final_prob),
            "probability": float(final_prob),  # Alias pour compatibilité
            "decision": "flag" if predicted else "pass",  # Décision claire
            "exploration_used": bool(explored),
            "decision_source": "ml_rl_integrated" if ml_used else "advanced_rl",
            "confidence_score": float(confidence_score),
            "context_complexity": complexity,
            "seasonal_factor": seasonal_factor,
            "bureau_risk_score": bureau_risk_score,
            "strategy_info": strategy_info,
            "ml_probability": ml_probability,  # Retourner la probabilité ML
            "ml_integration_used": ml_used,
            "optimal_threshold_used": float(optimal_threshold),
            "model_performance": {
                "auc": thresholds.get("auc", 0.0),
                "f1": thresholds.get("f1", 0.0),
                "precision": thresholds.get("precision", 0.0),
                "recall": thresholds.get("recall", 0.0)
            }
        }

    def _calculate_base_probability(self, context: Dict[str, Any]) -> float:
        """Calcule une probabilité de base contextuelle avec les nouvelles features de détection de fraude"""
        p = 0.05  # Base
        
        # Utiliser les scores de détection de fraude avancée
        bienayme_score = context.get("BIENAYME_CHEBYCHEV_SCORE", 0)
        mirror_score = context.get("MIRROR_TEI_SCORE", 0)
        spectral_score = context.get("SPECTRAL_CLUSTER_SCORE", 0)
        hierarchical_score = context.get("HIERARCHICAL_CLUSTER_SCORE", 0)
        admin_score = context.get("ADMIN_VALUES_SCORE", 0)
        composite_score = context.get("COMPOSITE_FRAUD_SCORE", 0)
        
        # Pondération des scores de détection de fraude
        if bienayme_score > 10:  # Score élevé
            p += 0.2
        elif bienayme_score > 5:
            p += 0.1
        
        if mirror_score > 1000:  # Score élevé
            p += 0.15
        elif mirror_score > 500:
            p += 0.08
        
        if spectral_score > 0.8:  # Score élevé
            p += 0.15
        elif spectral_score > 0.5:
            p += 0.08
        
        if hierarchical_score > 0.8:  # Score élevé
            p += 0.15
        elif hierarchical_score > 0.5:
            p += 0.08
        
        if admin_score > 100:  # Score élevé
            p += 0.1
        elif admin_score > 50:
            p += 0.05
        
        # Score composite (le plus important)
        if composite_score > 0.3:  # Score élevé
            p += 0.25
        elif composite_score > 0.15:
            p += 0.15
        elif composite_score > 0.05:
            p += 0.08
        
        # Facteurs de valeur avec nouvelles métriques
        valeur_caf = context.get("VALEUR_CAF", 0)
        valeur_unitaire_kg = context.get("VALEUR_UNITAIRE_KG", 0)
        
        if valeur_unitaire_kg and valeur_unitaire_kg > 500000:  # > 500k FCFA/kg
            p += 0.2
        elif valeur_unitaire_kg and valeur_unitaire_kg > 200000:  # > 200k FCFA/kg
            p += 0.1
        
        if valeur_caf > 100000000:  # > 100M FCFA
            p += 0.15
        elif valeur_caf > 50000000:  # > 50M FCFA
            p += 0.08
        
        # Facteurs géographiques
        pays_origine = str(context.get("PAYS_ORIGINE_STR", "")).upper()
        if pays_origine in {"CN", "HK", "TW", "VN", "MY", "TH", "IN"}:
            p += 0.1
        
        # Discordance origine/provenance
        if context.get("PAYS_ORIGINE_STR") != context.get("PAYS_PROVENANCE_STR"):
            p += 0.08
        
        # Facteurs de bureau
        bureau = str(context.get("BUREAU", "")).upper()
        if bureau in {"19C", "20D", "18N"}:
            p += 0.08
        
        # Facteurs de régime
        regime = str(context.get("REGIME_FISCAL", "")).upper()
        if any(code in regime for code in ["11", "12", "13", "21", "22"]):
            p += 0.06
        
        # Features business spécifiques par chapitre
        if self.chapter == "chap30":
            # Facteurs pharmaceutiques
            if context.get("BUSINESS_GLISSEMENT_COSMETIQUE", 0) > 0.5:
                p += 0.15
            if context.get("BUSINESS_GLISSEMENT_PAYS_COSMETIQUES", 0) > 0.5:
                p += 0.1
        elif self.chapter == "chap84":
            # Facteurs machines
            if context.get("BUSINESS_GLISSEMENT_MACHINE", 0) > 0.5:
                p += 0.15
            if context.get("BUSINESS_GLISSEMENT_PAYS_MACHINES", 0) > 0.5:
                p += 0.1
        elif self.chapter == "chap85":
            # Facteurs électroniques
            if context.get("BUSINESS_GLISSEMENT_ELECTRONIQUE", 0) > 0.5:
                p += 0.15
            if context.get("BUSINESS_GLISSEMENT_PAYS_ELECTRONIQUES", 0) > 0.5:
                p += 0.1
        
        return max(0.01, min(0.99, p))

    def add_feedback(self,
                    declaration_id: str,
                    predicted_fraud: bool,
                    predicted_probability: float,
                    inspector_decision: bool,
                    inspector_confidence: float,
                    inspector_id: str,
                    context: Optional[Dict[str, Any]] = None,
                    notes: Optional[str] = None,
                    exploration_used: Optional[bool] = None,
                    review_time_seconds: Optional[int] = None,
                    feedback_category: str = "regular") -> Dict[str, Any]:
        """
        Ajoute un feedback avec mise à jour du profil inspecteur
        """
        ts = datetime.utcnow().isoformat()
        ctx = context or {}
        ctx_key = self._enhanced_context_key(ctx)
        ctx_json = json.dumps(ctx, ensure_ascii=False, default=str)
        
        # Récupérer le profil de l'inspecteur
        inspector_profile = self._get_inspector_profile(inspector_id)
        
        # Calculer la qualité du feedback
        agreement = (bool(predicted_fraud) == bool(inspector_decision))
        confidence_factor = min(inspector_confidence, 1.0)
        
        # Qualité basée sur l'accord et la confiance
        quality_score = (
            0.6 * (1.0 if agreement else 0.2) +  # Accord principal
            0.3 * confidence_factor +  # Confiance de l'inspecteur
            0.1 * min(1.0, inspector_profile.total_reviews / 100)  # Expérience
        )
        
        # Déduire l'action et calculer la récompense
        action = "flag" if predicted_fraud else "pass"
        reward = quality_score if agreement else (1 - quality_score)
        
        # Pondération par expertise
        expertise_bonus = {
            "junior": 0.8,
            "standard": 1.0,
            "senior": 1.2,
            "expert": 1.4
        }.get(inspector_profile.expertise_level, 1.0)
        
        weighted_reward = reward * expertise_bonus
        
        # Mise à jour du bandit
        update_result = self.bandit.update(
            ctx_key, action, weighted_reward,
            inspector_expertise=inspector_profile.expertise_level,
            feedback_quality=quality_score,
            review_time=review_time_seconds
        )
        
        # Mise à jour du profil inspecteur
        new_total = inspector_profile.total_reviews + 1
        new_accuracy = (
            (inspector_profile.accuracy_rate * inspector_profile.total_reviews + (1.0 if agreement else 0.0)) 
            / new_total
        )
        new_avg_confidence = (
            (inspector_profile.avg_confidence * inspector_profile.total_reviews + inspector_confidence) 
            / new_total
        )
        
        if review_time_seconds:
            new_avg_review_time = (
                (inspector_profile.avg_review_time * inspector_profile.total_reviews + review_time_seconds) 
                / new_total
            )
        else:
            new_avg_review_time = inspector_profile.avg_review_time
        
        # Calculer la tendance de performance
        if new_total >= 20:  # Assez de données pour une tendance
            recent_feedbacks = 10  # Comparer les 10 derniers
            old_accuracy = inspector_profile.accuracy_rate
            if new_accuracy > old_accuracy + 0.02:
                performance_trend = "improving"
            elif new_accuracy < old_accuracy - 0.02:
                performance_trend = "declining"
            else:
                performance_trend = "stable"
        else:
            performance_trend = "learning"
        
        # Compter les cas similaires
        similar_cases = self._count_similar_cases(inspector_id, ctx_key)
        
        # Mettre à jour le profil
        updated_profile = InspectorProfile(
            inspector_id=inspector_id,
            name=inspector_profile.name,
            expertise_level=inspector_profile.expertise_level,
            specialization=inspector_profile.specialization,
            total_reviews=new_total,
            accuracy_rate=new_accuracy,
            avg_confidence=new_avg_confidence,
            avg_review_time=new_avg_review_time,
            created_at=inspector_profile.created_at,
            last_active=ts,
            performance_trend=performance_trend
        )
        
        self.store.save_or_update_inspector_profile(updated_profile)
        self.inspector_cache[inspector_id] = updated_profile
        
        # Enregistrer le feedback
        feedback_record = AdvancedFeedbackRecord(
            chapter=self.chapter,
            declaration_id=str(declaration_id),
            timestamp=ts,
            inspector_id=str(inspector_id),
            inspector_decision=bool(inspector_decision),
            inspector_confidence=float(inspector_confidence),
            predicted_fraud=bool(predicted_fraud),
            predicted_probability=float(predicted_probability),
            predicted_action=action,
            notes=notes,
            exploration_used=exploration_used,
            context_key=ctx_key,
            context_json=ctx_json,
            feedback_quality_score=quality_score,
            inspector_expertise_level=inspector_profile.expertise_level,
            review_time_seconds=review_time_seconds,
            feedback_category=feedback_category,
            similar_cases_count=similar_cases
        )
        
        self.store.save_advanced_feedback(feedback_record)
        
        # Sauvegarder l'état du bandit
        self._persist_bandit()
        
        # Mettre à jour les métriques de session
        self.session_metrics["feedbacks_count"] += 1
        
        # Enregistrer des métriques de performance (méthode simplifiée)
        try:
            # Enregistrer la métrique de qualité du feedback
            self.store._record_performance_metric("feedback_quality", quality_score, {
                "inspector_expertise": inspector_profile.expertise_level,
                "agreement": agreement,
                "inspector_confidence": inspector_confidence,
                "review_time": review_time_seconds
            })
            
            # Enregistrer la métrique de précision de l'inspecteur
            self.store._record_performance_metric("inspector_accuracy", new_accuracy, {
                "inspector_id": inspector_id,
                "total_reviews": new_total,
                "performance_trend": performance_trend
            })
        except Exception as e:
            logger.warning(f"Erreur enregistrement métriques: {e}")
        
        return {
            "status": "success",
            "reward": float(weighted_reward),
            "quality_score": float(quality_score),
            "agreement": agreement,
            "context_key": ctx_key,
            "bandit_update": update_result,
            "inspector_profile": {
                "accuracy_rate": new_accuracy,
                "total_reviews": new_total,
                "performance_trend": performance_trend
            },
            "similar_cases": similar_cases
        }

    def _count_similar_cases(self, inspector_id: str, context_key: str) -> int:
        """Compte le nombre de cas similaires vus par l'inspecteur"""
        try:
            con = self.store._connect()
            cur = con.cursor()
            cur.execute("""
            SELECT COUNT(*) FROM advanced_feedbacks 
            WHERE inspector_id = %s AND context_key = %s
            """, (inspector_id, context_key))
            count = cur.fetchone()[0]
            con.close()
            return count
        except Exception:
            return 0

    def get_performance_summary(self) -> Dict[str, Any]:
        """Résumé de performance enrichi"""
        bandit_metrics = self.bandit.get_performance_metrics()
        store_analytics = self.store.get_advanced_analytics()
        
        # Métriques globales
        con = self.store._connect()
        cur = con.cursor()
        
        # Statistiques générales
        cur.execute("""
        SELECT 
            COUNT(*) as total_decisions,
            AVG(rl_proba) as avg_fraud_prob,
            AVG(confidence_score) as avg_confidence,
            SUM(CASE WHEN exploration = true THEN 1 ELSE 0 END) as exploration_count
        FROM advanced_decisions
        """)
        
        general_stats = cur.fetchone()
        
        # Accord modèle-inspecteur
        cur.execute("""
        SELECT 
            COUNT(*) as total_feedbacks,
            AVG(CASE WHEN predicted_fraud = inspector_decision THEN 1.0 ELSE 0.0 END) as agreement_rate,
            AVG(feedback_quality_score) as avg_quality
        FROM advanced_feedbacks
        """)
        
        feedback_stats = cur.fetchone()
        
        con.close()
        
        summary = {
            "chapter": self.chapter,
            "strategy": self.strategy,
            "session_metrics": self.session_metrics,
            "general_statistics": {
                "total_decisions": general_stats[0] if general_stats else 0,
                "avg_fraud_probability": round(general_stats[1], 3) if general_stats and general_stats[1] else 0,
                "avg_confidence": round(general_stats[2], 3) if general_stats and general_stats[2] else 0,
                "exploration_rate": round(general_stats[3] / max(general_stats[0], 1), 3) if general_stats else 0
            },
            "feedback_statistics": {
                "total_feedbacks": feedback_stats[0] if feedback_stats else 0,
                "model_inspector_agreement": round(feedback_stats[1], 3) if feedback_stats and feedback_stats[1] else 0,
                "avg_feedback_quality": round(feedback_stats[2], 3) if feedback_stats and feedback_stats[2] else 0
            },
            "bandit_performance": bandit_metrics,
            "advanced_analytics": store_analytics,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return summary

    def get_feedback_analytics(self) -> Dict[str, Any]:
        """Analytics de feedback détaillés"""
        analytics = self.store.get_advanced_analytics()
        
        # Tendances de performance
        performance_trends = {}
        for metric in ["feedback_quality", "inspector_accuracy", "prediction_confidence"]:
            trends = self.store.get_performance_trends(metric, days=30)
            performance_trends[metric] = trends
        
        return {
            "chapter": self.chapter,
            "analytics": analytics,
            "performance_trends": performance_trends,
            "inspector_cache_size": len(self.inspector_cache),
            "session_metrics": self.session_metrics,
            "generated_at": datetime.utcnow().isoformat()
        }

    def get_recent_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère les décisions récentes"""
        con = self._connect()
        cur = con.cursor()
        
        cur.execute("""
            SELECT decision_id, chapter_id, declaration_id, ts, 
                   inspector_id, rl_proba, confidence_score, action, 
                   exploration, context_key, context_json
            FROM advanced_decisions
            ORDER BY ts DESC
            LIMIT %s
        """, (limit,))
        
        decisions = []
        for row in cur.fetchall():
            decisions.append({
                "decision_id": row[0],
                "chapter_id": row[1],
                "declaration_id": row[2],
                "timestamp": row[3],
                "inspector_id": row[4],
                "rl_probability": float(row[5]),
                "confidence_score": float(row[6]),
                "action": row[7],
                "exploration": row[8],
                "context_key": row[9],
                "context_json": row[10]
            })
        
        con.close()
        return decisions

    def get_recent_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère les feedbacks récents"""
        con = self._connect()
        cur = con.cursor()
        
        cur.execute("""
            SELECT feedback_id, chapter_id, declaration_id, ts, 
                   inspector_id, inspector_decision, inspector_confidence,
                   predicted_fraud, predicted_probability, predicted_action,
                   notes, exploration_used, context_key, context_json,
                   feedback_quality_score, inspector_expertise_level
            FROM advanced_feedbacks
            ORDER BY ts DESC
            LIMIT %s
        """, (limit,))
        
        feedbacks = []
        for row in cur.fetchall():
            feedbacks.append({
                "feedback_id": row[0],
                "chapter_id": row[1],
                "declaration_id": row[2],
                "timestamp": row[3],
                "inspector_id": row[4],
                "inspector_decision": row[5],
                "inspector_confidence": float(row[6]),
                "predicted_fraud": row[7],
                "predicted_probability": float(row[8]),
                "predicted_action": row[9],
                "notes": row[10],
                "exploration_used": row[11],
                "context_key": row[12],
                "context_json": row[13],
                "feedback_quality_score": float(row[14]) if row[14] else None,
                "inspector_expertise_level": row[15]
            })
        
        con.close()
        return feedbacks

    def get_inspector_profiles(self) -> List[Dict[str, Any]]:
        """Récupère les profils d'inspecteurs"""
        con = self._connect()
        cur = con.cursor()
        
        cur.execute("""
            SELECT inspector_id, chapter_id, expertise_level, 
                   total_decisions, total_feedbacks, accuracy_score,
                   avg_confidence, avg_review_time, last_activity,
                   profile_data
            FROM inspector_profiles
            ORDER BY accuracy_score DESC
        """)
        
        profiles = []
        for row in cur.fetchall():
            profiles.append({
                "inspector_id": row[0],
                "chapter_id": row[1],
                "expertise_level": row[2],
                "total_decisions": row[3],
                "total_feedbacks": row[4],
                "accuracy_score": float(row[5]) if row[5] else None,
                "avg_confidence": float(row[6]) if row[6] else None,
                "avg_review_time": row[7],
                "last_activity": row[8],
                "profile_data": row[9]
            })
        
        con.close()
        return profiles

    def get_statistics(self) -> Dict[str, Any]:
        """Récupère les statistiques du store"""
        con = self._connect()
        cur = con.cursor()
        
        # Statistiques générales
        cur.execute("SELECT COUNT(*) FROM advanced_decisions")
        total_decisions = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM advanced_feedbacks")
        total_feedbacks = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM inspector_profiles")
        total_inspectors = cur.fetchone()[0]
        
        # Statistiques par chapitre
        cur.execute("""
            SELECT chapter_id, COUNT(*) as count
            FROM advanced_decisions
            GROUP BY chapter_id
        """)
        decisions_by_chapter = {row[0]: row[1] for row in cur.fetchall()}
        
        con.close()
        
        return {
            "total_decisions": total_decisions,
            "total_feedbacks": total_feedbacks,
            "total_inspectors": total_inspectors,
            "decisions_by_chapter": decisions_by_chapter,
            "database_status": "connected"
        }

        
        # Tendances de performance
        performance_trends = {}
        for metric in ["feedback_quality", "inspector_accuracy", "prediction_confidence"]:
            trends = self.store.get_performance_trends(metric, days=30)
            performance_trends[metric] = trends
        
        return {
            "chapter": self.chapter,
            "analytics": analytics,
            "performance_trends": performance_trends,
            "inspector_cache_size": len(self.inspector_cache),
            "session_metrics": self.session_metrics,
            "generated_at": datetime.utcnow().isoformat()
        }

    def get_recent_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère les décisions récentes"""
        con = self._connect()
        cur = con.cursor()
        
        cur.execute("""
            SELECT decision_id, chapter_id, declaration_id, ts, 
                   inspector_id, rl_proba, confidence_score, action, 
                   exploration, context_key, context_json
            FROM advanced_decisions
            ORDER BY ts DESC
            LIMIT %s
        """, (limit,))
        
        decisions = []
        for row in cur.fetchall():
            decisions.append({
                "decision_id": row[0],
                "chapter_id": row[1],
                "declaration_id": row[2],
                "timestamp": row[3],
                "inspector_id": row[4],
                "rl_probability": float(row[5]),
                "confidence_score": float(row[6]),
                "action": row[7],
                "exploration": row[8],
                "context_key": row[9],
                "context_json": row[10]
            })
        
        con.close()
        return decisions

    def get_recent_feedback(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Récupère les feedbacks récents"""
        con = self._connect()
        cur = con.cursor()
        
        cur.execute("""
            SELECT feedback_id, chapter_id, declaration_id, ts, 
                   inspector_id, inspector_decision, inspector_confidence,
                   predicted_fraud, predicted_probability, predicted_action,
                   notes, exploration_used, context_key, context_json,
                   feedback_quality_score, inspector_expertise_level
            FROM advanced_feedbacks
            ORDER BY ts DESC
            LIMIT %s
        """, (limit,))
        
        feedbacks = []
        for row in cur.fetchall():
            feedbacks.append({
                "feedback_id": row[0],
                "chapter_id": row[1],
                "declaration_id": row[2],
                "timestamp": row[3],
                "inspector_id": row[4],
                "inspector_decision": row[5],
                "inspector_confidence": float(row[6]),
                "predicted_fraud": row[7],
                "predicted_probability": float(row[8]),
                "predicted_action": row[9],
                "notes": row[10],
                "exploration_used": row[11],
                "context_key": row[12],
                "context_json": row[13],
                "feedback_quality_score": float(row[14]) if row[14] else None,
                "inspector_expertise_level": row[15]
            })
        
        con.close()
        return feedbacks

    def get_inspector_profiles(self) -> List[Dict[str, Any]]:
        """Récupère les profils d'inspecteurs"""
        con = self._connect()
        cur = con.cursor()
        
        cur.execute("""
            SELECT inspector_id, chapter_id, expertise_level, 
                   total_decisions, total_feedbacks, accuracy_score,
                   avg_confidence, avg_review_time, last_activity,
                   profile_data
            FROM inspector_profiles
            ORDER BY accuracy_score DESC
        """)
        
        profiles = []
        for row in cur.fetchall():
            profiles.append({
                "inspector_id": row[0],
                "chapter_id": row[1],
                "expertise_level": row[2],
                "total_decisions": row[3],
                "total_feedbacks": row[4],
                "accuracy_score": float(row[5]) if row[5] else None,
                "avg_confidence": float(row[6]) if row[6] else None,
                "avg_review_time": row[7],
                "last_activity": row[8],
                "profile_data": row[9]
            })
        
        con.close()
        return profiles

    def get_statistics(self) -> Dict[str, Any]:
        """Récupère les statistiques du store"""
        con = self._connect()
        cur = con.cursor()
        
        # Statistiques générales
        cur.execute("SELECT COUNT(*) FROM advanced_decisions")
        total_decisions = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM advanced_feedbacks")
        total_feedbacks = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM inspector_profiles")
        total_inspectors = cur.fetchone()[0]
        
        # Statistiques par chapitre
        cur.execute("""
            SELECT chapter_id, COUNT(*) as count
            FROM advanced_decisions
            GROUP BY chapter_id
        """)
        decisions_by_chapter = {row[0]: row[1] for row in cur.fetchall()}
        
        con.close()
        
        return {
            "total_decisions": total_decisions,
            "total_feedbacks": total_feedbacks,
            "total_inspectors": total_inspectors,
            "decisions_by_chapter": decisions_by_chapter,
            "database_status": "connected"
        }