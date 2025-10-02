#!/usr/bin/env python3
"""
Test complet pour vérifier l'absence d'overfitting dans les 3 chapitres
Évalue la robustesse et la généralisation des modèles
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import logging
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OverfittingTester:
    """Testeur complet pour détecter l'overfitting"""
    
    def __init__(self):
        self.results = {}
        self.base_path = Path("/Users/macbook/Desktop/inspectia_app/backend")
        self.chapters = ['chap30', 'chap84', 'chap85']
        
    def load_model_data(self, chapter):
        """Charger les modèles et données pour un chapitre"""
        logger.info(f"📊 Chargement des données pour {chapter}...")
        
        models_path = self.base_path / "models" / chapter
        results_path = self.base_path / "results" / chapter
        splits_path = self.base_path / "data" / "ml_splits" / chapter
        
        # Charger les modèles
        models = {}
        model_names = ['lightgbm', 'xgboost', 'catboost', 'randomforest', 'logisticregression']
        
        for name in model_names:
            model_file = models_path / f"{name}_model.pkl"
            if model_file.exists():
                try:
                    models[name] = joblib.load(model_file)
                    logger.info(f"   ✅ {name} chargé")
                except Exception as e:
                    logger.warning(f"   ❌ Erreur chargement {name}: {e}")
        
        # Charger les données splittées
        try:
            X_train = pd.read_csv(splits_path / "X_train.csv")
            X_valid = pd.read_csv(splits_path / "X_valid.csv")
            X_test = pd.read_csv(splits_path / "X_test.csv")
            y_train = pd.read_csv(splits_path / "y_train.csv").values.ravel()
            y_valid = pd.read_csv(splits_path / "y_valid.csv").values.ravel()
            y_test = pd.read_csv(splits_path / "y_test.csv").values.ravel()
            
            logger.info(f"   ✅ Données chargées: Train={X_train.shape}, Valid={X_valid.shape}, Test={X_test.shape}")
            
        except Exception as e:
            logger.error(f"   ❌ Erreur chargement données: {e}")
            return None, None, None, None, None, None
        
        return models, X_train, X_valid, X_test, y_train, y_valid, y_test
    
    def evaluate_model_robustness(self, model, X_train, X_valid, X_test, y_train, y_valid, y_test, model_name):
        """Évaluer la robustesse d'un modèle"""
        logger.info(f"   🔍 Test robustesse {model_name}...")
        
        results = {}
        
        try:
            # Prédictions sur tous les sets
            y_pred_train = model.predict(X_train)
            y_pred_valid = model.predict(X_valid)
            y_pred_test = model.predict(X_test)
            
            y_proba_train = model.predict_proba(X_train)[:, 1]
            y_proba_valid = model.predict_proba(X_valid)[:, 1]
            y_proba_test = model.predict_proba(X_test)[:, 1]
            
            # Métriques F1
            f1_train = f1_score(y_train, y_pred_train)
            f1_valid = f1_score(y_valid, y_pred_valid)
            f1_test = f1_score(y_test, y_pred_test)
            
            # AUC
            auc_train = roc_auc_score(y_train, y_proba_train)
            auc_valid = roc_auc_score(y_valid, y_proba_valid)
            auc_test = roc_auc_score(y_test, y_proba_test)
            
            # Écarts (indicateurs d'overfitting)
            f1_gap_train_valid = abs(f1_train - f1_valid)
            f1_gap_train_test = abs(f1_train - f1_test)
            f1_gap_valid_test = abs(f1_valid - f1_test)
            
            auc_gap_train_valid = abs(auc_train - auc_valid)
            auc_gap_train_test = abs(auc_train - auc_test)
            auc_gap_valid_test = abs(auc_valid - auc_test)
            
            results = {
                'f1_scores': {
                    'train': f1_train,
                    'valid': f1_valid,
                    'test': f1_test,
                    'gap_train_valid': f1_gap_train_valid,
                    'gap_train_test': f1_gap_train_test,
                    'gap_valid_test': f1_gap_valid_test
                },
                'auc_scores': {
                    'train': auc_train,
                    'valid': auc_valid,
                    'test': auc_test,
                    'gap_train_valid': auc_gap_train_valid,
                    'gap_train_test': auc_gap_train_test,
                    'gap_valid_test': auc_gap_valid_test
                },
                'overfitting_risk': {
                    'f1_gap_high': f1_gap_train_valid > 0.05 or f1_gap_train_test > 0.05,
                    'auc_gap_high': auc_gap_train_valid > 0.05 or auc_gap_train_test > 0.05,
                    'test_performance_low': f1_test < 0.7 or auc_test < 0.8
                }
            }
            
            logger.info(f"      F1: Train={f1_train:.4f}, Valid={f1_valid:.4f}, Test={f1_test:.4f}")
            logger.info(f"      AUC: Train={auc_train:.4f}, Valid={auc_valid:.4f}, Test={auc_test:.4f}")
            logger.info(f"      Écarts F1: Train-Valid={f1_gap_train_valid:.4f}, Train-Test={f1_gap_train_test:.4f}")
            
        except Exception as e:
            logger.error(f"      ❌ Erreur évaluation {model_name}: {e}")
            results = {'error': str(e)}
        
        return results
    
    def test_cross_validation_stability(self, model, X_train, y_train, model_name):
        """Tester la stabilité en validation croisée"""
        logger.info(f"   🔄 Test stabilité CV {model_name}...")
        
        try:
            # Validation croisée F1
            cv_f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
            cv_f1_mean = cv_f1.mean()
            cv_f1_std = cv_f1.std()
            
            # Validation croisée AUC
            cv_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            cv_auc_mean = cv_auc.mean()
            cv_auc_std = cv_auc.std()
            
            stability_risk = cv_f1_std > 0.05 or cv_auc_std > 0.05
            
            logger.info(f"      CV F1: {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")
            logger.info(f"      CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
            
            return {
                'cv_f1_mean': cv_f1_mean,
                'cv_f1_std': cv_f1_std,
                'cv_auc_mean': cv_auc_mean,
                'cv_auc_std': cv_auc_std,
                'stability_risk': stability_risk
            }
            
        except Exception as e:
            logger.error(f"      ❌ Erreur CV {model_name}: {e}")
            return {'error': str(e)}
    
    def test_feature_importance_consistency(self, models, X_train, y_train):
        """Tester la cohérence de l'importance des features"""
        logger.info("   🎯 Test cohérence importance features...")
        
        importance_scores = {}
        
        for name, model in models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importance_scores[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Pour LogisticRegression, utiliser les coefficients absolus
                    importance_scores[name] = np.abs(model.coef_[0])
                else:
                    logger.warning(f"      ⚠️ Pas d'importance features pour {name}")
                    continue
                    
            except Exception as e:
                logger.error(f"      ❌ Erreur importance {name}: {e}")
        
        # Calculer la corrélation entre les importances
        correlations = {}
        model_names = list(importance_scores.keys())
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                try:
                    corr = np.corrcoef(importance_scores[name1], importance_scores[name2])[0, 1]
                    correlations[f"{name1}_{name2}"] = corr
                    logger.info(f"      Corrélation {name1}-{name2}: {corr:.4f}")
                except:
                    continue
        
        return {
            'importance_scores': importance_scores,
            'correlations': correlations,
            'consistency_risk': any(abs(corr) < 0.3 for corr in correlations.values())
        }
    
    def generate_overfitting_report(self, chapter, chapter_results):
        """Générer un rapport d'overfitting pour un chapitre"""
        logger.info(f"📋 Génération rapport overfitting pour {chapter}...")
        
        report = {
            'chapter': chapter,
            'timestamp': pd.Timestamp.now().isoformat(),
            'models_analyzed': len(chapter_results['models']),
            'overfitting_analysis': {},
            'recommendations': []
        }
        
        # Analyser chaque modèle
        for model_name, model_results in chapter_results['models'].items():
            if 'error' in model_results:
                continue
                
            overfitting_signs = []
            
            # Signes d'overfitting
            if model_results['robustness']['overfitting_risk']['f1_gap_high']:
                overfitting_signs.append("Écart F1 Train-Test élevé")
            
            if model_results['robustness']['overfitting_risk']['auc_gap_high']:
                overfitting_signs.append("Écart AUC Train-Test élevé")
            
            if model_results['robustness']['overfitting_risk']['test_performance_low']:
                overfitting_signs.append("Performance test faible")
            
            if model_results.get('cv_stability', {}).get('stability_risk', False):
                overfitting_signs.append("Instabilité validation croisée")
            
            report['overfitting_analysis'][model_name] = {
                'overfitting_signs': overfitting_signs,
                'risk_level': 'HIGH' if len(overfitting_signs) >= 2 else 'MEDIUM' if len(overfitting_signs) == 1 else 'LOW',
                'f1_gap': model_results['robustness']['f1_scores']['gap_train_test'],
                'auc_gap': model_results['robustness']['auc_scores']['gap_train_test'],
                'cv_std': model_results.get('cv_stability', {}).get('cv_f1_std', 0)
            }
        
        # Recommandations
        high_risk_models = [name for name, data in report['overfitting_analysis'].items() 
                           if data['risk_level'] == 'HIGH']
        
        if high_risk_models:
            report['recommendations'].append(f"Modèles à risque élevé: {', '.join(high_risk_models)}")
            report['recommendations'].append("Augmenter la régularisation")
            report['recommendations'].append("Réduire la complexité du modèle")
        
        return report
    
    def create_overfitting_visualization(self, chapter, chapter_results):
        """Créer des visualisations pour détecter l'overfitting"""
        logger.info(f"📊 Génération visualisations overfitting pour {chapter}...")
        
        # Créer le dossier de résultats
        results_dir = self.base_path / "test_results" / "overfitting_analysis"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Graphique des écarts Train-Test
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Analyse Overfitting - {chapter}', fontsize=16, fontweight='bold')
        
        # Données pour les graphiques
        models = []
        f1_gaps = []
        auc_gaps = []
        cv_stds = []
        
        for model_name, model_results in chapter_results['models'].items():
            if 'error' not in model_results:
                models.append(model_name)
                f1_gaps.append(model_results['robustness']['f1_scores']['gap_train_test'])
                auc_gaps.append(model_results['robustness']['auc_scores']['gap_train_test'])
                cv_stds.append(model_results.get('cv_stability', {}).get('cv_f1_std', 0))
        
        # Graphique 1: Écarts F1
        ax1 = axes[0, 0]
        bars1 = ax1.bar(models, f1_gaps, color=['red' if gap > 0.05 else 'orange' if gap > 0.02 else 'green' for gap in f1_gaps])
        ax1.set_title('Écart F1 Train-Test')
        ax1.set_ylabel('Écart F1')
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Seuil Risque')
        ax1.axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='Seuil Attention')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Graphique 2: Écarts AUC
        ax2 = axes[0, 1]
        bars2 = ax2.bar(models, auc_gaps, color=['red' if gap > 0.05 else 'orange' if gap > 0.02 else 'green' for gap in auc_gaps])
        ax2.set_title('Écart AUC Train-Test')
        ax2.set_ylabel('Écart AUC')
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Seuil Risque')
        ax2.axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='Seuil Attention')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        # Graphique 3: Stabilité CV
        ax3 = axes[1, 0]
        bars3 = ax3.bar(models, cv_stds, color=['red' if std > 0.05 else 'orange' if std > 0.02 else 'green' for std in cv_stds])
        ax3.set_title('Stabilité Validation Croisée (Std)')
        ax3.set_ylabel('Écart-type F1')
        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Seuil Risque')
        ax3.axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='Seuil Attention')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        # Graphique 4: Scores de performance
        ax4 = axes[1, 1]
        f1_train = [chapter_results['models'][m]['robustness']['f1_scores']['train'] for m in models]
        f1_test = [chapter_results['models'][m]['robustness']['f1_scores']['test'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax4.bar(x - width/2, f1_train, width, label='Train', alpha=0.8, color='skyblue')
        ax4.bar(x + width/2, f1_test, width, label='Test', alpha=0.8, color='lightcoral')
        
        ax4.set_title('Scores F1 Train vs Test')
        ax4.set_ylabel('F1 Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(results_dir / f'overfitting_analysis_{chapter}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"   ✅ Visualisation sauvegardée: overfitting_analysis_{chapter}.png")
    
    def run_comprehensive_test(self):
        """Lancer le test complet pour tous les chapitres"""
        logger.info("🚀 DÉBUT DU TEST COMPLET DE DÉTECTION D'OVERFITTING")
        logger.info("=" * 60)
        
        all_results = {}
        
        for chapter in self.chapters:
            logger.info(f"\n📊 ANALYSE DU CHAPITRE {chapter}")
            logger.info("-" * 40)
            
            # Charger les données
            models, X_train, X_valid, X_test, y_train, y_valid, y_test = self.load_model_data(chapter)
            
            if models is None:
                logger.error(f"❌ Impossible de charger les données pour {chapter}")
                continue
            
            chapter_results = {'models': {}}
            
            # Tester chaque modèle
            for model_name, model in models.items():
                logger.info(f"\n🔍 Analyse du modèle {model_name}...")
                
                model_results = {}
                
                # Test de robustesse
                model_results['robustness'] = self.evaluate_model_robustness(
                    model, X_train, X_valid, X_test, y_train, y_valid, y_test, model_name
                )
                
                # Test de stabilité CV
                model_results['cv_stability'] = self.test_cross_validation_stability(
                    model, X_train, y_train, model_name
                )
                
                chapter_results['models'][model_name] = model_results
            
            # Test de cohérence des features
            chapter_results['feature_consistency'] = self.test_feature_importance_consistency(
                models, X_train, y_train
            )
            
            # Générer le rapport
            report = self.generate_overfitting_report(chapter, chapter_results)
            
            # Créer les visualisations
            self.create_overfitting_visualization(chapter, chapter_results)
            
            all_results[chapter] = {
                'results': chapter_results,
                'report': report
            }
        
        # Sauvegarder le rapport global
        self.save_global_report(all_results)
        
        logger.info("\n🎉 TEST COMPLET TERMINÉ")
        logger.info("=" * 60)
        
        return all_results
    
    def save_global_report(self, all_results):
        """Sauvegarder le rapport global"""
        logger.info("💾 Sauvegarde du rapport global...")
        
        results_dir = self.base_path / "test_results" / "overfitting_analysis"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Rapport JSON détaillé
        with open(results_dir / "comprehensive_overfitting_report.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Rapport texte résumé
        with open(results_dir / "overfitting_summary.txt", 'w') as f:
            f.write("RAPPORT GLOBAL DE DÉTECTION D'OVERFITTING\n")
            f.write("=" * 50 + "\n\n")
            
            for chapter, data in all_results.items():
                f.write(f"CHAPITRE {chapter.upper()}\n")
                f.write("-" * 20 + "\n")
                
                report = data['report']
                f.write(f"Modèles analysés: {report['models_analyzed']}\n\n")
                
                for model_name, analysis in report['overfitting_analysis'].items():
                    f.write(f"  {model_name.upper()}:\n")
                    f.write(f"    Niveau de risque: {analysis['risk_level']}\n")
                    f.write(f"    Écart F1 Train-Test: {analysis['f1_gap']:.4f}\n")
                    f.write(f"    Écart AUC Train-Test: {analysis['auc_gap']:.4f}\n")
                    f.write(f"    Std CV: {analysis['cv_std']:.4f}\n")
                    
                    if analysis['overfitting_signs']:
                        f.write(f"    Signes d'overfitting: {', '.join(analysis['overfitting_signs'])}\n")
                    else:
                        f.write(f"    ✅ Aucun signe d'overfitting détecté\n")
                    
                    f.write("\n")
                
                if report['recommendations']:
                    f.write("  Recommandations:\n")
                    for rec in report['recommendations']:
                        f.write(f"    - {rec}\n")
                
                f.write("\n" + "=" * 50 + "\n\n")
        
        logger.info("   ✅ Rapport global sauvegardé")

def main():
    """Fonction principale"""
    tester = OverfittingTester()
    results = tester.run_comprehensive_test()
    
    # Affichage du résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DU TEST COMPLET")
    print("=" * 60)
    
    for chapter, data in results.items():
        print(f"\n{chapter.upper()}:")
        report = data['report']
        
        for model_name, analysis in report['overfitting_analysis'].items():
            risk_emoji = "🔴" if analysis['risk_level'] == 'HIGH' else "🟡" if analysis['risk_level'] == 'MEDIUM' else "🟢"
            print(f"  {risk_emoji} {model_name}: {analysis['risk_level']} (F1 gap: {analysis['f1_gap']:.4f})")

if __name__ == "__main__":
    main()

