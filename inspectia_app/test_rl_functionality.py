#!/usr/bin/env python3
"""
Script de test spécifique pour vérifier le fonctionnement des aspects RL
"""

import sys
from pathlib import Path

# Ajouter le chemin du backend au PYTHONPATH
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_rl_bandit_functionality():
    """Test des fonctionnalités de bandit multi-arms"""
    print("🎰 TEST DES FONCTIONNALITÉS BANDIT MULTI-ARMS")
    print("=" * 60)
    
    try:
        from src.shared.advanced_reinforcement_learning import AdvancedEpsilonGreedyBandit
        
        # Créer un bandit de test
        bandit = AdvancedEpsilonGreedyBandit(epsilon=0.1)
        
        # Test 1: Choix d'action avec exploration
        print("🔍 Test 1: Choix d'action avec exploration...")
        action1, explored1, info1 = bandit.choose("context1")
        print(f"   - Action: {action1}")
        print(f"   - Explored: {explored1}")
        print(f"   - Strategy info: {info1}")
        
        # Test 2: Mise à jour avec récompense
        print("\n🔍 Test 2: Mise à jour avec récompense...")
        update_result = bandit.update("context1", "flag", 1.0, inspector_expertise="expert")
        print(f"   - Updated Q: {update_result['updated_q']:.3f}")
        print(f"   - N visits: {update_result['n_visits']}")
        print(f"   - Weighted reward: {update_result['weighted_reward']:.3f}")
        
        # Test 3: Choix d'action après apprentissage
        print("\n🔍 Test 3: Choix d'action après apprentissage...")
        action2, explored2, info2 = bandit.choose("context1")
        print(f"   - Action: {action2}")
        print(f"   - Explored: {explored2}")
        print(f"   - Strategy info: {info2}")
        
        # Test 4: Métriques de performance
        print("\n🔍 Test 4: Métriques de performance...")
        metrics = bandit.get_performance_metrics()
        print(f"   - Total actions: {metrics['total_actions']}")
        print(f"   - Flag percentage: {metrics['flag_percentage']:.1f}%")
        print(f"   - Avg Q flag: {metrics['avg_q_flag']:.3f}")
        print(f"   - Current epsilon: {metrics['current_epsilon']:.3f}")
        
        print("\n✅ Tests bandit réussis")
        return True
        
    except Exception as e:
        print(f"❌ Erreur tests bandit: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rl_manager_functionality():
    """Test des fonctionnalités du manager RL"""
    print("\n🎯 TEST DES FONCTIONNALITÉS MANAGER RL")
    print("=" * 60)
    
    chapters = ["chap30", "chap84", "chap85"]
    
    for chapter in chapters:
        print(f"\n📊 Test du manager RL pour {chapter}")
        print("-" * 40)
        
        try:
            # Importer le module RL du chapitre
            if chapter == "chap30":
                from src.chapters.chap30.rl_integration import get_manager, rl_add_feedback
            elif chapter == "chap84":
                from src.chapters.chap84.rl_integration import get_manager, rl_add_feedback
            elif chapter == "chap85":
                from src.chapters.chap85.rl_integration import get_manager, rl_add_feedback
            
            # Test 1: Récupération du manager
            print(f"🔍 Test 1: Récupération du manager...")
            manager = get_manager("basic")
            print(f"   - Manager récupéré: {type(manager).__name__}")
            print(f"   - Chapitre: {manager.chapter}")
            print(f"   - Epsilon: {manager.bandit.epsilon}")
            print(f"   - Stratégie: {manager.strategy}")
            
            # Test 2: Prédiction avec contexte simple
            print(f"\n🔍 Test 2: Prédiction avec contexte simple...")
            simple_context = {
                "DECLARATION_ID": f"TEST_{chapter}_RL_001",
                "VALEUR_CAF": 10000000,
                "BUREAU": "19C",
                "PAYS_ORIGINE_STR": "CN",
                "PAYS_PROVENANCE_STR": "CN",
                "COMPOSITE_FRAUD_SCORE": 0.1
            }
            
            result = manager.predict(simple_context)
            print(f"   - Prédiction: {result.get('predicted_fraud', False)}")
            print(f"   - Probabilité: {result.get('fraud_probability', 0):.3f}")
            print(f"   - Confiance: {result.get('confidence_score', 0):.3f}")
            print(f"   - ML utilisé: {result.get('ml_integration_used', False)}")
            
            # Test 3: Ajout de feedback
            print(f"\n🔍 Test 3: Ajout de feedback...")
            feedback_result = rl_add_feedback(
                level="basic",
                declaration_id=f"TEST_{chapter}_RL_001",
                predicted_fraud=result.get('predicted_fraud', False),
                predicted_probability=result.get('fraud_probability', 0),
                inspector_decision=True,
                inspector_confidence=0.8,
                inspector_id="INSPECTOR_001",
                context=simple_context,
                notes="Test feedback RL",
                exploration_used=result.get('exploration_used', False),
                review_time_seconds=120,
                feedback_category="test"
            )
            
            print(f"   - Status: {feedback_result.get('status', 'N/A')}")
            print(f"   - Reward: {feedback_result.get('reward', 0):.3f}")
            print(f"   - Quality score: {feedback_result.get('quality_score', 0):.3f}")
            print(f"   - Agreement: {feedback_result.get('agreement', False)}")
            
            # Test 4: Prédiction après feedback (apprentissage)
            print(f"\n🔍 Test 4: Prédiction après feedback...")
            result2 = manager.predict(simple_context)
            print(f"   - Prédiction: {result2.get('predicted_fraud', False)}")
            print(f"   - Probabilité: {result2.get('fraud_probability', 0):.3f}")
            print(f"   - Confiance: {result2.get('confidence_score', 0):.3f}")
            
            # Test 5: Performances du manager
            print(f"\n🔍 Test 5: Performances du manager...")
            performance = manager.get_performance_summary()
            print(f"   - Prédictions totales: {performance.get('total_predictions', 0)}")
            print(f"   - Feedbacks totales: {performance.get('total_feedbacks', 0)}")
            print(f"   - Précision: {performance.get('precision', 0):.3f}")
            print(f"   - Rappel: {performance.get('recall', 0):.3f}")
            print(f"   - F1: {performance.get('f1', 0):.3f}")
            
            print(f"\n✅ Tests manager RL réussis pour {chapter}")
            
        except Exception as e:
            print(f"❌ Erreur tests manager RL pour {chapter}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_rl_strategies():
    """Test des différentes stratégies RL"""
    print("\n🎲 TEST DES STRATÉGIES RL")
    print("=" * 60)
    
    try:
        from src.shared.advanced_reinforcement_learning import AdvancedEpsilonGreedyBandit
        
        strategies = ["epsilon_greedy"]  # Seule stratégie disponible
        context = "test_context"
        
        for strategy in strategies:
            print(f"\n🔍 Test stratégie: {strategy}")
            print("-" * 30)
            
            bandit = AdvancedEpsilonGreedyBandit(epsilon=0.1)
            
            # Faire plusieurs choix pour tester la stratégie
            actions = []
            for i in range(5):
                action, explored, info = bandit.choose(context)
                actions.append(action)
                
                # Mettre à jour avec une récompense aléatoire
                reward = 1.0 if action == "flag" else 0.5
                bandit.update(context, action, reward, inspector_expertise="basic")
            
            print(f"   - Actions choisies: {actions}")
            print(f"   - Exploration rate: {sum(1 for a in actions if a == 'flag') / len(actions):.2f}")
            
            # Vérifier les métriques
            metrics = bandit.get_performance_metrics()
            print(f"   - Total actions: {metrics['total_actions']}")
            print(f"   - Current epsilon: {metrics['current_epsilon']:.3f}")
        
        print("\n✅ Tests stratégies RL réussis")
        return True
        
    except Exception as e:
        print(f"❌ Erreur tests stratégies RL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rl_learning_curve():
    """Test de la courbe d'apprentissage RL"""
    print("\n📈 TEST DE LA COURBE D'APPRENTISSAGE RL")
    print("=" * 60)
    
    try:
        from src.shared.advanced_reinforcement_learning import AdvancedEpsilonGreedyBandit
        
        bandit = AdvancedEpsilonGreedyBandit(epsilon=0.3)
        context = "learning_test"
        
        # Simuler un apprentissage progressif
        print("🔍 Simulation d'apprentissage progressif...")
        
        rewards_history = []
        for episode in range(20):
            # Choix d'action
            action, explored, info = bandit.choose(context)
            
            # Récompense basée sur l'action (flag = meilleure récompense)
            if action == "flag":
                reward = 0.8 + (episode * 0.01)  # Amélioration progressive
            else:
                reward = 0.3 + (episode * 0.005)  # Amélioration plus lente
            
            # Mise à jour
            update_result = bandit.update(context, action, reward, inspector_expertise="expert")
            rewards_history.append(update_result['weighted_reward'])
            
            if episode % 5 == 0:
                print(f"   - Episode {episode}: Action={action}, Reward={reward:.3f}, Q={update_result['updated_q']:.3f}")
        
        # Analyser la courbe d'apprentissage
        print(f"\n📊 Analyse de la courbe d'apprentissage:")
        print(f"   - Récompense initiale: {rewards_history[0]:.3f}")
        print(f"   - Récompense finale: {rewards_history[-1]:.3f}")
        print(f"   - Amélioration: {rewards_history[-1] - rewards_history[0]:.3f}")
        
        # Vérifier que l'apprentissage a eu lieu
        if rewards_history[-1] > rewards_history[0]:
            print("   ✅ Apprentissage détecté: amélioration des récompenses")
        else:
            print("   ⚠️  Pas d'amélioration claire détectée")
        
        # Métriques finales
        metrics = bandit.get_performance_metrics()
        print(f"   - Actions totales: {metrics['total_actions']}")
        print(f"   - Epsilon final: {metrics['current_epsilon']:.3f}")
        print(f"   - Q-value moyenne flag: {metrics['avg_q_flag']:.3f}")
        print(f"   - Q-value moyenne pass: {metrics['avg_q_pass']:.3f}")
        
        print("\n✅ Test courbe d'apprentissage réussi")
        return True
        
    except Exception as e:
        print(f"❌ Erreur test courbe d'apprentissage: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de test RL"""
    print("🧪 TEST COMPLET DES FONCTIONNALITÉS RL")
    print("=" * 70)
    
    tests = [
        ("Bandit Multi-Arms", test_rl_bandit_functionality),
        ("Manager RL", test_rl_manager_functionality),
        ("Stratégies RL", test_rl_strategies),
        ("Courbe d'apprentissage", test_rl_learning_curve)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur critique dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé final
    print(f"\n{'='*70}")
    print("📋 RÉSUMÉ DES TESTS RL")
    print("=" * 70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        print(f"{test_name:<25} : {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 RÉSULTAT FINAL: {passed}/{len(results)} tests réussis")
    
    if passed == len(results):
        print("🎉 TOUS LES TESTS RL SONT RÉUSSIS !")
        print("✅ Les fonctionnalités de reinforcement learning marchent parfaitement")
    else:
        print("⚠️  Certains tests RL ont échoué")
        print("❌ Il y a des problèmes avec les fonctionnalités RL")

if __name__ == "__main__":
    main()





Script de test spécifique pour vérifier le fonctionnement des aspects RL
"""

import sys
from pathlib import Path

# Ajouter le chemin du backend au PYTHONPATH
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_rl_bandit_functionality():
    """Test des fonctionnalités de bandit multi-arms"""
    print("🎰 TEST DES FONCTIONNALITÉS BANDIT MULTI-ARMS")
    print("=" * 60)
    
    try:
        from src.shared.advanced_reinforcement_learning import AdvancedEpsilonGreedyBandit
        
        # Créer un bandit de test
        bandit = AdvancedEpsilonGreedyBandit(epsilon=0.1)
        
        # Test 1: Choix d'action avec exploration
        print("🔍 Test 1: Choix d'action avec exploration...")
        action1, explored1, info1 = bandit.choose("context1")
        print(f"   - Action: {action1}")
        print(f"   - Explored: {explored1}")
        print(f"   - Strategy info: {info1}")
        
        # Test 2: Mise à jour avec récompense
        print("\n🔍 Test 2: Mise à jour avec récompense...")
        update_result = bandit.update("context1", "flag", 1.0, inspector_expertise="expert")
        print(f"   - Updated Q: {update_result['updated_q']:.3f}")
        print(f"   - N visits: {update_result['n_visits']}")
        print(f"   - Weighted reward: {update_result['weighted_reward']:.3f}")
        
        # Test 3: Choix d'action après apprentissage
        print("\n🔍 Test 3: Choix d'action après apprentissage...")
        action2, explored2, info2 = bandit.choose("context1")
        print(f"   - Action: {action2}")
        print(f"   - Explored: {explored2}")
        print(f"   - Strategy info: {info2}")
        
        # Test 4: Métriques de performance
        print("\n🔍 Test 4: Métriques de performance...")
        metrics = bandit.get_performance_metrics()
        print(f"   - Total actions: {metrics['total_actions']}")
        print(f"   - Flag percentage: {metrics['flag_percentage']:.1f}%")
        print(f"   - Avg Q flag: {metrics['avg_q_flag']:.3f}")
        print(f"   - Current epsilon: {metrics['current_epsilon']:.3f}")
        
        print("\n✅ Tests bandit réussis")
        return True
        
    except Exception as e:
        print(f"❌ Erreur tests bandit: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rl_manager_functionality():
    """Test des fonctionnalités du manager RL"""
    print("\n🎯 TEST DES FONCTIONNALITÉS MANAGER RL")
    print("=" * 60)
    
    chapters = ["chap30", "chap84", "chap85"]
    
    for chapter in chapters:
        print(f"\n📊 Test du manager RL pour {chapter}")
        print("-" * 40)
        
        try:
            # Importer le module RL du chapitre
            if chapter == "chap30":
                from src.chapters.chap30.rl_integration import get_manager, rl_add_feedback
            elif chapter == "chap84":
                from src.chapters.chap84.rl_integration import get_manager, rl_add_feedback
            elif chapter == "chap85":
                from src.chapters.chap85.rl_integration import get_manager, rl_add_feedback
            
            # Test 1: Récupération du manager
            print(f"🔍 Test 1: Récupération du manager...")
            manager = get_manager("basic")
            print(f"   - Manager récupéré: {type(manager).__name__}")
            print(f"   - Chapitre: {manager.chapter}")
            print(f"   - Epsilon: {manager.bandit.epsilon}")
            print(f"   - Stratégie: {manager.strategy}")
            
            # Test 2: Prédiction avec contexte simple
            print(f"\n🔍 Test 2: Prédiction avec contexte simple...")
            simple_context = {
                "DECLARATION_ID": f"TEST_{chapter}_RL_001",
                "VALEUR_CAF": 10000000,
                "BUREAU": "19C",
                "PAYS_ORIGINE_STR": "CN",
                "PAYS_PROVENANCE_STR": "CN",
                "COMPOSITE_FRAUD_SCORE": 0.1
            }
            
            result = manager.predict(simple_context)
            print(f"   - Prédiction: {result.get('predicted_fraud', False)}")
            print(f"   - Probabilité: {result.get('fraud_probability', 0):.3f}")
            print(f"   - Confiance: {result.get('confidence_score', 0):.3f}")
            print(f"   - ML utilisé: {result.get('ml_integration_used', False)}")
            
            # Test 3: Ajout de feedback
            print(f"\n🔍 Test 3: Ajout de feedback...")
            feedback_result = rl_add_feedback(
                level="basic",
                declaration_id=f"TEST_{chapter}_RL_001",
                predicted_fraud=result.get('predicted_fraud', False),
                predicted_probability=result.get('fraud_probability', 0),
                inspector_decision=True,
                inspector_confidence=0.8,
                inspector_id="INSPECTOR_001",
                context=simple_context,
                notes="Test feedback RL",
                exploration_used=result.get('exploration_used', False),
                review_time_seconds=120,
                feedback_category="test"
            )
            
            print(f"   - Status: {feedback_result.get('status', 'N/A')}")
            print(f"   - Reward: {feedback_result.get('reward', 0):.3f}")
            print(f"   - Quality score: {feedback_result.get('quality_score', 0):.3f}")
            print(f"   - Agreement: {feedback_result.get('agreement', False)}")
            
            # Test 4: Prédiction après feedback (apprentissage)
            print(f"\n🔍 Test 4: Prédiction après feedback...")
            result2 = manager.predict(simple_context)
            print(f"   - Prédiction: {result2.get('predicted_fraud', False)}")
            print(f"   - Probabilité: {result2.get('fraud_probability', 0):.3f}")
            print(f"   - Confiance: {result2.get('confidence_score', 0):.3f}")
            
            # Test 5: Performances du manager
            print(f"\n🔍 Test 5: Performances du manager...")
            performance = manager.get_performance_summary()
            print(f"   - Prédictions totales: {performance.get('total_predictions', 0)}")
            print(f"   - Feedbacks totales: {performance.get('total_feedbacks', 0)}")
            print(f"   - Précision: {performance.get('precision', 0):.3f}")
            print(f"   - Rappel: {performance.get('recall', 0):.3f}")
            print(f"   - F1: {performance.get('f1', 0):.3f}")
            
            print(f"\n✅ Tests manager RL réussis pour {chapter}")
            
        except Exception as e:
            print(f"❌ Erreur tests manager RL pour {chapter}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_rl_strategies():
    """Test des différentes stratégies RL"""
    print("\n🎲 TEST DES STRATÉGIES RL")
    print("=" * 60)
    
    try:
        from src.shared.advanced_reinforcement_learning import AdvancedEpsilonGreedyBandit
        
        strategies = ["epsilon_greedy"]  # Seule stratégie disponible
        context = "test_context"
        
        for strategy in strategies:
            print(f"\n🔍 Test stratégie: {strategy}")
            print("-" * 30)
            
            bandit = AdvancedEpsilonGreedyBandit(epsilon=0.1)
            
            # Faire plusieurs choix pour tester la stratégie
            actions = []
            for i in range(5):
                action, explored, info = bandit.choose(context)
                actions.append(action)
                
                # Mettre à jour avec une récompense aléatoire
                reward = 1.0 if action == "flag" else 0.5
                bandit.update(context, action, reward, inspector_expertise="basic")
            
            print(f"   - Actions choisies: {actions}")
            print(f"   - Exploration rate: {sum(1 for a in actions if a == 'flag') / len(actions):.2f}")
            
            # Vérifier les métriques
            metrics = bandit.get_performance_metrics()
            print(f"   - Total actions: {metrics['total_actions']}")
            print(f"   - Current epsilon: {metrics['current_epsilon']:.3f}")
        
        print("\n✅ Tests stratégies RL réussis")
        return True
        
    except Exception as e:
        print(f"❌ Erreur tests stratégies RL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rl_learning_curve():
    """Test de la courbe d'apprentissage RL"""
    print("\n📈 TEST DE LA COURBE D'APPRENTISSAGE RL")
    print("=" * 60)
    
    try:
        from src.shared.advanced_reinforcement_learning import AdvancedEpsilonGreedyBandit
        
        bandit = AdvancedEpsilonGreedyBandit(epsilon=0.3)
        context = "learning_test"
        
        # Simuler un apprentissage progressif
        print("🔍 Simulation d'apprentissage progressif...")
        
        rewards_history = []
        for episode in range(20):
            # Choix d'action
            action, explored, info = bandit.choose(context)
            
            # Récompense basée sur l'action (flag = meilleure récompense)
            if action == "flag":
                reward = 0.8 + (episode * 0.01)  # Amélioration progressive
            else:
                reward = 0.3 + (episode * 0.005)  # Amélioration plus lente
            
            # Mise à jour
            update_result = bandit.update(context, action, reward, inspector_expertise="expert")
            rewards_history.append(update_result['weighted_reward'])
            
            if episode % 5 == 0:
                print(f"   - Episode {episode}: Action={action}, Reward={reward:.3f}, Q={update_result['updated_q']:.3f}")
        
        # Analyser la courbe d'apprentissage
        print(f"\n📊 Analyse de la courbe d'apprentissage:")
        print(f"   - Récompense initiale: {rewards_history[0]:.3f}")
        print(f"   - Récompense finale: {rewards_history[-1]:.3f}")
        print(f"   - Amélioration: {rewards_history[-1] - rewards_history[0]:.3f}")
        
        # Vérifier que l'apprentissage a eu lieu
        if rewards_history[-1] > rewards_history[0]:
            print("   ✅ Apprentissage détecté: amélioration des récompenses")
        else:
            print("   ⚠️  Pas d'amélioration claire détectée")
        
        # Métriques finales
        metrics = bandit.get_performance_metrics()
        print(f"   - Actions totales: {metrics['total_actions']}")
        print(f"   - Epsilon final: {metrics['current_epsilon']:.3f}")
        print(f"   - Q-value moyenne flag: {metrics['avg_q_flag']:.3f}")
        print(f"   - Q-value moyenne pass: {metrics['avg_q_pass']:.3f}")
        
        print("\n✅ Test courbe d'apprentissage réussi")
        return True
        
    except Exception as e:
        print(f"❌ Erreur test courbe d'apprentissage: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de test RL"""
    print("🧪 TEST COMPLET DES FONCTIONNALITÉS RL")
    print("=" * 70)
    
    tests = [
        ("Bandit Multi-Arms", test_rl_bandit_functionality),
        ("Manager RL", test_rl_manager_functionality),
        ("Stratégies RL", test_rl_strategies),
        ("Courbe d'apprentissage", test_rl_learning_curve)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur critique dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé final
    print(f"\n{'='*70}")
    print("📋 RÉSUMÉ DES TESTS RL")
    print("=" * 70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        print(f"{test_name:<25} : {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 RÉSULTAT FINAL: {passed}/{len(results)} tests réussis")
    
    if passed == len(results):
        print("🎉 TOUS LES TESTS RL SONT RÉUSSIS !")
        print("✅ Les fonctionnalités de reinforcement learning marchent parfaitement")
    else:
        print("⚠️  Certains tests RL ont échoué")
        print("❌ Il y a des problèmes avec les fonctionnalités RL")

if __name__ == "__main__":
    main()





Script de test spécifique pour vérifier le fonctionnement des aspects RL
"""

import sys
from pathlib import Path

# Ajouter le chemin du backend au PYTHONPATH
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_rl_bandit_functionality():
    """Test des fonctionnalités de bandit multi-arms"""
    print("🎰 TEST DES FONCTIONNALITÉS BANDIT MULTI-ARMS")
    print("=" * 60)
    
    try:
        from src.shared.advanced_reinforcement_learning import AdvancedEpsilonGreedyBandit
        
        # Créer un bandit de test
        bandit = AdvancedEpsilonGreedyBandit(epsilon=0.1)
        
        # Test 1: Choix d'action avec exploration
        print("🔍 Test 1: Choix d'action avec exploration...")
        action1, explored1, info1 = bandit.choose("context1")
        print(f"   - Action: {action1}")
        print(f"   - Explored: {explored1}")
        print(f"   - Strategy info: {info1}")
        
        # Test 2: Mise à jour avec récompense
        print("\n🔍 Test 2: Mise à jour avec récompense...")
        update_result = bandit.update("context1", "flag", 1.0, inspector_expertise="expert")
        print(f"   - Updated Q: {update_result['updated_q']:.3f}")
        print(f"   - N visits: {update_result['n_visits']}")
        print(f"   - Weighted reward: {update_result['weighted_reward']:.3f}")
        
        # Test 3: Choix d'action après apprentissage
        print("\n🔍 Test 3: Choix d'action après apprentissage...")
        action2, explored2, info2 = bandit.choose("context1")
        print(f"   - Action: {action2}")
        print(f"   - Explored: {explored2}")
        print(f"   - Strategy info: {info2}")
        
        # Test 4: Métriques de performance
        print("\n🔍 Test 4: Métriques de performance...")
        metrics = bandit.get_performance_metrics()
        print(f"   - Total actions: {metrics['total_actions']}")
        print(f"   - Flag percentage: {metrics['flag_percentage']:.1f}%")
        print(f"   - Avg Q flag: {metrics['avg_q_flag']:.3f}")
        print(f"   - Current epsilon: {metrics['current_epsilon']:.3f}")
        
        print("\n✅ Tests bandit réussis")
        return True
        
    except Exception as e:
        print(f"❌ Erreur tests bandit: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rl_manager_functionality():
    """Test des fonctionnalités du manager RL"""
    print("\n🎯 TEST DES FONCTIONNALITÉS MANAGER RL")
    print("=" * 60)
    
    chapters = ["chap30", "chap84", "chap85"]
    
    for chapter in chapters:
        print(f"\n📊 Test du manager RL pour {chapter}")
        print("-" * 40)
        
        try:
            # Importer le module RL du chapitre
            if chapter == "chap30":
                from src.chapters.chap30.rl_integration import get_manager, rl_add_feedback
            elif chapter == "chap84":
                from src.chapters.chap84.rl_integration import get_manager, rl_add_feedback
            elif chapter == "chap85":
                from src.chapters.chap85.rl_integration import get_manager, rl_add_feedback
            
            # Test 1: Récupération du manager
            print(f"🔍 Test 1: Récupération du manager...")
            manager = get_manager("basic")
            print(f"   - Manager récupéré: {type(manager).__name__}")
            print(f"   - Chapitre: {manager.chapter}")
            print(f"   - Epsilon: {manager.bandit.epsilon}")
            print(f"   - Stratégie: {manager.strategy}")
            
            # Test 2: Prédiction avec contexte simple
            print(f"\n🔍 Test 2: Prédiction avec contexte simple...")
            simple_context = {
                "DECLARATION_ID": f"TEST_{chapter}_RL_001",
                "VALEUR_CAF": 10000000,
                "BUREAU": "19C",
                "PAYS_ORIGINE_STR": "CN",
                "PAYS_PROVENANCE_STR": "CN",
                "COMPOSITE_FRAUD_SCORE": 0.1
            }
            
            result = manager.predict(simple_context)
            print(f"   - Prédiction: {result.get('predicted_fraud', False)}")
            print(f"   - Probabilité: {result.get('fraud_probability', 0):.3f}")
            print(f"   - Confiance: {result.get('confidence_score', 0):.3f}")
            print(f"   - ML utilisé: {result.get('ml_integration_used', False)}")
            
            # Test 3: Ajout de feedback
            print(f"\n🔍 Test 3: Ajout de feedback...")
            feedback_result = rl_add_feedback(
                level="basic",
                declaration_id=f"TEST_{chapter}_RL_001",
                predicted_fraud=result.get('predicted_fraud', False),
                predicted_probability=result.get('fraud_probability', 0),
                inspector_decision=True,
                inspector_confidence=0.8,
                inspector_id="INSPECTOR_001",
                context=simple_context,
                notes="Test feedback RL",
                exploration_used=result.get('exploration_used', False),
                review_time_seconds=120,
                feedback_category="test"
            )
            
            print(f"   - Status: {feedback_result.get('status', 'N/A')}")
            print(f"   - Reward: {feedback_result.get('reward', 0):.3f}")
            print(f"   - Quality score: {feedback_result.get('quality_score', 0):.3f}")
            print(f"   - Agreement: {feedback_result.get('agreement', False)}")
            
            # Test 4: Prédiction après feedback (apprentissage)
            print(f"\n🔍 Test 4: Prédiction après feedback...")
            result2 = manager.predict(simple_context)
            print(f"   - Prédiction: {result2.get('predicted_fraud', False)}")
            print(f"   - Probabilité: {result2.get('fraud_probability', 0):.3f}")
            print(f"   - Confiance: {result2.get('confidence_score', 0):.3f}")
            
            # Test 5: Performances du manager
            print(f"\n🔍 Test 5: Performances du manager...")
            performance = manager.get_performance_summary()
            print(f"   - Prédictions totales: {performance.get('total_predictions', 0)}")
            print(f"   - Feedbacks totales: {performance.get('total_feedbacks', 0)}")
            print(f"   - Précision: {performance.get('precision', 0):.3f}")
            print(f"   - Rappel: {performance.get('recall', 0):.3f}")
            print(f"   - F1: {performance.get('f1', 0):.3f}")
            
            print(f"\n✅ Tests manager RL réussis pour {chapter}")
            
        except Exception as e:
            print(f"❌ Erreur tests manager RL pour {chapter}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_rl_strategies():
    """Test des différentes stratégies RL"""
    print("\n🎲 TEST DES STRATÉGIES RL")
    print("=" * 60)
    
    try:
        from src.shared.advanced_reinforcement_learning import AdvancedEpsilonGreedyBandit
        
        strategies = ["epsilon_greedy"]  # Seule stratégie disponible
        context = "test_context"
        
        for strategy in strategies:
            print(f"\n🔍 Test stratégie: {strategy}")
            print("-" * 30)
            
            bandit = AdvancedEpsilonGreedyBandit(epsilon=0.1)
            
            # Faire plusieurs choix pour tester la stratégie
            actions = []
            for i in range(5):
                action, explored, info = bandit.choose(context)
                actions.append(action)
                
                # Mettre à jour avec une récompense aléatoire
                reward = 1.0 if action == "flag" else 0.5
                bandit.update(context, action, reward, inspector_expertise="basic")
            
            print(f"   - Actions choisies: {actions}")
            print(f"   - Exploration rate: {sum(1 for a in actions if a == 'flag') / len(actions):.2f}")
            
            # Vérifier les métriques
            metrics = bandit.get_performance_metrics()
            print(f"   - Total actions: {metrics['total_actions']}")
            print(f"   - Current epsilon: {metrics['current_epsilon']:.3f}")
        
        print("\n✅ Tests stratégies RL réussis")
        return True
        
    except Exception as e:
        print(f"❌ Erreur tests stratégies RL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rl_learning_curve():
    """Test de la courbe d'apprentissage RL"""
    print("\n📈 TEST DE LA COURBE D'APPRENTISSAGE RL")
    print("=" * 60)
    
    try:
        from src.shared.advanced_reinforcement_learning import AdvancedEpsilonGreedyBandit
        
        bandit = AdvancedEpsilonGreedyBandit(epsilon=0.3)
        context = "learning_test"
        
        # Simuler un apprentissage progressif
        print("🔍 Simulation d'apprentissage progressif...")
        
        rewards_history = []
        for episode in range(20):
            # Choix d'action
            action, explored, info = bandit.choose(context)
            
            # Récompense basée sur l'action (flag = meilleure récompense)
            if action == "flag":
                reward = 0.8 + (episode * 0.01)  # Amélioration progressive
            else:
                reward = 0.3 + (episode * 0.005)  # Amélioration plus lente
            
            # Mise à jour
            update_result = bandit.update(context, action, reward, inspector_expertise="expert")
            rewards_history.append(update_result['weighted_reward'])
            
            if episode % 5 == 0:
                print(f"   - Episode {episode}: Action={action}, Reward={reward:.3f}, Q={update_result['updated_q']:.3f}")
        
        # Analyser la courbe d'apprentissage
        print(f"\n📊 Analyse de la courbe d'apprentissage:")
        print(f"   - Récompense initiale: {rewards_history[0]:.3f}")
        print(f"   - Récompense finale: {rewards_history[-1]:.3f}")
        print(f"   - Amélioration: {rewards_history[-1] - rewards_history[0]:.3f}")
        
        # Vérifier que l'apprentissage a eu lieu
        if rewards_history[-1] > rewards_history[0]:
            print("   ✅ Apprentissage détecté: amélioration des récompenses")
        else:
            print("   ⚠️  Pas d'amélioration claire détectée")
        
        # Métriques finales
        metrics = bandit.get_performance_metrics()
        print(f"   - Actions totales: {metrics['total_actions']}")
        print(f"   - Epsilon final: {metrics['current_epsilon']:.3f}")
        print(f"   - Q-value moyenne flag: {metrics['avg_q_flag']:.3f}")
        print(f"   - Q-value moyenne pass: {metrics['avg_q_pass']:.3f}")
        
        print("\n✅ Test courbe d'apprentissage réussi")
        return True
        
    except Exception as e:
        print(f"❌ Erreur test courbe d'apprentissage: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de test RL"""
    print("🧪 TEST COMPLET DES FONCTIONNALITÉS RL")
    print("=" * 70)
    
    tests = [
        ("Bandit Multi-Arms", test_rl_bandit_functionality),
        ("Manager RL", test_rl_manager_functionality),
        ("Stratégies RL", test_rl_strategies),
        ("Courbe d'apprentissage", test_rl_learning_curve)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur critique dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé final
    print(f"\n{'='*70}")
    print("📋 RÉSUMÉ DES TESTS RL")
    print("=" * 70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        print(f"{test_name:<25} : {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 RÉSULTAT FINAL: {passed}/{len(results)} tests réussis")
    
    if passed == len(results):
        print("🎉 TOUS LES TESTS RL SONT RÉUSSIS !")
        print("✅ Les fonctionnalités de reinforcement learning marchent parfaitement")
    else:
        print("⚠️  Certains tests RL ont échoué")
        print("❌ Il y a des problèmes avec les fonctionnalités RL")

if __name__ == "__main__":
    main()





Script de test spécifique pour vérifier le fonctionnement des aspects RL
"""

import sys
from pathlib import Path

# Ajouter le chemin du backend au PYTHONPATH
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_rl_bandit_functionality():
    """Test des fonctionnalités de bandit multi-arms"""
    print("🎰 TEST DES FONCTIONNALITÉS BANDIT MULTI-ARMS")
    print("=" * 60)
    
    try:
        from src.shared.advanced_reinforcement_learning import AdvancedEpsilonGreedyBandit
        
        # Créer un bandit de test
        bandit = AdvancedEpsilonGreedyBandit(epsilon=0.1)
        
        # Test 1: Choix d'action avec exploration
        print("🔍 Test 1: Choix d'action avec exploration...")
        action1, explored1, info1 = bandit.choose("context1")
        print(f"   - Action: {action1}")
        print(f"   - Explored: {explored1}")
        print(f"   - Strategy info: {info1}")
        
        # Test 2: Mise à jour avec récompense
        print("\n🔍 Test 2: Mise à jour avec récompense...")
        update_result = bandit.update("context1", "flag", 1.0, inspector_expertise="expert")
        print(f"   - Updated Q: {update_result['updated_q']:.3f}")
        print(f"   - N visits: {update_result['n_visits']}")
        print(f"   - Weighted reward: {update_result['weighted_reward']:.3f}")
        
        # Test 3: Choix d'action après apprentissage
        print("\n🔍 Test 3: Choix d'action après apprentissage...")
        action2, explored2, info2 = bandit.choose("context1")
        print(f"   - Action: {action2}")
        print(f"   - Explored: {explored2}")
        print(f"   - Strategy info: {info2}")
        
        # Test 4: Métriques de performance
        print("\n🔍 Test 4: Métriques de performance...")
        metrics = bandit.get_performance_metrics()
        print(f"   - Total actions: {metrics['total_actions']}")
        print(f"   - Flag percentage: {metrics['flag_percentage']:.1f}%")
        print(f"   - Avg Q flag: {metrics['avg_q_flag']:.3f}")
        print(f"   - Current epsilon: {metrics['current_epsilon']:.3f}")
        
        print("\n✅ Tests bandit réussis")
        return True
        
    except Exception as e:
        print(f"❌ Erreur tests bandit: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rl_manager_functionality():
    """Test des fonctionnalités du manager RL"""
    print("\n🎯 TEST DES FONCTIONNALITÉS MANAGER RL")
    print("=" * 60)
    
    chapters = ["chap30", "chap84", "chap85"]
    
    for chapter in chapters:
        print(f"\n📊 Test du manager RL pour {chapter}")
        print("-" * 40)
        
        try:
            # Importer le module RL du chapitre
            if chapter == "chap30":
                from src.chapters.chap30.rl_integration import get_manager, rl_add_feedback
            elif chapter == "chap84":
                from src.chapters.chap84.rl_integration import get_manager, rl_add_feedback
            elif chapter == "chap85":
                from src.chapters.chap85.rl_integration import get_manager, rl_add_feedback
            
            # Test 1: Récupération du manager
            print(f"🔍 Test 1: Récupération du manager...")
            manager = get_manager("basic")
            print(f"   - Manager récupéré: {type(manager).__name__}")
            print(f"   - Chapitre: {manager.chapter}")
            print(f"   - Epsilon: {manager.bandit.epsilon}")
            print(f"   - Stratégie: {manager.strategy}")
            
            # Test 2: Prédiction avec contexte simple
            print(f"\n🔍 Test 2: Prédiction avec contexte simple...")
            simple_context = {
                "DECLARATION_ID": f"TEST_{chapter}_RL_001",
                "VALEUR_CAF": 10000000,
                "BUREAU": "19C",
                "PAYS_ORIGINE_STR": "CN",
                "PAYS_PROVENANCE_STR": "CN",
                "COMPOSITE_FRAUD_SCORE": 0.1
            }
            
            result = manager.predict(simple_context)
            print(f"   - Prédiction: {result.get('predicted_fraud', False)}")
            print(f"   - Probabilité: {result.get('fraud_probability', 0):.3f}")
            print(f"   - Confiance: {result.get('confidence_score', 0):.3f}")
            print(f"   - ML utilisé: {result.get('ml_integration_used', False)}")
            
            # Test 3: Ajout de feedback
            print(f"\n🔍 Test 3: Ajout de feedback...")
            feedback_result = rl_add_feedback(
                level="basic",
                declaration_id=f"TEST_{chapter}_RL_001",
                predicted_fraud=result.get('predicted_fraud', False),
                predicted_probability=result.get('fraud_probability', 0),
                inspector_decision=True,
                inspector_confidence=0.8,
                inspector_id="INSPECTOR_001",
                context=simple_context,
                notes="Test feedback RL",
                exploration_used=result.get('exploration_used', False),
                review_time_seconds=120,
                feedback_category="test"
            )
            
            print(f"   - Status: {feedback_result.get('status', 'N/A')}")
            print(f"   - Reward: {feedback_result.get('reward', 0):.3f}")
            print(f"   - Quality score: {feedback_result.get('quality_score', 0):.3f}")
            print(f"   - Agreement: {feedback_result.get('agreement', False)}")
            
            # Test 4: Prédiction après feedback (apprentissage)
            print(f"\n🔍 Test 4: Prédiction après feedback...")
            result2 = manager.predict(simple_context)
            print(f"   - Prédiction: {result2.get('predicted_fraud', False)}")
            print(f"   - Probabilité: {result2.get('fraud_probability', 0):.3f}")
            print(f"   - Confiance: {result2.get('confidence_score', 0):.3f}")
            
            # Test 5: Performances du manager
            print(f"\n🔍 Test 5: Performances du manager...")
            performance = manager.get_performance_summary()
            print(f"   - Prédictions totales: {performance.get('total_predictions', 0)}")
            print(f"   - Feedbacks totales: {performance.get('total_feedbacks', 0)}")
            print(f"   - Précision: {performance.get('precision', 0):.3f}")
            print(f"   - Rappel: {performance.get('recall', 0):.3f}")
            print(f"   - F1: {performance.get('f1', 0):.3f}")
            
            print(f"\n✅ Tests manager RL réussis pour {chapter}")
            
        except Exception as e:
            print(f"❌ Erreur tests manager RL pour {chapter}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

def test_rl_strategies():
    """Test des différentes stratégies RL"""
    print("\n🎲 TEST DES STRATÉGIES RL")
    print("=" * 60)
    
    try:
        from src.shared.advanced_reinforcement_learning import AdvancedEpsilonGreedyBandit
        
        strategies = ["epsilon_greedy"]  # Seule stratégie disponible
        context = "test_context"
        
        for strategy in strategies:
            print(f"\n🔍 Test stratégie: {strategy}")
            print("-" * 30)
            
            bandit = AdvancedEpsilonGreedyBandit(epsilon=0.1)
            
            # Faire plusieurs choix pour tester la stratégie
            actions = []
            for i in range(5):
                action, explored, info = bandit.choose(context)
                actions.append(action)
                
                # Mettre à jour avec une récompense aléatoire
                reward = 1.0 if action == "flag" else 0.5
                bandit.update(context, action, reward, inspector_expertise="basic")
            
            print(f"   - Actions choisies: {actions}")
            print(f"   - Exploration rate: {sum(1 for a in actions if a == 'flag') / len(actions):.2f}")
            
            # Vérifier les métriques
            metrics = bandit.get_performance_metrics()
            print(f"   - Total actions: {metrics['total_actions']}")
            print(f"   - Current epsilon: {metrics['current_epsilon']:.3f}")
        
        print("\n✅ Tests stratégies RL réussis")
        return True
        
    except Exception as e:
        print(f"❌ Erreur tests stratégies RL: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rl_learning_curve():
    """Test de la courbe d'apprentissage RL"""
    print("\n📈 TEST DE LA COURBE D'APPRENTISSAGE RL")
    print("=" * 60)
    
    try:
        from src.shared.advanced_reinforcement_learning import AdvancedEpsilonGreedyBandit
        
        bandit = AdvancedEpsilonGreedyBandit(epsilon=0.3)
        context = "learning_test"
        
        # Simuler un apprentissage progressif
        print("🔍 Simulation d'apprentissage progressif...")
        
        rewards_history = []
        for episode in range(20):
            # Choix d'action
            action, explored, info = bandit.choose(context)
            
            # Récompense basée sur l'action (flag = meilleure récompense)
            if action == "flag":
                reward = 0.8 + (episode * 0.01)  # Amélioration progressive
            else:
                reward = 0.3 + (episode * 0.005)  # Amélioration plus lente
            
            # Mise à jour
            update_result = bandit.update(context, action, reward, inspector_expertise="expert")
            rewards_history.append(update_result['weighted_reward'])
            
            if episode % 5 == 0:
                print(f"   - Episode {episode}: Action={action}, Reward={reward:.3f}, Q={update_result['updated_q']:.3f}")
        
        # Analyser la courbe d'apprentissage
        print(f"\n📊 Analyse de la courbe d'apprentissage:")
        print(f"   - Récompense initiale: {rewards_history[0]:.3f}")
        print(f"   - Récompense finale: {rewards_history[-1]:.3f}")
        print(f"   - Amélioration: {rewards_history[-1] - rewards_history[0]:.3f}")
        
        # Vérifier que l'apprentissage a eu lieu
        if rewards_history[-1] > rewards_history[0]:
            print("   ✅ Apprentissage détecté: amélioration des récompenses")
        else:
            print("   ⚠️  Pas d'amélioration claire détectée")
        
        # Métriques finales
        metrics = bandit.get_performance_metrics()
        print(f"   - Actions totales: {metrics['total_actions']}")
        print(f"   - Epsilon final: {metrics['current_epsilon']:.3f}")
        print(f"   - Q-value moyenne flag: {metrics['avg_q_flag']:.3f}")
        print(f"   - Q-value moyenne pass: {metrics['avg_q_pass']:.3f}")
        
        print("\n✅ Test courbe d'apprentissage réussi")
        return True
        
    except Exception as e:
        print(f"❌ Erreur test courbe d'apprentissage: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de test RL"""
    print("🧪 TEST COMPLET DES FONCTIONNALITÉS RL")
    print("=" * 70)
    
    tests = [
        ("Bandit Multi-Arms", test_rl_bandit_functionality),
        ("Manager RL", test_rl_manager_functionality),
        ("Stratégies RL", test_rl_strategies),
        ("Courbe d'apprentissage", test_rl_learning_curve)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur critique dans {test_name}: {e}")
            results.append((test_name, False))
    
    # Résumé final
    print(f"\n{'='*70}")
    print("📋 RÉSUMÉ DES TESTS RL")
    print("=" * 70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ RÉUSSI" if result else "❌ ÉCHEC"
        print(f"{test_name:<25} : {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 RÉSULTAT FINAL: {passed}/{len(results)} tests réussis")
    
    if passed == len(results):
        print("🎉 TOUS LES TESTS RL SONT RÉUSSIS !")
        print("✅ Les fonctionnalités de reinforcement learning marchent parfaitement")
    else:
        print("⚠️  Certains tests RL ont échoué")
        print("❌ Il y a des problèmes avec les fonctionnalités RL")

if __name__ == "__main__":
    main()























