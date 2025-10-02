import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'lib/services/complete_backend_service.dart';

/// 🧪 TEST DES COMMUNICATIONS FRONTEND-BACKEND
/// Teste l'intégration complète entre le frontend Flutter et le backend Python
void main() {
  group('Frontend-Backend Communication Tests', () {
    late CompleteBackendService backendService;

    setUp(() {
      backendService = CompleteBackendService();
    });

    testWidgets('Test de santé du backend', (WidgetTester tester) async {
      // Test avec CompleteBackendService
      final response = await backendService.healthCheck();

      expect(response, isA<Map<String, dynamic>?>());
      expect(response, isNotNull);
      final responseData = response!;
      expect(responseData['success'], isTrue);
      expect(responseData['data'], isA<Map>());

      final data = responseData['data'] as Map<String, dynamic>;
      expect(data['status'], 'healthy');
      expect(data['version'], isA<String>());
      expect(data['system'], 'ML-RL Hybrid OCR');
    });

    testWidgets('Test des chapitres disponibles', (WidgetTester tester) async {
      // Test avec CompleteBackendService
      final response = await backendService.getAvailableChapters();

      expect(response, isA<List>());
      expect(response.length, greaterThan(0));

      // Vérifier la structure des chapitres
      for (final chapter in response) {
        expect(chapter, isA<Map<String, dynamic>>());
        expect(chapter['chapter_id'], isA<String>());
        expect(chapter['chapter_name'], isA<String>());
      }
    });

    testWidgets('Test des dépendances système', (WidgetTester tester) async {
      // Test avec CompleteBackendService
      final response = await backendService.getSystemStatus();

      expect(response, isA<Map<String, dynamic>?>());
      expect(response, isNotNull);
      final responseData = response!;
      expect(responseData['success'], isTrue);
      expect(responseData['data'], isA<Map>());

      final data = responseData['data'] as Map<String, dynamic>;
      expect(data, isA<Map>());
    });

    testWidgets('Test de configuration des chapitres', (
      WidgetTester tester,
    ) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await backendService.getChapterConfig(chapter);

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test des informations des modèles', (
      WidgetTester tester,
    ) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await backendService.getModelInfo(chapter);

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test des features des chapitres', (WidgetTester tester) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await backendService.getChapterFeatures(chapter);

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test du statut des chapitres', (WidgetTester tester) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await backendService.getChapterStatus(chapter);

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test des performances des modèles', (
      WidgetTester tester,
    ) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await backendService.getModelInfo(chapter);

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test de prédiction automatique', (WidgetTester tester) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      final testData = {
        'declaration_id': 'TEST_${DateTime.now().millisecondsSinceEpoch}',
        'valeur_caf': 10000.0,
        'poids_net_kg': 500.0,
        'nombre_colis': 10,
        'code_sh_complet': '30049000',
        'pays_origine': 'FR',
        'bureau_douane': '01A',
      };

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await backendService.predictFromDeclarations(chapter, [
          testData,
        ]);

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test de validation des déclarations', (
      WidgetTester tester,
    ) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      final validationData = {
        'declaration_id': 'TEST_${DateTime.now().millisecondsSinceEpoch}',
        'valeur_caf': 10000.0,
        'poids_net_kg': 500.0,
        'nombre_colis': 10,
        'code_sh_complet': '30049000',
        'pays_origine': 'FR',
        'bureau_douane': '01A',
      };

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await backendService.validateDeclaration(
          chapter,
          validationData,
        );

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test de feedback général', (WidgetTester tester) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      final feedbackData = {
        'declaration_id': 'TEST_${DateTime.now().millisecondsSinceEpoch}',
        'predicted_fraud': false,
        'predicted_probability': 0.3,
        'inspector_decision': true,
        'inspector_confidence': 0.8,
        'inspector_id': 'INSP001',
        'additional_notes': 'Test feedback automatique',
        'exploration_used': true,
      };

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await CompleteBackendService.submitRLFeedback(
          chapter,
          feedbackData,
        );

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test de traitement par lot (agrégation)', (
      WidgetTester tester,
    ) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      final batchData = {
        'declarations': [
          {
            'DECLARATION_ID': '2023/01A/12345',
            'VALEUR_CAF': 15000.0,
            'POIDS_NET_KG': 750.0,
            'NOMBRE_COLIS': 15,
            'CODE_SH_COMPLET': '30049000',
            'PAYS_ORIGINE': 'FR',
            'BUREAU_DOUANE': '01A',
          },
          {
            'DECLARATION_ID': '2023/01A/12346',
            'VALEUR_CAF': 8000.0,
            'POIDS_NET_KG': 400.0,
            'NOMBRE_COLIS': 8,
            'CODE_SH_COMPLET': '30049000',
            'PAYS_ORIGINE': 'DE',
            'BUREAU_DOUANE': '01A',
          },
        ],
      };

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await backendService.predictFromDeclarations(
          chapter,
          batchData['declarations'] as List<Map<String, dynamic>>,
        );

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test du pipeline complet', (WidgetTester tester) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      final pipelineData = {
        'declaration_id':
            'PIPELINE_TEST_${DateTime.now().millisecondsSinceEpoch}',
        'valeur_caf': 12000.0,
        'poids_net_kg': 600.0,
        'nombre_colis': 12,
        'code_sh_complet': '30049000',
        'pays_origine': 'FR',
        'bureau_douane': '01A',
      };

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await backendService.testPipeline(
          chapter,
          pipelineData,
        );

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test du CompleteBackendService avec le backend', (
      WidgetTester tester,
    ) async {
      // Test du CompleteBackendService avec les nouvelles fonctionnalités
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: Builder(
              builder: (context) {
                // Test de chargement des chapitres
                backendService.getAvailableChapters().then((chapters) {
                  expect(chapters, isA<List>());
                  expect(chapters.length, greaterThan(0));
                });

                // Test de vérification des dépendances
                backendService.getSystemStatus().then((status) {
                  expect(status, isA<Map>());
                  expect(status, isNotNull);
                  final statusData = status!;
                  expect(statusData['success'], isTrue);
                });

                // Test de vérification de santé
                backendService.healthCheck().then((health) {
                  expect(health, isA<Map>());
                  expect(health, isNotNull);
                  final healthData = health!;
                  expect(healthData['success'], isTrue);
                });

                return Container();
              },
            ),
          ),
        ),
      );

      await tester.pumpAndSettle();
    });

    testWidgets('Test des endpoints avec gestion d\'erreur', (
      WidgetTester tester,
    ) async {
      // Test avec CompleteBackendService pour vérifier la gestion d'erreur
      try {
        // Test avec un chapitre invalide
        final response = await backendService.getModelInfo('invalid_chapter');

        // Si on arrive ici, vérifier que la réponse indique une erreur
        if (response != null) {
          expect(response['success'], isFalse);
          expect(response['error'], isA<String>());
        }
      } catch (e) {
        // C'est normal d'avoir une erreur pour un chapitre invalide
        expect(e, isA<Exception>());
      }
    });

    testWidgets('Test de la persistance des données', (
      WidgetTester tester,
    ) async {
      // Test de la persistance des données avec CompleteBackendService
      // Test avec CompleteBackendService
      final response = await CompleteBackendService.getRecentDeclarations(
        limit: 10,
      );

      expect(response, isA<Map<String, dynamic>?>());
      expect(response, isNotNull);
      final responseData = response!;
      expect(responseData['success'], isTrue);
      expect(responseData['declarations'], isA<List>());
    });

    testWidgets('Test des statistiques d\'agrégation', (
      WidgetTester tester,
    ) async {
      // Test des statistiques d'agrégation avec CompleteBackendService
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await CompleteBackendService.getDeclarationsForChapter(
          chapter,
          limit: 10,
        );

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['declarations'], isA<List>());

        // Vérifier la structure des déclarations
        final declarations = responseData['declarations'] as List;
        for (final declaration in declarations) {
          expect(declaration, isA<Map<String, dynamic>>());
          expect(declaration['declaration_id'], isA<String>());
        }
      }
    });

    testWidgets('Test des performances RL', (WidgetTester tester) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await CompleteBackendService.getRLPerformance(
          chapter,
          'basic',
        );

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test des analytics RL', (WidgetTester tester) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await CompleteBackendService.getRLAnalytics(
          chapter,
          'basic',
        );

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test du retraining ML', (WidgetTester tester) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await CompleteBackendService.getRetrainingStatus(
          chapter,
        );

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test du dashboard chef', (WidgetTester tester) async {
      // Test avec CompleteBackendService
      final response = await CompleteBackendService.getChefDashboard();

      expect(response, isA<Map<String, dynamic>?>());
      expect(response, isNotNull);
      final responseData = response!;
      expect(responseData['success'], isTrue);
      expect(responseData['data'], isA<Map>());
    });

    testWidgets('Test des profils d\'inspecteurs', (WidgetTester tester) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await CompleteBackendService.getInspectorProfiles(
          chapter,
        );

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test du déclenchement de retraining', (
      WidgetTester tester,
    ) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await CompleteBackendService.triggerRetraining(
          chapter,
        );

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test du déclenchement global de retraining', (
      WidgetTester tester,
    ) async {
      // Test avec CompleteBackendService
      final response = await CompleteBackendService.triggerRetrainingAll();

      expect(response, isA<Map<String, dynamic>?>());
      expect(response, isNotNull);
      final responseData = response!;
      expect(responseData['success'], isTrue);
      expect(responseData['data'], isA<Map>());
    });

    testWidgets('Test des détails de déclaration', (WidgetTester tester) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await CompleteBackendService.getDeclarationDetailsById(
          chapter,
          'test_declaration_id',
        );

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test du dashboard ML', (WidgetTester tester) async {
      // Test avec CompleteBackendService
      final response = await CompleteBackendService.getMLDashboard();

      expect(response, isA<Map<String, dynamic>?>());
      expect(response, isNotNull);
      final responseData = response!;
      expect(responseData['success'], isTrue);
      expect(responseData['data'], isA<Map>());
    });

    testWidgets('Test de la liste des PVs', (WidgetTester tester) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await CompleteBackendService.getChapterPvList(chapter);

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });

    testWidgets('Test des détails de PV', (WidgetTester tester) async {
      final chapters = ['chap30', 'chap84', 'chap85'];

      for (final chapter in chapters) {
        // Test avec CompleteBackendService
        final response = await CompleteBackendService.getPVDetail(
          chapter,
          'test_pv_id',
        );

        expect(response, isA<Map<String, dynamic>?>());
        expect(response, isNotNull);
        final responseData = response!;
        expect(responseData['success'], isTrue);
        expect(responseData['data'], isA<Map>());
      }
    });
  });
}
