import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  group('Frontend-Backend Communication Tests', () {
    const String baseUrl = 'http://localhost:8000';

    setUpAll(() async {
      // Vérifier que le backend est accessible
      try {
        final response = await http.get(Uri.parse('$baseUrl/health'));
        if (response.statusCode != 200) {
          throw Exception('Backend non accessible');
        }
      } catch (e) {
        throw Exception('Impossible de se connecter au backend: $e');
      }
    });

    group('Tests de connectivité', () {
      test('Backend accessible', () async {
        final response = await http.get(Uri.parse('$baseUrl/health'));
        expect(response.statusCode, 200);

        final data = json.decode(response.body);
        expect(data['status'], 'healthy');
        expect(data['service'], 'InspectIA API');
      });

      test('Endpoints de base accessibles', () async {
        final endpoints = [
          '/predict/yaml-coverage',
          '/predict/chap30/config',
          '/predict/chap84/config',
          '/predict/chap85/config',
        ];

        for (final endpoint in endpoints) {
          final response = await http.get(Uri.parse('$baseUrl$endpoint'));
          expect(response.statusCode, 200, reason: 'Endpoint $endpoint échoué');

          final data = json.decode(response.body);
          expect(data, isNotNull, reason: 'Endpoint $endpoint retourne null');
        }
      });
    });

    group('Tests des configurations par chapitre', () {
      test('Configuration chap30', () async {
        final response = await http.get(
          Uri.parse('$baseUrl/predict/chap30/config'),
        );
        expect(response.statusCode, 200);

        final data = json.decode(response.body);
        expect(data['chapter'], 'chap30');
        expect(data['config'], isNotNull);
        expect(data['config']['chapter'], 30);
      });

      test('Configuration chap84', () async {
        final response = await http.get(
          Uri.parse('$baseUrl/predict/chap84/config'),
        );
        expect(response.statusCode, 200);

        final data = json.decode(response.body);
        expect(data['chapter'], 'chap84');
        expect(data['config'], isNotNull);
        expect(data['config']['chapter'], 84);
      });

      test('Configuration chap85', () async {
        final response = await http.get(
          Uri.parse('$baseUrl/predict/chap85/config'),
        );
        expect(response.statusCode, 200);

        final data = json.decode(response.body);
        expect(data['chapter'], 'chap85');
        expect(data['config'], isNotNull);
        expect(data['config']['features_count'], 43);
      });
    });

    group('Tests du système hybride', () {
      test('Test hybride chap30', () async {
        final testData = {
          "test_data": [
            {
              "declaration_id": "TEST_SIMPLE_30",
              "libelle_produit": "Test produit simple",
              "valeur_caf": 500000,
              "poids_net": 50,
              "pays_origine": "FR",
              "code_bureau": "75001",
            },
          ],
        };

        final response = await http.post(
          Uri.parse('$baseUrl/predict/chap30/test-hybrid'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode(testData),
        );

        expect(response.statusCode, 200);

        final data = json.decode(response.body);
        expect(data['chapter'], 'chap30');
        expect(data['test_input'], isNotNull);
        expect(data['hybrid_result'], isNotNull);
        expect(data['hybrid_result']['score_ml'], isNotNull);
        expect(data['hybrid_result']['score_metier'], isNotNull);
        expect(data['hybrid_result']['score_final'], isNotNull);
        expect(data['hybrid_result']['decision'], isNotNull);
      });

      test('Test hybride chap84', () async {
        final testData = {
          "test_data": [
            {
              "declaration_id": "TEST_SIMPLE_84",
              "libelle_produit": "Test produit simple",
              "valeur_caf": 500000,
              "poids_net": 50,
              "pays_origine": "FR",
              "code_bureau": "75001",
            },
          ],
        };

        final response = await http.post(
          Uri.parse('$baseUrl/predict/chap84/test-hybrid'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode(testData),
        );

        expect(response.statusCode, 200);

        final data = json.decode(response.body);
        expect(data['chapter'], 'chap84');
        expect(data['test_input'], isNotNull);
        expect(data['hybrid_result'], isNotNull);
      });

      test('Test hybride chap85', () async {
        final testData = {
          "test_data": [
            {
              "declaration_id": "TEST_SIMPLE_85",
              "libelle_produit": "Test produit simple",
              "valeur_caf": 500000,
              "poids_net": 50,
              "pays_origine": "FR",
              "code_bureau": "75001",
            },
          ],
        };

        final response = await http.post(
          Uri.parse('$baseUrl/predict/chap85/test-hybrid'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode(testData),
        );

        expect(response.statusCode, 200);

        final data = json.decode(response.body);
        expect(data['chapter'], 'chap85');
        expect(data['test_input'], isNotNull);
        expect(data['hybrid_result'], isNotNull);
      });
    });

    group('Tests d\'analyse', () {
      test('Analyse summary chap30', () async {
        final testData = [
          {
            "declaration_id": "ANALYSIS_TEST_1",
            "hybrid_score_ml": 0.7,
            "hybrid_score_metier": 0.6,
            "hybrid_score_final": 0.65,
            "hybrid_decision": "ZONE_GRISE",
            "hybrid_confidence": 0.8,
          },
          {
            "declaration_id": "ANALYSIS_TEST_2",
            "hybrid_score_ml": 0.3,
            "hybrid_score_metier": 0.2,
            "hybrid_score_final": 0.25,
            "hybrid_decision": "CONFORME",
            "hybrid_confidence": 0.9,
          },
        ];

        final response = await http.post(
          Uri.parse('$baseUrl/predict/chap30/analysis-summary'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode({"test_data": testData}),
        );

        expect(response.statusCode, 200);

        final data = json.decode(response.body);
        expect(data['chapter'], 'chap30');
        expect(data['summary'], isNotNull);
      });

      test('Risk analysis chap30', () async {
        final testData = [
          {
            "declaration_id": "RISK_TEST_1",
            "hybrid_score_ml": 0.8,
            "hybrid_score_metier": 0.9,
            "hybrid_score_final": 0.85,
            "hybrid_decision": "FRAUDE",
            "hybrid_confidence": 0.95,
          },
        ];

        final response = await http.post(
          Uri.parse('$baseUrl/predict/chap30/risk-analysis'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode({"test_data": testData}),
        );

        expect(response.statusCode, 200);

        final data = json.decode(response.body);
        expect(data['chapter'], 'chap30');
        expect(data['risk_analysis'], isNotNull);
      });
    });

    group('Tests de validation des données', () {
      test('Validation des données minimales', () async {
        final testData = {
          "test_data": [
            {"declaration_id": "MINIMAL_TEST"},
          ],
        };

        final response = await http.post(
          Uri.parse('$baseUrl/predict/chap30/test-hybrid'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode(testData),
        );

        expect(response.statusCode, 200);

        final data = json.decode(response.body);
        expect(data['chapter'], 'chap30');
        expect(data['test_input'], isNotNull);
        expect(data['hybrid_result'], isNotNull);
      });

      test('Validation des données complètes', () async {
        final testData = {
          "test_data": [
            {
              "declaration_id": "COMPLETE_TEST",
              "libelle_produit": "Produit complet avec accents éàç",
              "valeur_caf": 999999.99,
              "poids_net": 123.45,
              "pays_origine": "CN",
              "code_bureau": "15M",
            },
          ],
        };

        final response = await http.post(
          Uri.parse('$baseUrl/predict/chap30/test-hybrid'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode(testData),
        );

        expect(response.statusCode, 200);

        final data = json.decode(response.body);
        expect(data['chapter'], 'chap30');
        expect(data['test_input'], isNotNull);
        expect(data['hybrid_result'], isNotNull);
      });
    });

    group('Tests de performance', () {
      test('Tests de charge pour les endpoints critiques', () async {
        final startTime = DateTime.now();
        int successCount = 0;
        int totalTests = 10;

        for (int i = 0; i < totalTests; i++) {
          try {
            final response = await http.get(Uri.parse('$baseUrl/health'));
            if (response.statusCode == 200) {
              successCount++;
            }
          } catch (e) {
            // Ignorer les erreurs pour ce test de performance
          }
        }

        final endTime = DateTime.now();
        final duration = endTime.difference(startTime);

        expect(successCount, greaterThan(0));
        expect(duration.inSeconds, lessThan(30)); // Maximum 30 secondes

        print(
          'Performance: $successCount/$totalTests succès en ${duration.inMilliseconds}ms',
        );
      });
    });

    group('Tests de gestion d\'erreurs', () {
      test('Gestion des chapitres invalides', () async {
        final response = await http.get(
          Uri.parse('$baseUrl/predict/invalid/config'),
        );
        expect(response.statusCode, 400);

        final data = json.decode(response.body);
        expect(data['detail'], contains('non autorisé'));
      });

      test('Gestion des endpoints inexistants', () async {
        final response = await http.get(
          Uri.parse('$baseUrl/predict/chap30/inexistant'),
        );
        expect(response.statusCode, 404);
      });
    });

    group('Tests d\'intégration complète', () {
      test('Workflow complet d\'analyse', () async {
        // 1. Charger la configuration
        final configResponse = await http.get(
          Uri.parse('$baseUrl/predict/chap30/config'),
        );
        expect(configResponse.statusCode, 200);

        final configData = json.decode(configResponse.body);
        expect(configData['chapter'], 'chap30');

        // 2. Charger la configuration hybride
        final hybridResponse = await http.get(
          Uri.parse('$baseUrl/predict/chap30/hybrid-config'),
        );
        expect(hybridResponse.statusCode, 200);

        final hybridData = json.decode(hybridResponse.body);
        expect(hybridData['chapter'], 'chap30');

        // 3. Tester le système hybride
        final testData = {
          "test_data": [
            {
              "declaration_id": "WORKFLOW_TEST",
              "libelle_produit": "Test workflow complet",
              "valeur_caf": 1000000,
              "poids_net": 100,
              "pays_origine": "FR",
              "code_bureau": "75001",
            },
          ],
        };

        final hybridTestResponse = await http.post(
          Uri.parse('$baseUrl/predict/chap30/test-hybrid'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode(testData),
        );

        expect(hybridTestResponse.statusCode, 200);

        final hybridResult = json.decode(hybridTestResponse.body);
        expect(hybridResult['chapter'], 'chap30');

        // 4. Analyser les résultats
        final analysisData = [
          {
            "declaration_id": "WORKFLOW_TEST",
            "hybrid_score_ml": hybridResult['hybrid_result']['score_ml'],
            "hybrid_score_metier":
                hybridResult['hybrid_result']['score_metier'],
            "hybrid_score_final": hybridResult['hybrid_result']['score_final'],
            "hybrid_decision": hybridResult['hybrid_result']['decision'],
            "hybrid_confidence": hybridResult['hybrid_result']['confidence'],
          },
        ];

        final summaryResponse = await http.post(
          Uri.parse('$baseUrl/predict/chap30/analysis-summary'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode({"test_data": analysisData}),
        );

        expect(summaryResponse.statusCode, 200);

        final summary = json.decode(summaryResponse.body);
        expect(summary['chapter'], 'chap30');

        final riskResponse = await http.post(
          Uri.parse('$baseUrl/predict/chap30/risk-analysis'),
          headers: {'Content-Type': 'application/json'},
          body: json.encode({"test_data": analysisData}),
        );

        expect(riskResponse.statusCode, 200);

        final riskAnalysis = json.decode(riskResponse.body);
        expect(riskAnalysis['chapter'], 'chap30');

        // 5. Vérifier la cohérence des données
        expect(summary['chapter'], 'chap30');
        expect(riskAnalysis['chapter'], 'chap30');
        expect(summary['summary'], isNotNull);
        expect(riskAnalysis['risk_analysis'], isNotNull);
      });

      test('Validation de la cohérence entre chapitres', () async {
        final chapters = ['chap30', 'chap84', 'chap85'];
        final results = <String, dynamic>{};

        // Charger les configurations pour tous les chapitres
        for (final chapter in chapters) {
          final response = await http.get(
            Uri.parse('$baseUrl/predict/$chapter/config'),
          );
          expect(response.statusCode, 200);

          results[chapter] = json.decode(response.body);
        }

        // Vérifier que chaque chapitre a sa propre configuration
        for (final chapter in chapters) {
          expect(results[chapter]['chapter'], chapter);
          expect(results[chapter]['config'], isNotNull);
          expect(results[chapter]['config']['chapter'], isNotNull);
        }

        // Vérifier que les configurations sont différentes
        expect(results['chap30']['config']['chapter'], 30);
        expect(results['chap84']['config']['chapter'], 84);
        expect(results['chap85']['config']['chapter'], 85);
      });
    });
  });
}
