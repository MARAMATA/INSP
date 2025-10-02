import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  group('Tests des métriques du Chapitre 85', () {
    const String baseUrl = 'http://localhost:8000';

    test('Vérification des métriques mises à jour du chapitre 85', () async {
      final response = await http.get(
        Uri.parse('$baseUrl/predict/chap85/config'),
      );
      expect(response.statusCode, 200);

      final data = json.decode(response.body);
      final config = data['config'];

      // Vérifier les nouvelles métriques
      expect(config['f1_score'], 0.965);
      expect(config['auc_score'], 0.994);
      expect(config['precision'], 0.99);
      expect(config['recall'], 0.942);
      expect(config['accuracy'], 0.965);
      expect(config['fraud_rate'], 19.2);
      expect(config['features_count'], 23);
      expect(config['brier_score'], 0.003);
      expect(config['ece'], 0.0006);
      expect(config['bss'], 0.9891);

      print('✅ Toutes les métriques du chapitre 85 sont correctes !');
    });

    test('Vérification des seuils de décision du chapitre 85', () async {
      final response = await http.get(
        Uri.parse('$baseUrl/predict/thresholds/chap85'),
      );
      expect(response.statusCode, 200);

      final data = json.decode(response.body);
      final thresholds = data['thresholds'];

      // Vérifier les nouveaux seuils
      expect(thresholds['conforme'], 0.192);
      expect(thresholds['fraude'], 0.557);
      expect(thresholds['optimal_threshold'], 0.5);

      print('✅ Les seuils de décision du chapitre 85 sont corrects !');
    });

    test('Test de prédiction avec les nouvelles métriques', () async {
      final testData = {
        'declaration_id':
            'TEST_METRICS_${DateTime.now().millisecondsSinceEpoch}',
        'valeur_caf': 1000000,
        'poids_net': 1000,
        'nombre_colis': 50,
        'code_sh_complet': '8528729000',
      };

      final response = await http.post(
        Uri.parse('$baseUrl/predict/chap85/test-pipeline'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode(testData),
      );

      expect(response.statusCode, 200);

      final data = json.decode(response.body);
      final pipelineResult = data['pipeline_result'];

      // Vérifier que la prédiction utilise les bonnes métriques
      expect(pipelineResult['chapter_info']['f1_score'], 0.965);
      expect(pipelineResult['chapter_info']['auc_score'], 0.994);
      expect(pipelineResult['chapter_info']['fraud_rate'], 19.2);
      expect(pipelineResult['calibration_metrics']['brier_score'], 0.003);
      expect(pipelineResult['calibration_metrics']['ece'], 0.0006);
      expect(pipelineResult['calibration_metrics']['bss'], 0.9891);

      print('✅ La prédiction utilise les bonnes métriques !');
    });
  });
}
