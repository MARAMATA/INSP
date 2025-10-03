import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:io';
import 'screens/login_screen.dart';
import 'screens/home_screen.dart';
import 'screens/upload_screen.dart';
import 'screens/pv_screen.dart';
import 'screens/feedback_screen.dart';
import 'screens/rl_performance_screen.dart';
import 'screens/rl_analytics_screen.dart';
import 'screens/pv_list_screen.dart';
import 'screens/pv_detail_screen.dart';
import 'screens/backend_test_screen.dart';
import 'screens/postgresql_test_screen.dart';
import 'screens/dashboard_screen.dart';
import 'screens/fraud_analytics_screen.dart';
import 'screens/ml_dashboard_screen.dart';
import 'utils/constants.dart';
import 'services/hybrid_backend_service.dart';
import 'services/user_session_service.dart';
import 'middleware/route_guard.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialiser les services
  await HybridBackendService.initialize();
  await UserSessionService.initialize();

  // Configuration spécifique pour macOS (uniquement si disponible)
  try {
    if (Platform.isMacOS) {
      // Permettre le redimensionnement de la fenêtre
      SystemChrome.setSystemUIOverlayStyle(
        SystemUiOverlayStyle(
          statusBarColor: Colors.transparent,
          systemNavigationBarColor: Colors.transparent,
        ),
      );
    }
  } catch (e) {
    // Ignorer l'erreur en mode web où Platform.isMacOS n'est pas disponible
    print('Configuration macOS ignorée (mode web)');
  }

  runApp(InspectIAApp());
}

class InspectIAApp extends StatelessWidget {
  const InspectIAApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'InspectIA',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        // 🎨 THÈME INSTITUTIONNEL - Douanes Sénégalaises
        primarySwatch: Colors.green,
        colorScheme: ColorScheme.fromSeed(
          seedColor: AppColors.primaryGreen,
          primary: AppColors.primaryGreen,
          secondary: AppColors.goldenYellow,
          surface: AppColors.surfaceLight,
          background: AppColors.backgroundLight,
          error: AppColors.discreetRed,
        ),
        scaffoldBackgroundColor: AppColors.backgroundLight,

        // Typographie institutionnelle
        fontFamily: 'Roboto',
        textTheme: TextTheme(
          headlineLarge: TextStyle(
            fontSize: 32,
            fontWeight: FontWeight.bold,
            color: AppColors.textDark,
            letterSpacing: 0.5,
          ),
          headlineMedium: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.w600,
            color: AppColors.textDark,
            letterSpacing: 0.2,
          ),
          titleLarge: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w600,
            color: AppColors.textDark,
            letterSpacing: 0.1,
          ),
          bodyLarge: TextStyle(
            fontSize: 16,
            color: AppColors.textDark,
            height: 1.5,
          ),
          bodyMedium: TextStyle(
            fontSize: 14,
            color: AppColors.textDark,
            height: 1.4,
          ),
        ),

        // AppBar institutionnel
        appBarTheme: AppBarTheme(
          backgroundColor: AppColors.primaryGreen,
          foregroundColor: AppColors.textInverse,
          elevation: AppSizes.elevationMedium,
          systemOverlayStyle: SystemUiOverlayStyle.light,
          titleTextStyle: TextStyle(
            color: AppColors.textInverse,
            fontSize: 20,
            fontWeight: FontWeight.bold,
            letterSpacing: 0.2,
          ),
          iconTheme: IconThemeData(
            color: AppColors.textInverse,
            size: AppSizes.iconSizeMedium,
          ),
        ),

        // Boutons institutionnels
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            backgroundColor: AppColors.primaryGreen,
            foregroundColor: AppColors.textInverse,
            padding: EdgeInsets.symmetric(
              horizontal: AppSizes.paddingXL,
              vertical: AppSizes.paddingMedium,
            ),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(AppSizes.radiusLarge),
            ),
            elevation: AppSizes.elevationMedium,
            textStyle: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.2,
            ),
          ),
        ),

        // Cartes institutionnelles
        cardTheme: CardThemeData(
          color: AppColors.surfaceLight,
          elevation: AppSizes.elevationMedium,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppSizes.radiusLarge),
          ),
          shadowColor: AppColors.textDark.withValues(alpha: 0.1),
        ),

        // Input decoration institutionnel
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: AppColors.pastelGreen.withValues(alpha: 0.3),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(AppSizes.radiusLarge),
            borderSide: BorderSide(color: AppColors.borderLight),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(AppSizes.radiusLarge),
            borderSide: BorderSide(color: AppColors.borderLight),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(AppSizes.radiusLarge),
            borderSide: BorderSide(color: AppColors.primaryGreen, width: 2),
          ),
          contentPadding: EdgeInsets.symmetric(
            horizontal: AppSizes.paddingMedium,
            vertical: AppSizes.paddingMedium,
          ),
        ),
      ),
      initialRoute: '/login',
      routes: {
        '/login': (context) => LoginScreen(),
        '/home': (context) =>
            ProtectedRoute(route: '/home', child: HomeScreen()),
        '/upload': (context) => ProtectedRoute(
          route: '/upload',
          requiredPermission: 'upload_documents',
          child: UploadScreen(),
        ),
        '/pv': (context) => ProtectedRoute(
          route: '/pv',
          requiredPermission: 'generate_pv',
          child: PVScreen(),
        ),
        '/feedback': (context) => ProtectedRoute(
          route: '/feedback',
          requiredPermission: 'provide_feedback',
          child: FeedbackScreen(),
        ),
        '/rl-performance': (context) => ProtectedRoute(
          route: '/rl-performance',
          requiredPermission: 'view_rl_performance',
          child: RLPerformanceScreen(),
        ),
        '/rl-analytics': (context) => ProtectedRoute(
          route: '/rl-analytics',
          requiredPermission: 'view_rl_analytics',
          child: RLAnalyticsScreen(),
        ),
        '/backend-test': (context) => ProtectedRoute(
          route: '/backend-test',
          requiredPermission: 'view_system_metrics',
          child: BackendTestScreen(),
        ),
        '/postgresql-test': (context) => ProtectedRoute(
          route: '/postgresql-test',
          requiredPermission: 'view_system_metrics',
          child: PostgreSQLTestScreen(),
        ),
        '/analysis': (context) => ProtectedRoute(
          route: '/analysis',
          requiredPermission: 'view_fraud_analytics',
          child: FraudAnalyticsScreen(),
        ),
        '/dashboard': (context) => ProtectedRoute(
          route: '/dashboard',
          requiredPermission: 'view_dashboard',
          child: DashboardScreen(),
        ),
        '/fraud-analytics': (context) => ProtectedRoute(
          route: '/fraud-analytics',
          requiredPermission: 'view_fraud_analytics',
          child: FraudAnalyticsScreen(),
        ),
        '/ml-dashboard': (context) => ProtectedRoute(
          route: '/ml-dashboard',
          requiredPermission: 'view_ml_dashboard',
          child: MLDashboardScreen(),
        ),
        '/pv-list': (context) {
          final args =
              ModalRoute.of(context)!.settings.arguments
                  as Map<String, dynamic>?;
          String chapter = args?['chapter'] ?? 'chap30';
          return ProtectedRoute(
            route: '/pv-list',
            requiredPermission: 'view_pv_list',
            child: PVListScreen(chapter: chapter),
          );
        },
        '/pv-detail': (context) {
          final args =
              ModalRoute.of(context)!.settings.arguments
                  as Map<String, dynamic>;
          String chapter = args['chapter'] ?? '';
          String pvId = args['pv_id'] ?? '';

          // Si pas de chapitre, essayer de l'extraire de l'ID du PV de manière générique
          if (chapter.isEmpty && pvId.isNotEmpty) {
            // Extraire le chapitre de manière générique en cherchant le pattern chapXX
            final regex = RegExp(r'chap(\d+)');
            final match = regex.firstMatch(pvId);
            if (match != null) {
              chapter = 'chap${match.group(1)}';
            } else {
              // Si on ne peut pas déterminer, utiliser le chapitre par défaut
              chapter = 'chap30';
            }
          }

          return ProtectedRoute(
            route: '/pv-detail',
            requiredPermission: 'view_pv_details',
            child: PVDetailScreen(chapter: chapter, pvId: pvId),
          );
        },
      },
    );
  }
}
