import 'package:flutter/material.dart';
import '../services/user_session_service.dart';
import '../models/user_profile.dart';
import '../utils/constants.dart';

class RouteGuard extends StatelessWidget {
  final String route;
  final Widget child;
  final String? requiredPermission;

  const RouteGuard({
    super.key,
    required this.route,
    required this.child,
    this.requiredPermission,
  });

  @override
  Widget build(BuildContext context) {
    // Vérifier si l'utilisateur est connecté
    if (!UserSessionService.isLoggedIn) {
      print('RouteGuard: Utilisateur non connecté');
      return _buildUnauthorizedScreen(context);
    }

    // Debug: Afficher les informations de la session
    final currentUser = UserSessionService.currentUser;
    print('RouteGuard: Utilisateur connecté: ${currentUser?.username}');
    print('RouteGuard: Rôle: ${currentUser?.role.name}');
    print('RouteGuard: Route demandée: $route');
    print(
      'RouteGuard: Pages accessibles: ${UserSessionService.accessiblePages}',
    );
    print(
      'RouteGuard: Peut accéder à la route: ${UserSessionService.canAccessRoute(route)}',
    );

    // Vérifier si l'utilisateur peut accéder à cette route
    if (!UserSessionService.canAccessRoute(route)) {
      print('RouteGuard: Accès refusé - route non autorisée');

      // Si c'est un chef de service, rediriger vers le dashboard
      if (currentUser?.role == UserRole.chefService) {
        print('RouteGuard: Redirection chef de service vers dashboard');
        WidgetsBinding.instance.addPostFrameCallback((_) {
          Navigator.pushReplacementNamed(context, '/dashboard');
        });
        return Container(); // Widget vide pendant la redirection
      }

      return _buildForbiddenScreen(context);
    }

    // Vérifier la permission spécifique si demandée
    if (requiredPermission != null &&
        !UserSessionService.hasPermission(requiredPermission!)) {
      print(
        'RouteGuard: Accès refusé - permission manquante: $requiredPermission',
      );
      print('RouteGuard: Permissions disponibles: ${currentUser?.permissions}');
      return _buildForbiddenScreen(context);
    }

    print('RouteGuard: Accès autorisé');
    return child;
  }

  Widget _buildUnauthorizedScreen(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Accès non autorisé'),
        backgroundColor: AppColors.discreetRed,
        foregroundColor: Colors.white,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [AppColors.backgroundLight, Colors.white],
          ),
        ),
        child: Center(
          child: Padding(
            padding: EdgeInsets.all(AppSizes.paddingLarge),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  Icons.lock_outline,
                  size: 80,
                  color: AppColors.discreetRed,
                ),
                SizedBox(height: AppSizes.paddingLarge),
                Text(
                  'Accès non autorisé',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textDark,
                  ),
                ),
                SizedBox(height: AppSizes.paddingMedium),
                Text(
                  'Vous devez être connecté pour accéder à cette page.',
                  style: TextStyle(fontSize: 16, color: AppColors.textLight),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: AppSizes.paddingLarge),
                ElevatedButton(
                  onPressed: () {
                    Navigator.pushReplacementNamed(context, '/login');
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primaryGreen,
                    foregroundColor: Colors.white,
                    padding: EdgeInsets.symmetric(
                      horizontal: AppSizes.paddingLarge,
                      vertical: AppSizes.paddingMedium,
                    ),
                  ),
                  child: Text('Se connecter'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildForbiddenScreen(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Accès interdit'),
        backgroundColor: AppColors.warningOrange,
        foregroundColor: Colors.white,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [AppColors.backgroundLight, Colors.white],
          ),
        ),
        child: Center(
          child: Padding(
            padding: EdgeInsets.all(AppSizes.paddingLarge),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.block, size: 80, color: AppColors.warningOrange),
                SizedBox(height: AppSizes.paddingLarge),
                Text(
                  'Accès interdit',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textDark,
                  ),
                ),
                SizedBox(height: AppSizes.paddingMedium),
                Text(
                  'Vous n\'avez pas les permissions nécessaires pour accéder à cette page.',
                  style: TextStyle(fontSize: 16, color: AppColors.textLight),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: AppSizes.paddingMedium),
                Container(
                  padding: EdgeInsets.all(AppSizes.paddingMedium),
                  decoration: BoxDecoration(
                    color: AppColors.pastelYellow,
                    borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
                    border: Border.all(color: AppColors.warningOrange),
                  ),
                  child: Column(
                    children: [
                      Text(
                        'Rôle actuel: ${UserSessionService.displayName}',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: AppColors.textDark,
                        ),
                      ),
                      SizedBox(height: AppSizes.paddingSmall),
                      Text(
                        UserSessionService.roleDescription,
                        style: TextStyle(
                          fontSize: 14,
                          color: AppColors.textLight,
                        ),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
                SizedBox(height: AppSizes.paddingLarge),
                ElevatedButton(
                  onPressed: () {
                    Navigator.pushReplacementNamed(context, '/home');
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primaryGreen,
                    foregroundColor: Colors.white,
                    padding: EdgeInsets.symmetric(
                      horizontal: AppSizes.paddingLarge,
                      vertical: AppSizes.paddingMedium,
                    ),
                  ),
                  child: Text('Retour à l\'accueil'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// Widget helper pour protéger une route
class ProtectedRoute extends StatelessWidget {
  final String route;
  final Widget child;
  final String? requiredPermission;

  const ProtectedRoute({
    super.key,
    required this.route,
    required this.child,
    this.requiredPermission,
  });

  @override
  Widget build(BuildContext context) {
    return RouteGuard(
      route: route,
      requiredPermission: requiredPermission,
      child: child,
    );
  }
}
