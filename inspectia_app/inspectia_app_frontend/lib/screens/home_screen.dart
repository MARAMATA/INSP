import 'package:flutter/material.dart';
import '../utils/constants.dart';
import '../services/user_session_service.dart';
import '../models/user_profile.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    // Recharger le profil utilisateur pour s'assurer d'avoir les dernières permissions
    UserSessionService.reloadUserProfile();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(AppText.appName),
        actions: [
          IconButton(
            icon: Icon(Icons.refresh),
            onPressed: () async {
              // Recharger le profil utilisateur
              await UserSessionService.reloadUserProfile();
              setState(() {}); // Forcer le rechargement de l'interface
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text('Permissions mises à jour'),
                  backgroundColor: AppColors.successGreen,
                  duration: Duration(seconds: 2),
                ),
              );
            },
          ),
          IconButton(
            icon: Icon(Icons.logout),
            onPressed: () async {
              // Logout avec le nouveau système
              await UserSessionService.logout();
              Navigator.of(context).pushReplacementNamed('/login');
            },
          ),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [AppColors.backgroundLight, Colors.white],
          ),
        ),
        child: SingleChildScrollView(
          padding: EdgeInsets.all(AppSizes.paddingLarge),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // En-tête principal
              Container(
                padding: EdgeInsets.all(AppSizes.paddingLarge),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [AppColors.primaryGreen, AppColors.discreetRed],
                  ),
                  borderRadius: BorderRadius.circular(AppSizes.radiusLarge),
                ),
                child: Row(
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            'Bienvenue, ${UserSessionService.displayName}',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 20,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          SizedBox(height: AppSizes.paddingSmall),
                          Text(
                            AppText.appSubtitle,
                            style: TextStyle(
                              color: Colors.white.withValues(alpha: 0.9),
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Container(
                      padding: EdgeInsets.symmetric(
                        horizontal: AppSizes.paddingMedium,
                        vertical: AppSizes.paddingSmall,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white.withValues(alpha: 0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        UserSessionService.currentUser?.role.name ??
                            'InspectIA',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: Colors.white,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(height: AppSizes.paddingLarge),

              // Actions principales
              Text(
                'Actions Principales',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: AppColors.textDark,
                ),
              ),
              SizedBox(height: AppSizes.paddingMedium),
              GridView.count(
                crossAxisCount: 4,
                crossAxisSpacing: 8,
                mainAxisSpacing: 8,
                childAspectRatio: 1.0,
                shrinkWrap: true,
                physics: NeverScrollableScrollPhysics(),
                children: _buildActionCards(context),
              ),

              // ✅ Sections "Statistiques Temps Réel" et "Accès Rapide" supprimées pour l'inspecteur
              // Ces sections sont affichées uniquement pour les autres rôles (Expert ML, Chef de Service)
              if (UserSessionService.currentUser?.role !=
                  UserRole.inspecteur) ...[
                SizedBox(height: AppSizes.paddingLarge),

                // Statistiques
                Container(
                  padding: EdgeInsets.all(AppSizes.paddingMedium),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
                    border: Border.all(color: AppColors.borderLight),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.1),
                        blurRadius: 4,
                        offset: Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(
                            Icons.analytics,
                            color: AppColors.primaryGreen,
                            size: 20,
                          ),
                          SizedBox(width: AppSizes.paddingSmall),
                          Text(
                            'Statistiques Temps Réel',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: AppColors.textDark,
                            ),
                          ),
                        ],
                      ),
                      SizedBox(height: AppSizes.paddingSmall),

                      // Données temps réel
                      Row(
                        children: [
                          Icon(
                            Icons.category,
                            color: AppColors.primaryGreen,
                            size: 16,
                          ),
                          SizedBox(width: AppSizes.paddingSmall),
                          Text(
                            'Chapitre actuel: Sélectionnez un chapitre',
                            style: TextStyle(
                              color: AppColors.textDark,
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),

                SizedBox(height: AppSizes.paddingLarge),

                // Accès rapide
                Text(
                  'Accès Rapide',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: AppColors.textDark,
                  ),
                ),
                SizedBox(height: AppSizes.paddingMedium),
                Row(
                  children: [
                    Expanded(
                      child: _buildQuickAccessCard(
                        context,
                        'Analyse de Fichiers',
                        Icons.upload_file,
                        AppColors.primaryGreen,
                        '/upload',
                      ),
                    ),
                    SizedBox(width: AppSizes.paddingMedium),
                    Expanded(
                      child: _buildQuickAccessCard(
                        context,
                        'Performances RL',
                        Icons.psychology,
                        AppColors.discreetRed,
                        '/rl-performance',
                      ),
                    ),
                  ],
                ),
                SizedBox(height: AppSizes.paddingMedium),
                Row(
                  children: [
                    Expanded(
                      child: _buildQuickAccessCard(
                        context,
                        'Analytics RL',
                        Icons.analytics,
                        AppColors.successGreen,
                        '/rl-analytics',
                      ),
                    ),
                    SizedBox(width: AppSizes.paddingMedium),
                    Expanded(
                      child: _buildQuickAccessCard(
                        context,
                        'Dashboard Chef',
                        Icons.dashboard,
                        AppColors.warningOrange,
                        '/dashboard',
                      ),
                    ),
                  ],
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  List<Widget> _buildActionCards(BuildContext context) {
    // Obtenir les actions disponibles selon le rôle de l'utilisateur
    final currentUser = UserSessionService.currentUser;
    if (currentUser == null) return [];

    List<Map<String, dynamic>> actions = [];

    switch (currentUser.role) {
      case UserRole.inspecteur:
        actions = [
          {
            'title': 'Upload',
            'icon': Icons.upload_file,
            'route': '/upload',
            'color': AppColors.primaryGreen,
          },
          {
            'title': 'Analyse',
            'icon': Icons.analytics,
            'route': '/analysis',
            'color': AppColors.discreetRed,
          },
          {
            'title': 'Feedback',
            'icon': Icons.feedback,
            'route': '/feedback',
            'color': AppColors.infoBlue,
          },
          {
            'title': 'PV',
            'icon': Icons.description,
            'route': '/pv',
            'color': AppColors.goldenYellow,
          },
        ];
        break;
      case UserRole.expertML:
        actions = [
          {
            'title': 'Upload',
            'icon': Icons.upload_file,
            'route': '/upload',
            'color': AppColors.primaryGreen,
          },
          {
            'title': 'Analyse',
            'icon': Icons.analytics,
            'route': '/analysis',
            'color': AppColors.discreetRed,
          },
          {
            'title': 'RL Performance',
            'icon': Icons.psychology,
            'route': '/rl-performance',
            'color': AppColors.successGreen,
          },
          {
            'title': 'RL Analytics',
            'icon': Icons.analytics,
            'route': '/rl-analytics',
            'color': AppColors.successGreen,
          },
          {
            'title': 'ML Dashboard',
            'icon': Icons.dashboard,
            'route': '/ml-dashboard',
            'color': Colors.purple,
          },
          {
            'title': 'Feedback',
            'icon': Icons.feedback,
            'route': '/feedback',
            'color': AppColors.infoBlue,
          },
          {
            'title': 'PV',
            'icon': Icons.description,
            'route': '/pv',
            'color': AppColors.goldenYellow,
          },
          {
            'title': 'Test Backend',
            'icon': Icons.science,
            'route': '/backend-test',
            'color': AppColors.textDark,
          },
          {
            'title': 'PostgreSQL',
            'icon': Icons.storage,
            'route': '/postgresql-test',
            'color': AppColors.primaryGreen,
          },
        ];
        break;
      case UserRole.chefService:
        actions = [
          {
            'title': 'Dashboard',
            'icon': Icons.dashboard,
            'route': '/dashboard',
            'color': AppColors.warningOrange,
          },
          {
            'title': 'Analyse',
            'icon': Icons.analytics,
            'route': '/analysis',
            'color': AppColors.discreetRed,
          },
          {
            'title': 'Test Backend',
            'icon': Icons.science,
            'route': '/backend-test',
            'color': AppColors.textDark,
          },
          {
            'title': 'PostgreSQL',
            'icon': Icons.storage,
            'route': '/postgresql-test',
            'color': AppColors.primaryGreen,
          },
        ];
        break;
    }

    return actions
        .map(
          (action) => _buildActionCard(
            context,
            action['title'],
            action['icon'],
            action['color'],
            action['route'],
          ),
        )
        .toList();
  }

  Widget _buildActionCard(
    BuildContext context,
    String title,
    IconData icon,
    Color color,
    String route,
  ) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
      ),
      child: InkWell(
        onTap: () => Navigator.pushNamed(context, route),
        borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
        child: Container(
          padding: EdgeInsets.all(AppSizes.paddingSmall),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 32, color: color),
              SizedBox(height: AppSizes.paddingSmall),
              Text(
                title,
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textDark,
                ),
                textAlign: TextAlign.center,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildQuickAccessCard(
    BuildContext context,
    String title,
    IconData icon,
    Color color,
    String route,
  ) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
      ),
      child: InkWell(
        onTap: () => Navigator.pushNamed(context, route),
        borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
        child: Container(
          padding: EdgeInsets.all(AppSizes.paddingMedium),
          child: Column(
            children: [
              Icon(icon, size: 40, color: color),
              SizedBox(height: AppSizes.paddingSmall),
              Text(
                title,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textDark,
                ),
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
