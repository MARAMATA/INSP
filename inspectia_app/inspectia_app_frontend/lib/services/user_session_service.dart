import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/user_profile.dart';

/// 🔐 SERVICE DE GESTION DES SESSIONS UTILISATEUR
/// Gère la session de l'utilisateur connecté et ses permissions

class UserSessionService {
  static const String _userKey = 'current_user';
  static UserProfile? _currentUser;

  /// Initialise le service de session
  static Future<void> initialize() async {
    await _loadUserFromStorage();
  }

  /// Charge l'utilisateur depuis le stockage local
  static Future<void> _loadUserFromStorage() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final userJson = prefs.getString(_userKey);

      if (userJson != null) {
        final userMap = jsonDecode(userJson) as Map<String, dynamic>;
        _currentUser = UserProfile.fromJson(userMap);
      }
    } catch (e) {
      print('Erreur lors du chargement de la session: $e');
      _currentUser = null;
    }
  }

  /// Sauvegarde l'utilisateur dans le stockage local
  static Future<void> _saveUserToStorage(UserProfile user) async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final userJson = jsonEncode(user.toJson());
      await prefs.setString(_userKey, userJson);
    } catch (e) {
      print('Erreur lors de la sauvegarde de la session: $e');
    }
  }

  /// Connecte un utilisateur
  static Future<bool> login(String username, String password) async {
    // Vérifier les identifiants prédéfinis
    final validCredentials = {
      'inspecteur': 'inspecteur123',
      'expert_ml': 'expert123',
      'chef_service': 'chef123',
    };

    if (validCredentials.containsKey(username) &&
        validCredentials[username] == password) {
      // Déterminer le profil selon le nom d'utilisateur
      UserProfile profile;
      switch (username) {
        case 'inspecteur':
          profile = UserProfile.inspecteurProfile;
          break;
        case 'expert_ml':
          profile = UserProfile.expertMLProfile;
          break;
        case 'chef_service':
          profile = UserProfile.chefServiceProfile;
          break;
        default:
          return false;
      }

      _currentUser = profile;
      await _saveUserToStorage(_currentUser!);
      return true;
    }

    return false;
  }

  /// Déconnecte l'utilisateur actuel
  static Future<void> logout() async {
    _currentUser = null;
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(_userKey);
    } catch (e) {
      print('Erreur lors de la déconnexion: $e');
    }
  }

  /// Retourne l'utilisateur actuellement connecté
  static UserProfile? get currentUser => _currentUser;

  /// Vérifie si un utilisateur est connecté
  static bool get isLoggedIn => _currentUser != null;

  /// Retourne le rôle de l'utilisateur actuel
  static UserRole? get currentRole => _currentUser?.role;

  /// Vérifie si l'utilisateur actuel a une permission
  static bool hasPermission(String permission) {
    return _currentUser?.hasPermission(permission) ?? false;
  }

  /// Retourne les pages accessibles pour l'utilisateur actuel
  static List<String> get accessiblePages {
    return _currentUser?.accessiblePages ?? [];
  }

  /// Retourne le nom d'affichage de l'utilisateur actuel
  static String get displayName {
    if (_currentUser == null) return 'Utilisateur';
    return _currentUser!.fullName;
  }

  /// Retourne la description du rôle actuel
  static String get roleDescription {
    return _currentUser?.roleDescription ?? 'Aucun rôle défini';
  }

  /// Vérifie si l'utilisateur actuel peut accéder à une route
  static bool canAccessRoute(String route) {
    if (_currentUser == null) return false;

    // Vérifier si la route est dans les pages accessibles
    return _currentUser!.accessiblePages.contains(route);
  }

  /// Retourne la route de redirection par défaut selon le rôle
  static String get defaultRoute {
    if (_currentUser == null) return '/login';

    switch (_currentUser!.role) {
      case UserRole.chefService:
        return '/dashboard';
      case UserRole.inspecteur:
      case UserRole.expertML:
        return '/home';
    }
  }

  /// Retourne les informations de session
  static Map<String, dynamic> get sessionInfo {
    if (_currentUser == null) {
      return {
        'isLoggedIn': false,
        'username': null,
        'role': null,
        'permissions': [],
        'accessiblePages': [],
      };
    }

    return {
      'isLoggedIn': true,
      'username': _currentUser!.username,
      'role': _currentUser!.role.name,
      'permissions': _currentUser!.permissions,
      'accessiblePages': _currentUser!.accessiblePages,
      'defaultRoute': defaultRoute,
    };
  }

  /// Met à jour la session (utile pour les changements de rôle)
  static Future<void> updateSession(UserProfile user) async {
    _currentUser = user;
    await _saveUserToStorage(user);
  }

  /// Force le rechargement du profil utilisateur avec les dernières permissions
  static Future<void> reloadUserProfile() async {
    if (_currentUser != null) {
      // Recharger le profil depuis les profils prédéfinis
      UserProfile profile;
      switch (_currentUser!.username) {
        case 'inspecteur':
          profile = UserProfile.inspecteurProfile;
          break;
        case 'expert_ml':
          profile = UserProfile.expertMLProfile;
          break;
        case 'chef_service':
          profile = UserProfile.chefServiceProfile;
          break;
        default:
          return;
      }

      _currentUser = profile;
      await _saveUserToStorage(_currentUser!);
    }
  }
}
