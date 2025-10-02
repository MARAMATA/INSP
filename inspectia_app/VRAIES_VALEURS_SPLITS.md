# VRAIES VALEURS DES SPLITS TRAIN/TEST/VALIDATION

## ANALYSE DES FICHIERS DE SPLITS

### Chapitre 30 - Produits Pharmaceutiques
**Fichiers analysés :**
- `backend/data/ml_splits/chap30/X_train.csv` : 35,515 lignes (35,514 échantillons + header)
- `backend/data/ml_splits/chap30/X_valid.csv` : 8,881 lignes (8,879 échantillons + header)
- `backend/data/ml_splits/chap30/X_test.csv` : 11,101 lignes (11,099 échantillons + header)

**Vraies valeurs des splits :**
- **Train** : 35,514 échantillons
- **Validation** : 8,879 échantillons
- **Test** : 11,099 échantillons
- **Total** : 55,492 échantillons

### Chapitre 84 - Machines et Équipements Mécaniques
**Fichiers analysés :**
- `backend/data/ml_splits/chap84/X_train.csv` : 88,399 lignes (88,397 échantillons + header)
- `backend/data/ml_splits/chap84/X_valid.csv` : 22,102 lignes (22,100 échantillons + header)
- `backend/data/ml_splits/chap84/X_test.csv` : 27,627 lignes (27,625 échantillons + header)

**Vraies valeurs des splits :**
- **Train** : 88,397 échantillons
- **Validation** : 22,100 échantillons
- **Test** : 27,625 échantillons
- **Total** : 138,122 échantillons

### Chapitre 85 - Machines et Appareils Électriques
**Fichiers analysés :**
- `backend/data/ml_splits/chap85/X_train.csv` : 83,502 lignes (83,500 échantillons + header)
- `backend/data/ml_splits/chap85/X_valid.csv` : 20,878 lignes (20,876 échantillons + header)
- `backend/data/ml_splits/chap85/X_test.csv` : 26,097 lignes (26,095 échantillons + header)

**Vraies valeurs des splits :**
- **Train** : 83,500 échantillons
- **Validation** : 20,876 échantillons
- **Test** : 26,095 échantillons
- **Total** : 130,471 échantillons

## TOTAL GLOBAL

**Total de tous les chapitres :**
- **Chapitre 30** : 55,492 échantillons
- **Chapitre 84** : 138,122 échantillons
- **Chapitre 85** : 130,471 échantillons
- **TOTAL GLOBAL** : 324,085 échantillons

## COMPARAISON AVEC LES FICHIERS DE RÉSULTATS

### Incohérences détectées

**Chapitre 30 :**
- **Fichiers de splits** : 55,492 échantillons (35,514 train + 8,879 valid + 11,099 test)
- **Fichiers de résultats** : 55,495 échantillons (44,396 train + 11,099 test)
- **Différence** : +3 échantillons dans les résultats, pas de validation mentionnée

**Chapitre 84 :**
- **Fichiers de splits** : 138,122 échantillons (88,397 train + 22,100 valid + 27,625 test)
- **Fichiers de résultats** : 138,250 échantillons (110,500 train + 27,625 test)
- **Différence** : +128 échantillons dans les résultats, pas de validation mentionnée

**Chapitre 85 :**
- **Fichiers de splits** : 130,471 échantillons (83,500 train + 20,876 valid + 26,095 test)
- **Fichiers de résultats** : 130,475 échantillons (104,380 train + 26,095 test)
- **Différence** : +4 échantillons dans les résultats, pas de validation mentionnée

## CONCLUSION

Les **fichiers de splits** contiennent les vraies valeurs avec la répartition train/validation/test, tandis que les **fichiers de résultats** ne montrent que train/test et ont des totaux légèrement différents.

**Valeurs à utiliser dans le mémoire :**
- **Chapitre 30** : 55,492 échantillons (35,514 train + 8,879 valid + 11,099 test)
- **Chapitre 84** : 138,122 échantillons (88,397 train + 22,100 valid + 27,625 test)
- **Chapitre 85** : 130,471 échantillons (83,500 train + 20,876 valid + 26,095 test)
- **TOTAL GLOBAL** : 324,085 échantillons

Ces valeurs sont plus précises et reflètent la vraie répartition des données utilisée pour l'entraînement des modèles.









