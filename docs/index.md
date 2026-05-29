# 💧 Prédiction Vaporisation Eau — Tour de Refroidissement Data Center

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green)](https://xgboost.readthedocs.io)
[![Licence](https://img.shields.io/badge/Licence-MIT-lightgrey)](LICENSE)

**Auteur :** Yasser Khadiri  
**Référence :** *Cours Séries Temporelles* — Pr. Tawfik Masrour (ENSAM-IATD, 2024-2025)

</div>

---

## 🎯 En bref

Ce projet applique des **techniques avancées de séries temporelles** pour prédire la quantité d'eau vaporisée par heure depuis le bassin de la tour de refroidissement d'un **data center**, à partir de capteurs environnementaux et énergétiques.

Il compare trois familles de modèles sur des données réelles collectées sur **3 mois** à pas de **10 minutes** :

| Phase | Approche | Modèles |
|-------|----------|---------|
| **Phase 1** | Statistique Classique | ARIMA, SARIMAX, Holt-Winters |
| **Phase 2** | Deep Learning | ANN, LSTM, Bi-LSTM, GRU, CNN-LSTM, Attention-LSTM |
| **Phase 3** | Machine Learning | XGBoost + GridSearchCV |
| **Phase 4** | Comparaison | Classement final multi-métriques |

---

## 🏢 Contexte Industriel

Dans un data center, les serveurs génèrent une chaleur considérable. La dissipation de cette chaleur repose sur un circuit hydraulique qui aboutit à une **tour de refroidissement** évaporative.

```
┌─────────────┐    chaleur    ┌──────────────┐    eau chaude   ┌────────────────┐
│   Serveurs  │ ─────────────▶│  Échangeur   │ ───────────────▶│  Tour de       │
│  (salles)   │               │  de chaleur  │                 │  Refroid.      │
└─────────────┘               └──────────────┘                 │  (Bassin)      │
                                                                │  ↑ Évaporation │
                                                                └────────────────┘
```

L'eau du bassin s'évapore pour absorber la chaleur — c'est ce **débit d'évaporation (g/h)** que ce projet cherche à prédire.

---

## 🚀 Navigation

<div class="grid cards" markdown>

- 📋 **[Introduction](introduction.md)**  
  Problématique, objectifs et état de l'art

- 📊 **[Dataset](dataset.md)**  
  Description des variables, statistiques descriptives

- ⚙️ **[Installation](installation.md)**  
  Prérequis, dépendances et configuration

- 🔬 **[Méthodologie](methodologie/overview.md)**  
  Les 4 phases de la démarche complète

- 📈 **[Résultats](resultats.md)**  
  Tableaux comparatifs, graphiques, métriques

- 🏁 **[Conclusion](conclusion.md)**  
  Synthèse, limites et perspectives

</div>
