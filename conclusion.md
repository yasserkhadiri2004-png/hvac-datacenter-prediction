# 🏁 Conclusion

## Synthèse

Ce projet a démontré la faisabilité de **prédire la quantité d'eau vaporisée** du bassin de la tour de refroidissement d'un data center à partir de capteurs IoT (température, humidité, CO₂, consommation HVAC).

### Réponse à la Problématique

> *"Quelle famille de modèles offre le meilleur compromis entre précision et interprétabilité ?"*

| Critère | Phase 1 (Statistique) | Phase 2 (DL) | Phase 3 (XGBoost) |
|---------|----------------------|-------------|-------------------|
| **Précision (R²)** | Modérée | Haute | **Très haute** |
| **Interprétabilité** | **Haute** | Faible | Haute (SHAP) |
| **Vitesse d'entraînement** | **Rapide** | Lente | Rapide |
| **Intervalles de confiance** | **Oui (SARIMAX)** | Non | Non |
| **Usage opérationnel** | ✓ | ⚠️ | **✅** |

**Conclusion** : XGBoost représente le meilleur compromis pour un usage opérationnel dans un data center — précision élevée, interprétable via feature importance, et rapide à réentraîner.

---

## Limites

!!! warning "Limites identifiées"
    1. **Horizon mono-step** : Le projet prédit uniquement t+1. Un horizon multi-pas (h+6, h+12, h+24) serait plus utile opérationnellement.
    2. **Callbacks DL désactivés** : Les callbacks `EarlyStopping` et `ReduceLROnPlateau` sont définis mais non utilisés — à réactiver en production.
    3. **Pas de SHAP** : L'interprétabilité XGBoost se limite à la feature importance. L'ajout de valeurs SHAP permettrait une analyse plus fine.
    4. **Données d'un seul DC** : La généralisation à d'autres data centers nécessiterait une validation croisée sur plusieurs sites.

---

## Perspectives

| Amélioration | Impact | Difficulté |
|-------------|--------|-----------|
| Prévision multi-horizon (h+6/12/24) | ⭐⭐⭐ Élevé | Moyen |
| Ajout valeurs SHAP | ⭐⭐ Moyen | Faible |
| Réactivation callbacks EarlyStopping | ⭐⭐ Moyen | Très faible |
| Déploiement API REST (FastAPI) | ⭐⭐⭐ Élevé | Élevé |
| Dashboard temps réel (Streamlit) | ⭐⭐ Moyen | Moyen |
| Détection d'anomalies (fuite bassin) | ⭐⭐⭐ Élevé | Élevé |

---

## Références

- Masrour, T. (2024-2025). *Cours Séries Temporelles*. ENSAM-IATD.
- Hyndman, R. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*. OTexts.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- ASHRAE (2023). *Handbook — HVAC Systems and Equipment*. Chapter 40: Cooling Towers.
