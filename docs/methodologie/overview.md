# 🔬 Vue d'Ensemble de la Méthodologie

## Pipeline Complet

Le projet est structuré en **4 phases progressives**, chacune apportant une famille de modèles différente :

```
┌─────────────────────────────────────────────────────────────────┐
│  DONNÉES BRUTES (10 min) → Resampling 1h → Dataset propre       │
└─────────────────────────────────────────┬───────────────────────┘
                                          │
           ┌──────────────────────────────▼──────────────────────┐
           │         PHASE 1 — Pipeline Classique (Masrour)       │
           │   Étapes 1→7 : Stationnarité, Décomposition,         │
           │   ACF/PACF, ARIMA, SARIMAX, Holt-Winters, Résidus    │
           └──────────────────────────────┬──────────────────────┘
                                          │
           ┌──────────────────────────────▼──────────────────────┐
           │           PHASE 2 — Deep Learning                    │
           │   ANN · LSTM · Bi-LSTM · GRU · CNN-LSTM · Attn-LSTM  │
           └──────────────────────────────┬──────────────────────┘
                                          │
           ┌──────────────────────────────▼──────────────────────┐
           │           PHASE 3 — XGBoost                          │
           │   Feature Engineering + GridSearchCV + TimeSeriesSplit│
           └──────────────────────────────┬──────────────────────┘
                                          │
           ┌──────────────────────────────▼──────────────────────┐
           │           PHASE 4 — Comparaison Finale               │
           │   Classement R², MAE, RMSE, MAPE + Visualisations    │
           └─────────────────────────────────────────────────────┘
```

---

## Métriques d'Évaluation

Tous les modèles sont évalués avec les **mêmes 4 métriques** :

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **R²** | `1 - SS_res/SS_tot` | Plus proche de 1 = meilleur |
| **MAE** | `mean(|y - ŷ|)` | Erreur absolue moyenne (g/h) |
| **RMSE** | `sqrt(mean((y-ŷ)²))` | Sensible aux grosses erreurs |
| **MAPE** | `mean(|y-ŷ|/y) × 100` | Erreur relative (%) |

### Grille de qualité R²

```python
def qualite(r2):
    if r2 >= 0.70: return '✅ Excellent'
    if r2 >= 0.50: return '✓ Bon'
    if r2 >= 0.30: return '⚠️ Acceptable'
    return '❌ Faible'
```

---

## Split Train / Validation / Test

| Ensemble | Phase 1 & 3 | Phase 2 (DL) |
|----------|-------------|--------------|
| **Train** | 85% | 70% |
| **Validation** | — | 15% |
| **Test** | 15% | 15% |

!!! note "Respect de l'ordre temporel"
    Aucun shuffle n'est appliqué. Le split respecte strictement la chronologie pour éviter toute **fuite de données** (*data leakage*).

---

## Configuration Globale

```python
TARGET   = 'Eau_Vaporisee_g_par_heure'
EXOG     = ['Consommation_HVAC_kWh', 'Temperature_C',
            'Humidite_Relative_%', 'CO2_ppm']
SEQ_LEN  = 24     # fenêtre de 24 heures pour les modèles DL
BATCH    = 32
EPOCHS   = 150
SEED     = 42
```
