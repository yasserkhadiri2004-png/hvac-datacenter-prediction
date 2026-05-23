# 📈 Résultats

## Résumé des Performances

Le tableau suivant présente les résultats attendus après exécution complète du notebook sur les données de 3 mois.

### Phase 1 — Modèles Statistiques

| Modèle | R² | MAE (g/h) | RMSE (g/h) | Qualité |
|--------|-----|-----------|------------|---------|
| ARIMA(1,d,1) | ~0.35–0.55 | — | — | ⚠️ Acceptable |
| SARIMAX(1,d,1)×(1,0,1,24) | ~0.50–0.65 | — | — | ✓ Bon |
| Holt-Winters | ~0.45–0.60 | — | — | ✓ Bon |

!!! info "Atout du SARIMAX"
    SARIMAX est le meilleur de la Phase 1 car il intègre les **variables exogènes** (HVAC, température, humidité, CO₂) en plus de la composante saisonnière S=24h.

---

### Phase 2 — Deep Learning

| Modèle | Architecture | Atout principal |
|--------|-------------|-----------------|
| ANN | Dense Feed-Forward | Baseline simple |
| LSTM | 2 couches LSTM | Dépendances long terme |
| Bi-LSTM | LSTM bidirectionnel | Contexte passé + futur |
| GRU | 2 couches GRU | Léger, rapide |
| CNN-LSTM | Conv1D + LSTM | Motifs locaux + mémoire |
| Attention-LSTM | LSTM + Multi-Head Attention | Focus sur pas importants |

!!! tip "Modèles attendus les plus performants"
    **Attention-LSTM** et **CNN-LSTM** sont généralement les meilleurs sur les séries temporelles multivariées avec saisonnalité forte.

---

### Phase 3 — XGBoost

| Modèle | R² CV | R² Test | Qualité |
|--------|-------|---------|---------|
| XGBoost (optimisé) | ~0.80–0.92 | ~0.78–0.90 | ✅ Excellent |

!!! success "XGBoost souvent vainqueur"
    Grâce au feature engineering riche (27 features : lags, rolling stats, interactions, encodage cyclique), XGBoost surpasse souvent les modèles DL sur des séries courtes à saisonnalité régulière.

---

## Checklist Finale

```
╔══════════════════════════════════════════════════════════════╗
║  CHECKLIST — COURS PR. MASROUR (2024-2025)                   ║
╠══════════════════════════════════════════════════════════════╣
║  ✅ Données nettoyées et sans valeurs manquantes             ║
║  ✅ Étape 1 — Visualisation                                  ║
║  ✅ Étape 2 — Tests ADF + KPSS                               ║
║  ✅ Étape 3 — Décomposition STL  yt = Tt + St + Rt           ║
║  ✅ Étape 4 — ACF/PACF → p, d, q                             ║
║  ✅ Étape 5 — ARIMA + SARIMAX + Holt-Winters                 ║
║  ✅ Étape 6 — Résidus (Ljung-Box + règle 3σ)                 ║
║  ✅ Étape 7 — Prévision + IC 80% et 95%                      ║
║  ✅ Phase 2 — 6 modèles Deep Learning                        ║
║  ✅ Phase 3 — XGBoost (GridSearchCV + TimeSeriesSplit)       ║
║  ✅ Phase 4 — Comparaison finale multi-phases                ║
╚══════════════════════════════════════════════════════════════╝
```
