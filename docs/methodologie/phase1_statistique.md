# Phase 1 — Pipeline Statistique Classique

> Référence : *Cours Séries Temporelles* — Pr. Tawfik Masrour, ENSAM-IATD (2024-2025)

## Étape 1 — Visualisation

Visualisation des 5 variables sur la période complète pour détecter les tendances, saisonnalités et anomalies.

```python
fig, axes = plt.subplots(3, 1, figsize=(16, 10))

axes[0].plot(df[TARGET], color='steelblue', lw=1.5)
axes[0].set_title('Eau Vaporisée Bassin (g/h)')

axes[1].plot(df['Consommation_HVAC_kWh'], color='darkorange', label='HVAC (kWh)')
axes[1].set_title('Consommation HVAC & Température')

axes[2].plot(df['Humidite_Relative_%'], color='forestgreen', label='Humidité (%)')
axes[2].set_title('Humidité & CO2')
```

---

## Étape 2 — Tests de Stationnarité

### ADF (Augmented Dickey-Fuller)

- **H₀** : La série possède une racine unitaire (non-stationnaire)
- **Décision** : Si p-value < 0.05 → on rejette H₀ → **série stationnaire**

### KPSS

- **H₀** : La série est stationnaire
- **Décision** : Si p-value > 0.05 → on ne rejette pas H₀ → **série stationnaire**

```python
from statsmodels.tsa.stattools import adfuller, kpss

adf = adfuller(df[TARGET], autolag='AIC')
print(f"ADF : stat={adf[0]:.4f}  p={adf[1]:.6f}")

kp_stat, kp_p, _, _ = kpss(df[TARGET], regression='c')
print(f"KPSS: stat={kp_stat:.4f}  p={kp_p:.4f}")

d_order = 0 if adf[1] < 0.05 else 1
print(f"Ordre différenciation retenu : d = {d_order}")
```

---

## Étape 3 — Décomposition STL

Décomposition additive : **y_t = T_t + S_t + R_t**

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(df[TARGET], model='additive',
                             period=24, extrapolate_trend='freq')
```

| Composante | Signification physique |
|------------|----------------------|
| **T_t** — Tendance | Évolution de la charge thermique globale du DC |
| **S_t** — Saisonnalité (T=24h) | Cycles jour/nuit d'activité des serveurs |
| **R_t** — Résidu | Fluctuations aléatoires, incidents ponctuels |

---

## Étape 4 — ACF / PACF

Identification des ordres **p** (PACF) et **q** (ACF) pour les modèles ARIMA.

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, axes = plt.subplots(2, 2, figsize=(18, 9))
plot_acf (serie_stat, lags=48, ax=axes[0,0])
plot_pacf(serie_stat, lags=48, ax=axes[0,1], method='ywm')
plot_acf (serie,      lags=48, ax=axes[1,0])
plot_pacf(serie,      lags=48, ax=axes[1,1], method='ywm')
```

!!! tip "Lecture des graphiques"
    - **ACF** coupe après le lag q → ordre MA = q
    - **PACF** coupe après le lag p → ordre AR = p
    - Pic significatif au **lag 24** → saisonnalité de période S=24h

---

## Étape 5 — Modèles Statistiques

### ARIMA(p, d, q)

```python
from statsmodels.tsa.arima.model import ARIMA

arima_fit = ARIMA(train_y, order=(p, d, q)).fit()
pred_arima = arima_fit.get_forecast(steps=TEST_SIZE).predicted_mean
```

### SARIMAX(p,d,q)(P,D,Q,S) avec variables exogènes

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarimax_fit = SARIMAX(
    train_y, exog=train_exog,
    order=(1, d, 1),
    seasonal_order=(1, 0, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)
```

### Holt-Winters (Lissage Exponentiel Triple)

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

hw_fit = ExponentialSmoothing(
    train_y, trend='add', seasonal='add', seasonal_periods=24
).fit(optimized=True)
```

---

## Étape 6 — Analyse des Résidus

Six diagnostics visuels et statistiques :

| Diagnostic | Test | Attendu |
|------------|------|---------|
| Résidus dans le temps | Visuel | Centré autour de 0 |
| Distribution | Histogramme + loi normale | Forme gaussienne |
| Normalité | Q-Q plot | Points sur la droite |
| Autocorrélation | ACF résidus | Valeurs ≈ 0 |
| Homoscédasticité | Résidus vs Prédits | Dispersion uniforme |
| Test de Ljung-Box | p-value | p > 0.05 → bruit blanc |

---

## Étape 7 — Prévision avec Intervalles de Confiance

```python
fc   = sarimax_fit.get_forecast(steps=TEST_SIZE, exog=test_exog)
pm   = fc.predicted_mean
ci95 = fc.conf_int(alpha=0.05)   # IC 95%
ci80 = fc.conf_int(alpha=0.20)   # IC 80%

ax.fill_between(test_y.index,
                ci95.iloc[:,0], ci95.iloc[:,1],
                alpha=0.12, color='red', label='IC 95%')
ax.fill_between(test_y.index,
                ci80.iloc[:,0], ci80.iloc[:,1],
                alpha=0.25, color='red', label='IC 80%')
```
