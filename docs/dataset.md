# 📊 Dataset

## Source et Collecte

Les données ont été collectées sur un **data center réel** via un système de capteurs IoT déployé sur l'infrastructure de refroidissement.

| Attribut | Valeur |
|----------|--------|
| **Période** | 3 mois (données réelles) |
| **Résolution** | 10 minutes |
| **Fréquence de reéchantillonnage** | 1 heure (agrégation `mean`) |
| **Observations (après resampling)** | ~2 160 points |
| **Format source** | Excel (`.xlsx`) |
| **Valeurs manquantes** | 0 (données nettoyées) |

---

## Variables

### Variable Cible

| Variable | Type | Unité | Description |
|----------|------|-------|-------------|
| `Eau_Vaporisee_g_par_heure` | float | g/h | Quantité d'eau vaporisée depuis le bassin de la tour de refroidissement |

### Variables Exogènes

| Variable | Type | Unité | Description |
|----------|------|-------|-------------|
| `Consommation_HVAC_kWh` | float | kWh | Énergie électrique consommée par le système HVAC du data center |
| `Temperature_C` | float | °C | Température ambiante dans la salle serveurs |
| `Humidite_Relative_%` | float | % | Humidité relative de l'air autour de la tour |
| `CO2_ppm` | float | ppm | Concentration en CO₂ (indicateur de charge et ventilation) |

---

## Statistiques Descriptives

```python
TARGET   = 'Eau_Vaporisee_g_par_heure'
EXOG     = ['Consommation_HVAC_kWh', 'Temperature_C',
            'Humidite_Relative_%', 'CO2_ppm']
SEQ_LEN  = 24   # fenêtre temporelle = 24 heures
```

### Chargement des données

```python
import pandas as pd

df = pd.read_excel('HVAC_Vaporisation_CLEAN.xlsx')
df['Horodatage'] = pd.to_datetime(df['Horodatage'])
df.set_index('Horodatage', inplace=True)
df.sort_index(inplace=True)

# Reéchantillonnage horaire
df = df[ALL_VARS].resample('1h').mean().dropna()

print(f"Forme    : {df.shape}")
print(f"Période  : {df.index.min()} → {df.index.max()}")
print(df.describe().round(2))
```

---

## Structure Temporelle

### Décomposition STL

La série cible est décomposée selon le modèle additif :

```
y_t = T_t + S_t + R_t
```

Où :

- **T_t** — Tendance : évolution lente de la charge thermique du DC
- **S_t** — Saisonnalité : période de **24 heures** (cycle jour/nuit)
- **R_t** — Résidu : bruit stochastique

!!! note "Saisonnalité dominante"
    La composante saisonnière représente typiquement **40-60%** de la variance totale, reflétant les cycles d'activité du data center (heures de pointe vs heures creuses).

### Tests de Stationnarité

| Test | Hypothèse H₀ | Résultat attendu |
|------|-------------|-----------------|
| **ADF** (Augmented Dickey-Fuller) | Série non-stationnaire | p < 0.05 → stationnaire |
| **KPSS** | Série stationnaire | p > 0.05 → stationnaire |

```python
from statsmodels.tsa.stattools import adfuller, kpss

adf_stat, adf_p, _, _, _, _ = adfuller(df[TARGET], autolag='AIC')
kpss_stat, kpss_p, _, _     = kpss(df[TARGET], regression='c')

print(f"ADF  : stat={adf_stat:.4f}  p={adf_p:.6f}")
print(f"KPSS : stat={kpss_stat:.4f}  p={kpss_p:.4f}")
```

---

## Corrélations

La matrice de corrélation de Pearson révèle les relations entre variables :

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True)
plt.title('Corrélations — Variables HVAC Data Center')
plt.tight_layout()
```

!!! tip "Relations physiques attendues"
    - `Consommation_HVAC_kWh` ↔ `Eau_Vaporisee` : corrélation **forte positive** (plus d'énergie → plus de chaleur → plus d'évaporation)
    - `Temperature_C` ↔ `Eau_Vaporisee` : corrélation **positive** (températures élevées favorisent l'évaporation)
    - `Humidite_Relative_%` ↔ `Eau_Vaporisee` : corrélation **négative** (air saturé réduit l'évaporation)
