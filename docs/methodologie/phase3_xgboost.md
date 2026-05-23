# Phase 3 — XGBoost avec Feature Engineering

## Feature Engineering

La richesse de XGBoost repose sur la création de **features temporelles** à partir des variables brutes.

```python
def build_features_xgb(df_in, target, exog_vars):
    df_f = df_in.copy()

    # 1. Lags de la variable cible
    for lag in [1, 2, 3, 6, 12, 24]:
        df_f[f'y_lag{lag}'] = df_f[target].shift(lag)

    # 2. Lags des variables exogènes
    for var in exog_vars:
        s = var.split('_')[0]
        df_f[f'{s}_lag1'] = df_f[var].shift(1)
        df_f[f'{s}_lag3'] = df_f[var].shift(3)

    # 3. Statistiques glissantes (rolling)
    for w in [3, 6, 12]:
        df_f[f'y_rmean{w}'] = df_f[target].shift(1).rolling(w).mean()
        df_f[f'y_rstd{w}']  = df_f[target].shift(1).rolling(w).std()

    # 4. Encodage cyclique de l'heure
    df_f['hour_sin'] = np.sin(2 * np.pi * df_f.index.hour / 24)
    df_f['hour_cos'] = np.cos(2 * np.pi * df_f.index.hour / 24)
    df_f['dow_sin']  = np.sin(2 * np.pi * df_f.index.dayofweek / 7)
    df_f['dow_cos']  = np.cos(2 * np.pi * df_f.index.dayofweek / 7)

    # 5. Interactions physiques
    df_f['HVAC_x_T'] = df_f['Consommation_HVAC_kWh'] * df_f['Temperature_C']
    df_f['H_x_T']    = df_f['Humidite_Relative_%']    * df_f['Temperature_C']
    df_f['dHVAC']    = df_f['Consommation_HVAC_kWh'].diff(1)   # dérivée

    return df_f.dropna()
```

### Récapitulatif des features créées

| Catégorie | Features | Nb. |
|-----------|----------|-----|
| Lags cible | `y_lag1` → `y_lag24` | 6 |
| Lags exogènes | `{var}_lag1`, `{var}_lag3` | 8 |
| Rolling stats | `y_rmean3/6/12`, `y_rstd3/6/12` | 6 |
| Encodage cyclique | `hour_sin/cos`, `dow_sin/cos` | 4 |
| Interactions | `HVAC_x_T`, `H_x_T`, `dHVAC` | 3 |
| **Total** | | **~27 features** |

---

## Optimisation par GridSearchCV

```python
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

param_grid = {
    'n_estimators'    : [300, 500],
    'max_depth'       : [4, 6, 8],
    'learning_rate'   : [0.03, 0.05, 0.10],
    'subsample'       : [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
}

tscv = TimeSeriesSplit(n_splits=5)
gs = GridSearchCV(
    xgb.XGBRegressor(random_state=42, verbosity=0),
    param_grid,
    scoring='r2',
    cv=tscv,
    n_jobs=-1
)
gs.fit(Xtr_xgb, ytr_xgb)
print(f"Meilleurs paramètres : {gs.best_params_}")
print(f"R² CV moyen          : {gs.best_score_:.4f}")
```

!!! note "TimeSeriesSplit"
    Contrairement au `KFold` classique, `TimeSeriesSplit` respecte l'ordre chronologique — essentiel pour éviter le data leakage en séries temporelles.
