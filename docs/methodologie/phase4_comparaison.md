# Phase 4 — Comparaison Finale

## Tableau Comparatif

```python
df_final = pd.DataFrame(RESULTS).T.reset_index()
df_final = df_final.rename(columns={'index': 'Modèle'})
df_final = df_final.sort_values('R²', ascending=False).reset_index(drop=True)
df_final.insert(0, 'Rang', range(1, len(df_final)+1))

print(df_final[['Rang','Phase','Modèle','R²','MAE','RMSE','MAPE','Qualité']].to_string())
```

### Structure du tableau de résultats

| Colonne | Type | Description |
|---------|------|-------------|
| `Rang` | int | Classement global par R² décroissant |
| `Phase` | str | Phase 1, 2 ou 3 |
| `Modèle` | str | Nom du modèle |
| `R²` | float | Coefficient de détermination |
| `MAE` | float | Erreur absolue moyenne (g/h) |
| `RMSE` | float | Racine de l'erreur quadratique moyenne (g/h) |
| `MAPE` | float | Erreur relative moyenne (%) |
| `Qualité` | str | ✅ Excellent / ✓ Bon / ⚠️ Acceptable / ❌ Faible |

---

## Visualisations

### 1. Barplot R² — Tous les modèles

```python
bars = ax.barh(df_final['Modèle'], df_final['R²'], color=model_col)
ax.axvline(0.50, color='orange', ls='--', label='R²=0.50')
ax.axvline(0.70, color='green',  ls='--', label='R²=0.70')
ax.axvline(0.85, color='blue',   ls='--', label='R²=0.85')
```

### 2. Courbes d'apprentissage DL

```python
for name, hist in HISTORIES.items():
    ax.plot(hist.history['val_loss'], lw=2, label=name)
ax.set_yscale('log')
ax.set_title("Courbes d'Apprentissage — Deep Learning")
```

### 3. Réel vs Prédit — 3 phases

Scatter plot pour les meilleurs modèles de chaque phase, avec la droite parfaite y=x.

### 4. Courbes superposées sur le jeu de test

```python
ax.plot(PREDS[best_p1][0][:min_len], color='black',    label='Réel')
ax.plot(PREDS[best_p1][1][:min_len], color='#4e79a7', label=best_p1)
ax.plot(PREDS[best_p2][1][:min_len], color='#59a14f', label=best_p2)
ax.plot(PREDS[best_p3][1][:min_len], color='#e15759', label=best_p3)
```

---

## Export des Résultats

```python
with pd.ExcelWriter('resultats_complets_hvac.xlsx', engine='openpyxl') as w:
    df_final.to_excel(w, sheet_name='Comparaison_Finale', index=False)
    pd.DataFrame({
        'Réel': list(PREDS.values())[0][0],
        **{f'Pred_{k}': v[1] for k, v in PREDS.items()}
    }).to_excel(w, sheet_name='Predictions', index=False)
```
