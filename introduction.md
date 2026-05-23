# 📋 Introduction

## Contexte du Projet

### Le Data Center et son système de refroidissement

Un **data center** est une infrastructure critique qui héberge des milliers de serveurs fonctionnant en continu. Ces serveurs génèrent une chaleur importante qui doit être impérativement évacuée pour garantir leur fiabilité et leurs performances.

Le système de refroidissement repose sur une **tour de refroidissement évaporative** composée de :

- Un **circuit intérieur** : eau glacée circulant entre les serveurs et l'échangeur
- Un **circuit extérieur** : eau chaude envoyée vers la tour de refroidissement
- Un **bassin** : réservoir d'eau où se produit l'évaporation naturelle

```
Serveurs (chaleur) → Échangeur → Tour de refroidissement
                                         ↓
                                    Bassin d'eau
                                         ↓
                              Évaporation (g/h) ← VARIABLE CIBLE
```

### Pourquoi prédire la vaporisation ?

La quantité d'eau vaporisée est un **indicateur clé** pour :

!!! info "Efficacité énergétique"
    Anticiper la consommation en eau permet d'optimiser le réapprovisionnement du bassin et de réduire le gaspillage.

!!! warning "Maintenance préventive"
    Une vaporisation anormalement élevée ou basse peut signaler un dysfonctionnement de la tour (encrassement, panne de ventilateur, fuite).

!!! tip "Planification opérationnelle"
    Prévoir la vaporisation à h+6, h+12 et h+24 permet aux opérateurs d'anticiper les besoins en eau de compensation (make-up water).

---

## Problématique

> **Dans un système de refroidissement de data center, peut-on prédire avec précision la quantité d'eau vaporisée horaire depuis le bassin de la tour de refroidissement, à partir de capteurs environnementaux (température, humidité, CO₂, consommation énergétique), et quelle famille de modèles — statistique classique, deep learning ou gradient boosting — offre le meilleur compromis entre précision et interprétabilité pour un usage opérationnel ?**

---

## Objectifs

### Objectif Principal
Construire un modèle de prédiction fiable de `Eau_Vaporisee_g_par_heure` à horizon **h+1** avec un coefficient de détermination R² ≥ 0.70.

### Objectifs Secondaires

1. **Comparer** les performances des approches statistiques, deep learning et ML
2. **Identifier** les variables exogènes les plus influentes (feature importance XGBoost)
3. **Valider** la stationnarité et la structure temporelle des données (ADF, KPSS, ACF/PACF)
4. **Analyser** les résidus pour vérifier la qualité des modèles

---

## État de l'Art

La prédiction de séries temporelles dans les systèmes HVAC de data centers est un domaine actif de recherche. Plusieurs approches ont été proposées :

| Approche | Auteurs | Performance |
|----------|---------|-------------|
| LSTM pour température salle serveurs | Kim et al. (2022) | R²=0.89 |
| CNN-LSTM consommation énergie DC | Zhang et al. (2023) | RMSE=12.3 kW |
| XGBoost charge thermique | Liu et al. (2021) | MAE=8.1 kW |
| Bi-LSTM efficacité refroidissement | Chen et al. (2023) | R²=0.92 |

Ce projet se distingue en ciblant spécifiquement **l'évaporation du bassin** — une variable rarement modélisée directement dans la littérature.
