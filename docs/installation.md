# ⚙️ Installation

## Prérequis

| Outil | Version minimale |
|-------|-----------------|
| Python | 3.10+ |
| pip | 23.0+ |
| RAM | 8 GB recommandé |
| GPU | Optionnel (accélère Phase 2) |

---

## Installation des dépendances

```bash
# Cloner le dépôt
git clone https://github.com/yasserkhadiri/hvac-datacenter-prediction.git
cd hvac-datacenter-prediction

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### `requirements.txt`

```text
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
scipy>=1.10.0
scikit-learn>=1.3.0
xgboost>=2.0.0
tensorflow>=2.13.0
openpyxl>=3.1.0
```

---

## Structure du Projet

```
hvac-datacenter-prediction/
│
├── 📓 HVAC_KHADIRI_YASSER_PROJET.ipynb   ← Notebook principal
├── 📊 HVAC_Vaporisation_CLEAN.xlsx        ← Dataset (3 mois)
├── 📋 requirements.txt
├── 📖 README.md
│
├── outputs/                               ← Graphiques générés
│   ├── etape1_visualisation.png
│   ├── etape3_decomposition.png
│   ├── etape4_acf_pacf.png
│   ├── etape6_residus.png
│   ├── etape7_prevision_ic.png
│   └── comparaison_finale.png
│
└── docs/                                  ← Documentation ReadTheDocs
    ├── index.md
    ├── introduction.md
    ├── dataset.md
    ├── installation.md
    ├── resultats.md
    ├── conclusion.md
    └── methodologie/
        ├── overview.md
        ├── phase1_statistique.md
        ├── phase2_deeplearning.md
        ├── phase3_xgboost.md
        └── phase4_comparaison.md
```

---

## Lancer le Notebook

```bash
jupyter notebook HVAC_KHADIRI_YASSER_PROJET.ipynb
```

### Vérification de l'environnement

```python
import tensorflow as tf
import xgboost as xgb

gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow {tf.__version__} | GPU: {gpus[0].name if gpus else 'CPU'}")
print(f"XGBoost {xgb.__version__}")
```

!!! warning "GPU non obligatoire"
    Les modèles Deep Learning (Phase 2) peuvent tourner sur CPU mais seront plus lents.  
    Temps estimé sur CPU : ~15-30 min pour les 6 modèles DL.
