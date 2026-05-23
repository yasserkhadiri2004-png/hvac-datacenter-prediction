# Phase 2 — Deep Learning

## Preprocessing

Normalisation MinMaxScaler sur toutes les variables + construction de séquences glissantes.

```python
from sklearn.preprocessing import MinMaxScaler

scaler_X    = MinMaxScaler()
scaler_y_dl = MinMaxScaler()

data_scaled = scaler_X.fit_transform(df[ALL_VARS].values).astype(np.float32)

def build_seqs(data, target_idx, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i, :])   # fenêtre de 24h
        y.append(data[i, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_all, y_all = build_seqs(data_scaled, target_idx=0, seq_len=24)
# Split 70 / 15 / 15
```

---

## Modèles Implémentés

### 🔵 ANN — Dense Feed-Forward

Réseau de neurones entièrement connecté. Sert de **baseline** pour les modèles récurrents.

```python
inp = Input(shape=(SEQ_LEN, N_FEAT))
x   = Flatten()(inp)
for units in [256, 128, 64]:
    x = Dense(units, activation='relu',
              kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
ann = Model(inp, Dense(1)(x), name='ANN')
```

---

### 🟢 LSTM — Long Short-Term Memory

Capture les dépendances **à long terme** grâce aux portes d'oubli, d'entrée et de sortie.

```python
lstm_m = Sequential([
    Input(shape=(SEQ_LEN, N_FEAT)),
    LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)),
    BatchNormalization(), Dropout(0.3),
    LSTM(64,  return_sequences=False, kernel_regularizer=l2(1e-4)),
    BatchNormalization(), Dropout(0.3),
    Dense(32, activation='relu'), Dense(1)
], name='LSTM')
```

---

### 🔴 Bi-LSTM — Bidirectionnel

Traite la séquence dans les **deux sens** (passé → futur et futur → passé).

```python
bilstm = Sequential([
    Input(shape=(SEQ_LEN, N_FEAT)),
    Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-4))),
    BatchNormalization(), Dropout(0.25),
    Bidirectional(LSTM(32, kernel_regularizer=l2(1e-4))),
    BatchNormalization(), Dropout(0.25),
    Dense(16, activation='relu'), Dense(1)
], name='BiLSTM')
```

---

### 🟡 GRU — Gated Recurrent Unit

Architecture simplifiée du LSTM avec **moins de paramètres**, souvent aussi performante.

```python
gru_m = Sequential([
    Input(shape=(SEQ_LEN, N_FEAT)),
    GRU(128, return_sequences=True, kernel_regularizer=l2(1e-4)),
    BatchNormalization(), Dropout(0.3),
    GRU(64,  return_sequences=False, kernel_regularizer=l2(1e-4)),
    BatchNormalization(), Dropout(0.3),
    Dense(32, activation='relu'), Dense(1)
], name='GRU')
```

---

### 🟣 CNN-LSTM — Extraction locale + Mémoire

Les couches **Conv1D** extraient des motifs locaux (ex : pic horaire), le LSTM gère la mémoire séquentielle.

```python
cnn_lstm = Sequential([
    Input(shape=(SEQ_LEN, N_FEAT)),
    Conv1D(64,  3, activation='relu', padding='causal'),
    BatchNormalization(),
    Conv1D(128, 3, activation='relu', padding='causal'),
    BatchNormalization(),
    MaxPooling1D(2), Dropout(0.2),
    LSTM(64, kernel_regularizer=l2(1e-4)),
    BatchNormalization(), Dropout(0.3),
    Dense(32, activation='relu'), Dense(1)
], name='CNN_LSTM')
```

---

### ⭐ Attention-LSTM — Multi-Head Self-Attention

Mécanisme d'**attention** permettant au modèle de pondérer les pas de temps les plus importants.

```python
seq_in = Input(shape=(SEQ_LEN, N_FEAT))
x      = LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4))(seq_in)
x      = BatchNormalization()(x); x = Dropout(0.2)(x)

attn   = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
x      = LayerNormalization()(Add()([x, attn]))
pooled = GlobalAveragePooling1D()(x)

x      = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(pooled)
x      = BatchNormalization()(x); x = Dropout(0.3)(x)
attn_lstm = Model(seq_in, Dense(1)(x), name='Attention_LSTM')
```

---

## Entraînement

```python
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=MeanSquaredError(),
    metrics=[MeanAbsoluteError()]
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    verbose=0
)
```

!!! warning "Callbacks désactivés"
    Les callbacks `EarlyStopping` et `ReduceLROnPlateau` sont définis dans le code mais désactivés (`callbacks=None`) pour assurer la reproductibilité sur 150 epochs fixes. Il est recommandé de les réactiver en production.
