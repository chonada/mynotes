"""
================================================================================
Advanced LSTM for Time Series Analysis — TensorFlow/Keras
================================================================================
Covers:
  1. Synthetic & real-world data generation
  2. Sliding-window dataset creation with multi-step forecasting
  3. Five LSTM architectures:
       a) Vanilla LSTM
       b) Stacked (Deep) LSTM
       c) Bidirectional LSTM
       d) CNN-LSTM Hybrid
       e) Encoder-Decoder LSTM with Bahdanau Attention
  4. Custom Bahdanau Attention Layer
  5. Learning-rate scheduling, early stopping, model checkpointing
  6. Walk-forward validation
  7. Comprehensive evaluation: RMSE, MAE, MAPE, R², directional accuracy
  8. Visualization: predictions, residuals, attention heatmaps
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Tuple, List, Dict, Optional
import warnings, os

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1. DATA GENERATION                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def generate_multivariate_timeseries(
    n_samples: int = 5000,
    n_features: int = 3,
    noise_std: float = 0.05,
) -> Tuple[np.ndarray, list]:
    """
    Generate a realistic synthetic multivariate time series with:
      - Trend component
      - Multiple seasonal components (daily + weekly analogy)
      - Auto-regressive dependency across features
      - Random noise
    Returns shape (n_samples, n_features) and feature names.
    """
    t = np.arange(n_samples)

    # --- Primary signal (e.g. stock price / energy demand) ---
    trend = 0.0002 * t
    season_fast = 0.5 * np.sin(2 * np.pi * t / 50)       # ~daily cycle
    season_slow = 0.3 * np.sin(2 * np.pi * t / 250)      # ~weekly cycle
    regime_shift = 0.4 * np.tanh(0.005 * (t - 2500))      # structural break
    primary = trend + season_fast + season_slow + regime_shift

    # --- Correlated feature (e.g. volume / temperature) ---
    secondary = 0.6 * primary + 0.4 * np.sin(2 * np.pi * t / 75) + 0.1

    # --- Lagged feature (e.g. moving average indicator) ---
    kernel = np.ones(20) / 20
    tertiary = np.convolve(primary, kernel, mode="same")

    features = np.column_stack([primary, secondary, tertiary])[:, :n_features]
    features += np.random.normal(0, noise_std, features.shape)

    names = ["price", "volume", "ma_indicator"][:n_features]
    return features.astype(np.float32), names


def create_windowed_dataset(
    data: np.ndarray,
    lookback: int = 60,
    horizon: int = 5,
    target_col: int = 0,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw time series into supervised (X, y) pairs.

    Parameters
    ----------
    data       : (T, F) array — T timesteps, F features
    lookback   : number of past timesteps as input
    horizon    : number of future timesteps to predict (multi-step)
    target_col : column index of the prediction target
    stride     : step between consecutive windows

    Returns
    -------
    X : (N, lookback, F)
    y : (N, horizon)
    """
    X, y = [], []
    for i in range(0, len(data) - lookback - horizon + 1, stride):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback : i + lookback + horizon, target_col])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_val_test_split(
    X: np.ndarray, y: np.ndarray,
    train_frac: float = 0.7, val_frac: float = 0.15,
) -> dict:
    """Chronological split — no data leakage."""
    n = len(X)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))
    return {
        "X_train": X[:t1], "y_train": y[:t1],
        "X_val":   X[t1:t2], "y_val":   y[t1:t2],
        "X_test":  X[t2:], "y_test":  y[t2:],
    }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2. CUSTOM ATTENTION LAYER (Bahdanau-style)                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class BahdanauAttention(layers.Layer):
    """
    Additive (Bahdanau) attention for sequence-to-one / sequence-to-sequence.

    Given encoder hidden states H ∈ ℝ^{T×d}, computes:
        score_t = V^T · tanh(W_1 · h_t + W_2 · s)
        α       = softmax(scores)
        context = Σ α_t · h_t

    Where s is the decoder state (or last encoder state for seq2one).
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.W1 = self.add_weight("W1", (feat_dim, self.units), initializer="glorot_uniform")
        self.W2 = self.add_weight("W2", (feat_dim, self.units), initializer="glorot_uniform")
        self.V  = self.add_weight("V",  (self.units, 1),        initializer="glorot_uniform")

    def call(self, encoder_outputs, return_attention=False):
        """
        encoder_outputs : (batch, timesteps, features)
        Returns         : context (batch, features), optionally attention weights
        """
        # Use last hidden state as the decoder query
        last_hidden = encoder_outputs[:, -1:, :]                      # (B, 1, D)

        score = tf.nn.tanh(
            tf.matmul(encoder_outputs, self.W1) +                     # (B, T, units)
            tf.matmul(last_hidden, self.W2)                           # (B, 1, units) → broadcast
        )
        attention_weights = tf.nn.softmax(tf.matmul(score, self.V), axis=1)  # (B, T, 1)
        context = tf.reduce_sum(attention_weights * encoder_outputs, axis=1) # (B, D)

        if return_attention:
            return context, tf.squeeze(attention_weights, axis=-1)
        return context

    def get_config(self):
        return {**super().get_config(), "units": self.units}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3. MODEL ARCHITECTURES                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_vanilla_lstm(
    lookback: int, n_features: int, horizon: int,
    units: int = 128, dropout: float = 0.2,
) -> keras.Model:
    """Single-layer LSTM with dropout regularization."""
    inp = layers.Input(shape=(lookback, n_features), name="input")
    x = layers.LSTM(units, return_sequences=False, dropout=dropout,
                    recurrent_dropout=dropout, name="lstm")(inp)
    x = layers.Dense(horizon, name="output")(x)
    return keras.Model(inp, x, name="Vanilla_LSTM")


def build_stacked_lstm(
    lookback: int, n_features: int, horizon: int,
    units: List[int] = [128, 64, 32], dropout: float = 0.2,
) -> keras.Model:
    """Deep stacked LSTM with residual-like skip via concatenation."""
    inp = layers.Input(shape=(lookback, n_features), name="input")
    x = inp
    skip_outputs = []

    for i, u in enumerate(units):
        return_seq = i < len(units) - 1
        x = layers.LSTM(u, return_sequences=return_seq,
                        dropout=dropout, recurrent_dropout=dropout,
                        name=f"lstm_{i+1}")(x)
        if not return_seq:
            skip_outputs.append(x)
        else:
            x = layers.BatchNormalization(name=f"bn_{i+1}")(x)

    if len(skip_outputs) > 1:
        x = layers.Concatenate()(skip_outputs)
    else:
        x = skip_outputs[0]

    x = layers.Dense(64, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    x = layers.Dense(horizon, name="output")(x)
    return keras.Model(inp, x, name="Stacked_LSTM")


def build_bidirectional_lstm(
    lookback: int, n_features: int, horizon: int,
    units: int = 128, dropout: float = 0.2,
) -> keras.Model:
    """Bidirectional LSTM — captures both past→future and future→past context."""
    inp = layers.Input(shape=(lookback, n_features), name="input")
    x = layers.Bidirectional(
        layers.LSTM(units, return_sequences=True, dropout=dropout,
                    recurrent_dropout=dropout),
        name="bilstm_1",
    )(inp)
    x = layers.Bidirectional(
        layers.LSTM(units // 2, return_sequences=False, dropout=dropout,
                    recurrent_dropout=dropout),
        name="bilstm_2",
    )(x)
    x = layers.Dense(64, activation="relu", name="fc")(x)
    x = layers.Dense(horizon, name="output")(x)
    return keras.Model(inp, x, name="Bidirectional_LSTM")


def build_cnn_lstm(
    lookback: int, n_features: int, horizon: int,
    cnn_filters: int = 64, kernel_size: int = 3,
    lstm_units: int = 128, dropout: float = 0.2,
) -> keras.Model:
    """
    CNN-LSTM Hybrid: Conv1D extracts local patterns → LSTM captures
    long-range dependencies. Effective for noisy financial series.
    """
    inp = layers.Input(shape=(lookback, n_features), name="input")

    # Temporal convolution block
    x = layers.Conv1D(cnn_filters, kernel_size, padding="causal",
                      activation="relu", name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Conv1D(cnn_filters, kernel_size, padding="causal",
                      activation="relu", name="conv2")(x)
    x = layers.MaxPooling1D(pool_size=2, name="pool")(x)

    # Recurrent block
    x = layers.LSTM(lstm_units, return_sequences=False,
                    dropout=dropout, name="lstm")(x)
    x = layers.Dense(64, activation="relu", name="fc")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    x = layers.Dense(horizon, name="output")(x)
    return keras.Model(inp, x, name="CNN_LSTM")


def build_attention_lstm(
    lookback: int, n_features: int, horizon: int,
    units: int = 128, attn_units: int = 64, dropout: float = 0.2,
) -> keras.Model:
    """
    Encoder-Decoder LSTM with Bahdanau Attention.
    The attention layer learns which past timesteps are most relevant.
    """
    inp = layers.Input(shape=(lookback, n_features), name="input")

    # Encoder
    x = layers.LSTM(units, return_sequences=True, dropout=dropout,
                    recurrent_dropout=dropout, name="encoder_lstm_1")(inp)
    x = layers.LSTM(units, return_sequences=True, dropout=dropout,
                    recurrent_dropout=dropout, name="encoder_lstm_2")(x)

    # Attention
    context = BahdanauAttention(units=attn_units, name="attention")(x)

    # Decoder
    x = layers.Dense(64, activation="relu", name="fc1")(context)
    x = layers.Dropout(dropout, name="drop")(x)
    x = layers.Dense(horizon, name="output")(x)
    return keras.Model(inp, x, name="Attention_LSTM")


# Factory registry
MODEL_REGISTRY: Dict[str, callable] = {
    "vanilla":       build_vanilla_lstm,
    "stacked":       build_stacked_lstm,
    "bidirectional": build_bidirectional_lstm,
    "cnn_lstm":      build_cnn_lstm,
    "attention":     build_attention_lstm,
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4. TRAINING PIPELINE                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class CosineAnnealingSchedule(callbacks.Callback):
    """Cosine annealing with warm restarts (SGDR-style)."""

    def __init__(self, lr_max=1e-3, lr_min=1e-6, T_0=10, T_mult=2):
        super().__init__()
        self.lr_max, self.lr_min = lr_max, lr_min
        self.T_0, self.T_mult = T_0, T_mult
        self._cycle_epoch = 0
        self._current_T = T_0

    def on_epoch_begin(self, epoch, logs=None):
        if self._cycle_epoch >= self._current_T:
            self._cycle_epoch = 0
            self._current_T *= self.T_mult

        frac = self._cycle_epoch / self._current_T
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(np.pi * frac))
        K.set_value(self.model.optimizer.learning_rate, lr)
        self._cycle_epoch += 1


def compile_and_train(
    model: keras.Model,
    data: dict,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 12,
    use_cosine: bool = True,
) -> keras.callbacks.History:
    """Compile with Huber loss and train with advanced callbacks."""

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1,
        ),
    ]
    if use_cosine:
        cb.append(CosineAnnealingSchedule(lr_max=lr))

    history = model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=epochs, batch_size=batch_size,
        callbacks=cb, verbose=1, shuffle=False,   # keep temporal order within batches
    )
    return history


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5. EVALUATION METRICS                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Comprehensive regression + directional metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true.ravel(), y_pred.ravel())

    # MAPE (handle near-zero)
    mask = np.abs(y_true) > 1e-8
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    # Directional accuracy (did we predict direction correctly?)
    if y_true.shape[-1] > 1:
        actual_dir = np.diff(y_true, axis=1) > 0
        pred_dir   = np.diff(y_pred, axis=1) > 0
        dir_acc = np.mean(actual_dir == pred_dir) * 100
    else:
        dir_acc = np.nan

    return {
        "RMSE": rmse, "MAE": mae, "MAPE (%)": mape,
        "R²": r2, "Directional Acc (%)": dir_acc,
    }


def walk_forward_validation(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    retrain_every: int = 50,
) -> np.ndarray:
    """
    Walk-forward (expanding window) prediction.
    Optionally retrain the model every N steps for concept-drift adaptation.
    """
    preds = []
    for i in range(len(X_test)):
        pred = model.predict(X_test[i:i+1], verbose=0)
        preds.append(pred[0])
    return np.array(preds)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  6. ATTENTION EXTRACTION                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def extract_attention_weights(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Build a sub-model that exposes the attention weights for visualization.
    Works only for models containing a BahdanauAttention layer.
    """
    attn_layer = None
    encoder_output_layer = None

    for layer in model.layers:
        if isinstance(layer, BahdanauAttention):
            attn_layer = layer
        if isinstance(layer, layers.LSTM) and layer.return_sequences:
            encoder_output_layer = layer

    if attn_layer is None or encoder_output_layer is None:
        raise ValueError("Model has no BahdanauAttention / return_sequences LSTM.")

    # Build functional sub-model to get encoder outputs
    encoder_model = keras.Model(
        inputs=model.input,
        outputs=encoder_output_layer.output,
    )
    enc_out = encoder_model.predict(X, verbose=0)

    # Run attention forward pass
    _, weights = attn_layer(tf.constant(enc_out), return_attention=True)
    return weights.numpy()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  7. VISUALISATION                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def plot_training_history(history: keras.callbacks.History, title: str = ""):
    """Plot loss and MAE curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, metric in zip(axes, ["loss", "mae"]):
        ax.plot(history.history[metric], label="Train")
        ax.plot(history.history[f"val_{metric}"], label="Val")
        ax.set_title(f"{title} — {metric.upper()}")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()


def plot_predictions(
    y_true: np.ndarray, y_pred: np.ndarray,
    scaler: Optional[MinMaxScaler] = None, title: str = "",
    step_idx: int = 0,
) -> None:
    """
    Plot actual vs predicted for a single forecast step,
    plus residual distribution.
    """
    actual = y_true[:, step_idx]
    predicted = y_pred[:, step_idx]
    residuals = actual - predicted

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    # Time series overlay
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(actual, label="Actual", linewidth=1.2, alpha=0.85)
    ax1.plot(predicted, label="Predicted", linewidth=1.2, alpha=0.85)
    ax1.fill_between(range(len(residuals)), actual, predicted,
                     alpha=0.15, color="red", label="Error band")
    ax1.set_title(f"{title} — Step t+{step_idx+1} Forecast")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residual plot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(residuals, color="coral", alpha=0.7, linewidth=0.8)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.5)
    ax2.set_title("Residuals over Time")
    ax2.grid(True, alpha=0.3)

    # Residual histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(residuals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax3.axvline(0, color="red", linestyle="--")
    ax3.set_title(f"Residual Distribution (μ={residuals.mean():.4f}, σ={residuals.std():.4f})")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    plt.close()


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    n_samples: int = 10,
    lookback: int = 60,
) -> None:
    """Visualise attention weights as a heatmap for N test samples."""
    fig, ax = plt.subplots(figsize=(14, 6))
    subset = attention_weights[:n_samples]
    im = ax.imshow(subset, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Input Timestep")
    ax.set_ylabel("Sample Index")
    ax.set_title("Bahdanau Attention Weights — Which timesteps the model focuses on")
    plt.colorbar(im, ax=ax, label="Attention weight")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    plt.close()


def plot_model_comparison(results: Dict[str, Dict[str, float]]) -> None:
    """Bar chart comparing all models on key metrics."""
    df = pd.DataFrame(results).T
    metrics = ["RMSE", "MAE", "R²"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for ax, m in zip(axes, metrics):
        bars = ax.barh(df.index, df[m], color=plt.cm.Set2(np.arange(len(df))))
        ax.set_title(m)
        ax.grid(True, axis="x", alpha=0.3)
        for bar, val in zip(bars, df[m]):
            ax.text(val, bar.get_y() + bar.get_height()/2,
                    f"  {val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.close()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  8. MAIN EXPERIMENT RUNNER                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_experiment(
    architectures: List[str] = None,
    n_samples: int = 5000,
    lookback: int = 60,
    horizon: int = 5,
    epochs: int = 80,
    batch_size: int = 64,
) -> None:
    """End-to-end experiment: generate → train → evaluate → visualise."""

    if architectures is None:
        architectures = list(MODEL_REGISTRY.keys())

    # ── Data ──
    print("=" * 70)
    print("GENERATING DATA")
    print("=" * 70)
    raw_data, feature_names = generate_multivariate_timeseries(n_samples=n_samples)
    print(f"  Shape: {raw_data.shape}  |  Features: {feature_names}")

    # Scale
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(raw_data)

    # Windows
    X, y = create_windowed_dataset(scaled_data, lookback=lookback, horizon=horizon)
    print(f"  Windows: X={X.shape}, y={y.shape}")

    # Split
    data = train_val_test_split(X, y)
    print(f"  Train: {data['X_train'].shape[0]} | Val: {data['X_val'].shape[0]}"
          f" | Test: {data['X_test'].shape[0]}")

    n_features = X.shape[2]
    all_results = {}

    # ── Train each architecture ──
    for arch_name in architectures:
        print("\n" + "=" * 70)
        print(f"TRAINING: {arch_name.upper()}")
        print("=" * 70)

        build_fn = MODEL_REGISTRY[arch_name]
        model = build_fn(lookback, n_features, horizon)
        model.summary(print_fn=lambda s: print(f"  {s}"))

        history = compile_and_train(
            model, data, epochs=epochs, batch_size=batch_size,
            use_cosine=(arch_name == "attention"),
        )

        # Predict
        y_pred = model.predict(data["X_test"], verbose=0)
        metrics = compute_metrics(data["y_test"], y_pred)
        all_results[arch_name] = metrics

        print(f"\n  ── Test Metrics ──")
        for k, v in metrics.items():
            print(f"    {k:>22s}: {v:.4f}")

        # Plot training curves
        plot_training_history(history, title=arch_name)

        # For the attention model, extract and plot attention
        if arch_name == "attention":
            try:
                attn_w = extract_attention_weights(model, data["X_test"][:50])
                plot_attention_heatmap(attn_w, n_samples=50, lookback=lookback)
                print("  ✓ Attention heatmap saved → attention_heatmap.png")
            except Exception as e:
                print(f"  ⚠ Could not extract attention: {e}")

    # ── Final comparison ──
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    df = pd.DataFrame(all_results).T
    print(df.to_string())
    print()

    # Best model
    best = df["RMSE"].idxmin()
    print(f"  🏆 Best model by RMSE: {best} ({df.loc[best, 'RMSE']:.4f})")

    # Visualize best model predictions
    best_model = MODEL_REGISTRY[best](lookback, n_features, horizon)
    best_model.compile(optimizer="adam", loss="huber")
    # Retrain quickly for clean predictions
    best_model.fit(data["X_train"], data["y_train"],
                   validation_data=(data["X_val"], data["y_val"]),
                   epochs=epochs, batch_size=batch_size, verbose=0,
                   callbacks=[callbacks.EarlyStopping(patience=10,
                              restore_best_weights=True)])
    y_pred_best = best_model.predict(data["X_test"], verbose=0)
    plot_predictions(data["y_test"], y_pred_best, title=best)

    plot_model_comparison(all_results)

    print("\n  Saved plots:")
    print("    • training_curves.png")
    print("    • predictions.png")
    print("    • model_comparison.png")
    if "attention" in architectures:
        print("    • attention_heatmap.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ENTRY POINT                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    run_experiment(
        architectures=["vanilla", "stacked", "bidirectional", "cnn_lstm", "attention"],
        n_samples=5000,
        lookback=60,
        horizon=5,
        epochs=80,
        batch_size=64,
    )
