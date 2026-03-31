"""
train.py — Train & Evaluate the Passive vs. Active Predictive Models
=====================================================================

1. Generates the pendulum dataset via the simulator.
2. Splits into 80 % train / 20 % validation.
3. Trains both models with identical hyperparameters + early stopping.
4. Plots the validation-loss curves on a single figure and saves
   `loss_comparison.png`.

Expected outcome
----------------
Model A (Passive) plateaus at a **high** validation loss because it
cannot resolve the latent torque — it sees the same state map to many
different next-states depending on the (hidden) action.

Model B (APC) drives validation loss **toward zero** because the
physics are deterministic once (state, action) are both known.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless backend for CI / servers
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from simulator import generate_dataset
from models import build_passive_observer, build_active_predictor


# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
N_SAMPLES       = 50_000
VALIDATION_SPLIT = 0.2
BATCH_SIZE       = 256
MAX_EPOCHS       = 100
LEARNING_RATE    = 1e-3
PATIENCE         = 10          # early-stopping patience (epochs)
OUTPUT_DIR       = "."


def main():
    # ── 0. Apple Silicon: prefer Metal GPU if available ──────────────
    print("[train] TensorFlow version:", tf.__version__)
    print("[train] Physical devices:",
          tf.config.list_physical_devices())

    # ── 1. Generate data ─────────────────────────────────────────────
    states, actions, next_states = generate_dataset(N_SAMPLES)

    # ── 2. Prepare inputs ────────────────────────────────────────────
    #   Model A sees only the state.
    #   Model B sees state concatenated with the action (efference copy).
    x_passive = states                                        # (N, 2)
    x_active  = np.concatenate([states, actions], axis=1)     # (N, 3)
    y         = next_states                                   # (N, 2)

    # Train / validation split (deterministic)
    n_train = int(N_SAMPLES * (1 - VALIDATION_SPLIT))
    x_pass_train, x_pass_val = x_passive[:n_train], x_passive[n_train:]
    x_act_train,  x_act_val  = x_active[:n_train],  x_active[n_train:]
    y_train,      y_val      = y[:n_train],          y[n_train:]

    print(f"[train] Train samples : {n_train:,}")
    print(f"[train] Val   samples : {N_SAMPLES - n_train:,}")

    # ── 3. Build & compile models ────────────────────────────────────
    model_a = build_passive_observer()
    model_b = build_active_predictor()

    for m in (model_a, model_b):
        m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="mse",
        )
        m.summary()
        print()

    # ── 4. Train ─────────────────────────────────────────────────────
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    print("\n" + "=" * 60)
    print("  Training Model A  —  Passive Observer  (state only)")
    print("=" * 60)
    hist_a = model_a.fit(
        x_pass_train, y_train,
        validation_data=(x_pass_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    print("\n" + "=" * 60)
    print("  Training Model B  —  Active Predictor  (state + action)")
    print("=" * 60)
    hist_b = model_b.fit(
        x_act_train, y_train,
        validation_data=(x_act_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    # ── 5. Final evaluation ──────────────────────────────────────────
    loss_a = model_a.evaluate(x_pass_val, y_val, verbose=0)
    loss_b = model_b.evaluate(x_act_val,  y_val, verbose=0)
    print(f"\n[result] Model A  (Passive)  val MSE : {loss_a:.6f}")
    print(f"[result] Model B  (APC)      val MSE : {loss_b:.6f}")
    print(f"[result] Ratio A/B : {loss_a / max(loss_b, 1e-12):.1f}×")

    # ── 6. Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))

    epochs_a = range(1, len(hist_a.history["val_loss"]) + 1)
    epochs_b = range(1, len(hist_b.history["val_loss"]) + 1)

    ax.plot(epochs_a, hist_a.history["val_loss"],
            label=f"Model A — Passive Observer  (final {loss_a:.4f})",
            linewidth=2, color="#d62728")
    ax.plot(epochs_b, hist_b.history["val_loss"],
            label=f"Model B — Active Predictor / APC  (final {loss_b:.6f})",
            linewidth=2, color="#1f77b4")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation MSE Loss", fontsize=12)
    ax.set_title("Passive Observation vs. Active Predictive Coding\n"
                 "(1-Joint Motorised Pendulum)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="center right")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "loss_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"[train] Saved plot → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()