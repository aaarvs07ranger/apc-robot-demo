"""
models.py — Two Competing Forward Models
=========================================

Both models share the **exact same** hidden-layer architecture so that
any performance difference is attributable *only* to the input
representation, not to capacity.

Model A  — "Passive Observer"
    Input:  (θ, θ̇)          shape (2,)
    The model sees the current state but has **no knowledge** of the
    motor command that will be applied.  It must implicitly marginalise
    over all possible torques, which produces a statistical blur.

Model B  — "Active Predictive Coding" (APC)
    Input:  (θ, θ̇, τ)       shape (3,)
    The model receives the state *plus* the efference copy (the intended
    action).  Because the physics are deterministic given (state, action),
    the mapping is a clean, learnable function.

Both models output the next state (θ', θ̇')  shape (2,).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _build_forward_model(input_dim: int, name: str) -> keras.Model:
    """
    Shared factory: 3 hidden layers × 64 units, ReLU activations.

    Parameters
    ----------
    input_dim : int    Dimensionality of the input vector.
    name      : str    Human-readable model name (for logs & plots).

    Returns
    -------
    keras.Model  Uncompiled Sequential model.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(2),                       # predict (θ', θ̇')
    ], name=name)
    return model


def build_passive_observer() -> keras.Model:
    """Model A — state only, no efference copy."""
    return _build_forward_model(input_dim=2, name="Passive_Observer")


def build_active_predictor() -> keras.Model:
    """Model B — state + action (efference copy)."""
    return _build_forward_model(input_dim=3, name="Active_Predictor_APC")