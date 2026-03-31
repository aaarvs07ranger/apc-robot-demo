"""
simulator.py — Physics Engine for a 1-Link Motorized Pendulum
=============================================================

Simulates a single rigid link (pendulum) driven by a motor torque,
using Euler integration. The governing equation of motion is:

    θ̈ = (τ - m·g·L·sin(θ) - b·θ̇) / I

where:
    θ   = joint angle (rad)
    θ̇   = angular velocity (rad/s)
    τ   = applied motor torque (N·m)
    m   = link mass (kg)
    g   = gravitational acceleration (m/s²)
    L   = distance from pivot to centre of mass (m)
    b   = viscous damping coefficient (N·m·s/rad)
    I   = moment of inertia about the pivot (kg·m²)
"""

import numpy as np


# ─────────────────────────────────────────────
# Physical constants for the pendulum
# ─────────────────────────────────────────────
MASS       = 1.0        # kg
LENGTH     = 1.0        # m  (pivot to CoM)
GRAVITY    = 9.81       # m/s²
DAMPING    = 0.1        # N·m·s/rad  (viscous friction)
INERTIA    = MASS * LENGTH ** 2   # kg·m²  (thin-rod approx. about pivot)
DT         = 0.05       # s  (integration time-step)

# Torque sampling range — wide enough to swing the pendulum chaotically
TORQUE_MIN = -5.0       # N·m
TORQUE_MAX =  5.0       # N·m


def pendulum_step(theta: float, theta_dot: float, tau: float) -> tuple:
    """
    Advance the pendulum state by one Euler step.

    Parameters
    ----------
    theta     : float   Current angle (rad).
    theta_dot : float   Current angular velocity (rad/s).
    tau       : float   Applied motor torque (N·m).

    Returns
    -------
    theta_next     : float   Angle after one time-step.
    theta_dot_next : float   Angular velocity after one time-step.
    """
    # Angular acceleration from Newton's second law (rotational)
    theta_ddot = (tau - MASS * GRAVITY * LENGTH * np.sin(theta)
                  - DAMPING * theta_dot) / INERTIA

    # Simple (forward) Euler integration
    theta_dot_next = theta_dot + theta_ddot * DT
    theta_next     = theta     + theta_dot_next * DT

    return theta_next, theta_dot_next


def generate_dataset(n_samples: int = 50_000,
                     seed: int = 42) -> tuple:
    """
    Generate a dataset of continuous state transitions under random torques.

    At every time-step the motor torque is drawn uniformly at random from
    [TORQUE_MIN, TORQUE_MAX].  This produces chaotic, unpredictable motion —
    exactly the regime that separates a model *with* the efference copy from
    one without.

    Parameters
    ----------
    n_samples : int   Number of (state, action, next_state) tuples.
    seed      : int   Random seed for reproducibility.

    Returns
    -------
    states      : np.ndarray  shape (n_samples, 2)  — [θ, θ̇]
    actions     : np.ndarray  shape (n_samples, 1)  — [τ]
    next_states : np.ndarray  shape (n_samples, 2)  — [θ', θ̇']
    """
    rng = np.random.default_rng(seed)

    # Pre-allocate arrays
    states      = np.zeros((n_samples, 2), dtype=np.float32)
    actions     = np.zeros((n_samples, 1), dtype=np.float32)
    next_states = np.zeros((n_samples, 2), dtype=np.float32)

    # Initial conditions — start at a random angle, zero velocity
    theta     = rng.uniform(-np.pi, np.pi)
    theta_dot = 0.0

    for i in range(n_samples):
        # Random motor torque (the "efference copy" variable)
        tau = rng.uniform(TORQUE_MIN, TORQUE_MAX)

        # Record current state and action
        states[i]  = [theta, theta_dot]
        actions[i] = [tau]

        # Step the physics
        theta, theta_dot = pendulum_step(theta, theta_dot, tau)

        # Record resulting state
        next_states[i] = [theta, theta_dot]

    print(f"[simulator] Generated {n_samples:,} transitions  "
          f"(θ ∈ [{states[:,0].min():.2f}, {states[:,0].max():.2f}],  "
          f"τ ∈ [{TORQUE_MIN}, {TORQUE_MAX}])")

    return states, actions, next_states


# ─────────────────────────────────────────────
# Quick smoke-test when run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    s, a, ns = generate_dataset(1000)
    print(f"states      : {s.shape}")
    print(f"actions     : {a.shape}")
    print(f"next_states : {ns.shape}")