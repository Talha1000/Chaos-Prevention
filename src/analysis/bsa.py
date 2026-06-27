import numpy as np
import matplotlib.pyplot as plt
import os
from src.chaos.logistic import LogisticMap, LogisticParameters
from src.utils.logger import ExperimentLogger


def compute_lyapunov_at(r, x0, n_iter=1000):
    """Compute Lyapunov exponent for a specific x0."""
    params = LogisticParameters(r=r)
    lmap = LogisticMap(params)
    trajectory = lmap.simulate(x0=x0, iterations=n_iter)
    derivatives = np.abs(r * (1 - 2 * trajectory))
    derivatives = derivatives[derivatives > 0]
    return np.mean(np.log(derivatives))


def run_bsa(r=4.0, n_points=500, n_iter=1000):
    """
    Sweep x0 across [0.01, 0.99] and compute Lyapunov exponent at each point.
    Produces a sensitivity map of the state space.
    """
    x0_values = np.linspace(0.01, 0.99, n_points)
    lyapunov_values = np.zeros(n_points)

    print("Running BSA sweep...")
    for i, x0 in enumerate(x0_values):
        lyapunov_values[i] = compute_lyapunov_at(r=r, x0=x0, n_iter=n_iter)

    # --- Identify regions ---
    mean_lam = np.mean(lyapunov_values)
    hot_mask = lyapunov_values > mean_lam        # high sensitivity
    calm_mask = lyapunov_values <= mean_lam      # low sensitivity

    hot_zones = x0_values[hot_mask]
    calm_zones = x0_values[calm_mask]

    # --- Plot ---
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.plot(x0_values, lyapunov_values, color="steelblue",
             linewidth=1.2, label="λ(x₀)")
    plt.axhline(mean_lam, color="crimson", linestyle="--",
                linewidth=1, label=f"Mean λ = {mean_lam:.4f}")
    plt.axhline(np.log(2), color="green", linestyle=":",
                linewidth=1, label=f"Theoretical λ = {np.log(2):.4f}")

    plt.fill_between(x0_values, lyapunov_values, mean_lam,
                     where=hot_mask, alpha=0.2, color="red", label="Hot zones")
    plt.fill_between(x0_values, lyapunov_values, mean_lam,
                     where=calm_mask, alpha=0.2, color="blue", label="Calm zones")

    plt.xlabel("Initial Condition x₀")
    plt.ylabel("Lyapunov Exponent λ")
    plt.title("Butterfly Sensitivity Analyzer — State Space Sensitivity Map")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("figures/bsa_sensitivity_map.png", dpi=150)
    plt.show()
    print("Figure saved -> figures/bsa_sensitivity_map.png")

    # --- Log ---
    logger = ExperimentLogger()
    logger.save("bsa", {
        "r": r,
        "n_points": n_points,
        "n_iter": n_iter,
        "mean_lambda": float(mean_lam),
        "theoretical_lambda": float(np.log(2)),
        "hot_zone_count": int(np.sum(hot_mask)),
        "calm_zone_count": int(np.sum(calm_mask)),
        "hot_zone_x0_range": [float(hot_zones.min()), float(hot_zones.max())] if len(hot_zones) > 0 else [],
        "calm_zone_x0_range": [float(calm_zones.min()), float(calm_zones.max())] if len(calm_zones) > 0 else [],
    })

    return x0_values, lyapunov_values
