import numpy as np
import matplotlib.pyplot as plt
import os
from src.chaos.logistic import LogisticMap, LogisticParameters
from src.utils.logger import ExperimentLogger


def run_divergence_experiment(r=4.0, x0_a=0.3123, delta=1e-9, n_iter=100):

    x0_b = x0_a + delta

    params = LogisticParameters(r=r)
    lmap = LogisticMap(params)

    traj_a = lmap.simulate(x0=x0_a, iterations=n_iter)
    traj_b = lmap.simulate(x0=x0_b, iterations=n_iter)

    distance = np.abs(traj_a - traj_b)

    # --- Plot ---
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.semilogy(distance, color="crimson", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Distance |A(t) - B(t)| (log scale)")
    plt.title(f"Trajectory Divergence — Δx₀ = {delta:.0e}")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/divergence_experiment.png", dpi=150)
    plt.show()
    print("Figure saved -> figures/divergence_experiment.png")

    # --- Log ---
    logger = ExperimentLogger()
    logger.save("divergence", {
        "r": r,
        "x0_a": x0_a,
        "x0_b": x0_b,
        "delta": delta,
        "n_iter": n_iter,
        "max_distance": float(np.max(distance)),
        "iterations_to_saturation": int(np.argmax(distance > 0.5))
    })

    return distance
