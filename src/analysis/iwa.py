import numpy as np
import matplotlib.pyplot as plt
import os
from src.chaos.logistic import LogisticMap, LogisticParameters
from src.utils.logger import ExperimentLogger


def run_iwa(r=4.0, x0_a=0.3123, delta=1e-9, n_iter=100):
    """
    Intervention Window Analyzer.
    Finds the optimal timing window for external perturbation.

    Three zones:
    - Too Early: divergence growing but leverage too small
    - Optimal Window: maximum redirection per unit perturbation
    - Too Late: trajectories saturated, no recovery possible
    """

    params = LogisticParameters(r=r)
    lmap = LogisticMap(params)

    traj_a = lmap.simulate(x0=x0_a, iterations=n_iter)
    traj_b = lmap.simulate(x0=x0_a + delta, iterations=n_iter)

    distance = np.abs(traj_a - traj_b)

    # --- Compute divergence rate at each step ---
    rate = np.zeros(n_iter)
    for t in range(1, n_iter):
        if distance[t - 1] > 0:
            rate[t] = np.log(distance[t] / distance[t - 1] + 1e-12)

    # --- Define intervention zones ---
    # Saturation threshold: when distance exceeds 10% of state space
    saturation_threshold = 0.1
    saturation_idx = np.argmax(distance > saturation_threshold)
    if saturation_idx == 0:
        saturation_idx = n_iter - 1

    # Early zone: first 20% of time before saturation
    early_end = int(saturation_idx * 0.2)

    # Optimal window: 20% to 70% of time before saturation
    optimal_start = early_end
    optimal_end = int(saturation_idx * 0.7)

    # Late zone: after 70% of time before saturation
    late_start = optimal_end

    # --- Find peak leverage point ---
    # Maximum rate of change in divergence rate = best moment to intervene
    rate_of_rate = np.abs(np.diff(rate))
    peak_leverage = np.argmax(
        rate_of_rate[optimal_start:optimal_end]) + optimal_start

    # --- Plot ---
    os.makedirs("figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top plot: distance on log scale
    ax1.semilogy(distance, color="steelblue",
                 linewidth=1.5, label="|A(t) - B(t)|")
    ax1.axvspan(0, early_end, alpha=0.15, color="blue", label="Too Early")
    ax1.axvspan(optimal_start, optimal_end, alpha=0.15,
                color="green", label="Optimal Window")
    ax1.axvspan(late_start, n_iter, alpha=0.15, color="red", label="Too Late")
    ax1.axvline(peak_leverage, color="gold", linewidth=2,
                linestyle="--", label=f"Peak Leverage (t={peak_leverage})")
    ax1.axvline(saturation_idx, color="black", linewidth=1,
                linestyle=":", label=f"Saturation (t={saturation_idx})")
    ax1.set_ylabel("Distance |A(t) - B(t)| (log scale)")
    ax1.set_title("Intervention Window Analyzer — Optimal Perturbation Timing")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    # Bottom plot: divergence rate
    ax2.plot(rate, color="crimson", linewidth=1.2, label="Divergence Rate")
    ax2.axvspan(0, early_end, alpha=0.15, color="blue")
    ax2.axvspan(optimal_start, optimal_end, alpha=0.15, color="green")
    ax2.axvspan(late_start, n_iter, alpha=0.15, color="red")
    ax2.axvline(peak_leverage, color="gold", linewidth=2,
                linestyle="--", label=f"Peak Leverage (t={peak_leverage})")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("log(d(t) / d(t-1))")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("figures/iwa_intervention_window.png", dpi=150)
    plt.show()
    print("Figure saved -> figures/iwa_intervention_window.png")

    # --- Log ---
    logger = ExperimentLogger()
    logger.save("iwa", {
        "r": r,
        "x0_a": x0_a,
        "delta": delta,
        "n_iter": n_iter,
        "saturation_iteration": int(saturation_idx),
        "early_zone": [0, int(early_end)],
        "optimal_window": [int(optimal_start), int(optimal_end)],
        "late_zone": [int(late_start), n_iter],
        "peak_leverage_iteration": int(peak_leverage),
        "distance_at_peak": float(distance[peak_leverage]),
        "saturation_threshold": saturation_threshold
    })

    return distance, rate, peak_leverage, optimal_start, optimal_end
