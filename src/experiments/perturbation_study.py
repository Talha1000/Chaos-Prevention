import numpy as np
import matplotlib.pyplot as plt
import os
from src.chaos.logistic import LogisticMap, LogisticParameters
from src.utils.logger import ExperimentLogger


def run_perturbation_study(
    r=4.0,
    x0_safe=0.3123,
    delta=1e-9,
    n_iter=100,
    detection_threshold=1e-7,
    saturation_threshold=0.1
):
    """
    Sweep perturbation_scale from 0.0 to 1.0.
    For each scale, measure final divergence.
    Find the true stable threshold where recovery becomes reliable.
    """

    scales = np.linspace(0.0, 1.0, 200)
    final_distances = np.zeros(len(scales))
    improvement_pcts = np.zeros(len(scales))

    params = LogisticParameters(r=r)
    lmap = LogisticMap(params)

    # Baseline (no intervention)
    traj_safe = lmap.simulate(x0=x0_safe, iterations=n_iter)
    traj_base = lmap.simulate(x0=x0_safe + delta, iterations=n_iter)
    baseline_final = abs(traj_safe[-1] - traj_base[-1])

    print(f"Baseline final distance: {baseline_final:.6f}")
    print(f"Sweeping {len(scales)} perturbation scales...")

    for idx, scale in enumerate(scales):

        # Reset trajectories
        traj_s = np.zeros(n_iter)
        traj_u = np.zeros(n_iter)
        traj_s[0] = x0_safe
        traj_u[0] = x0_safe + delta

        for t in range(1, n_iter):
            traj_s[t] = lmap.step(traj_s[t - 1])
            traj_u[t] = lmap.step(traj_u[t - 1])

            d = abs(traj_s[t] - traj_u[t])

            if detection_threshold < d < saturation_threshold:
                correction = traj_s[t] - traj_u[t]
                traj_u[t] += scale * correction

        final_d = abs(traj_s[-1] - traj_u[-1])
        final_distances[idx] = final_d
        improvement_pcts[idx] = (baseline_final - final_d) / \
            (baseline_final + 1e-12) * 100

    # --- Find true stable effective scale ---
    success_threshold = 0.01
    window = 10  # must stay below threshold for 10 consecutive points

    true_effective_scale = None
    for i in range(len(scales) - window):
        if np.all(final_distances[i:i + window] < success_threshold):
            true_effective_scale = scales[i]
            break

    if true_effective_scale:
        print(f"True stable effective scale: {true_effective_scale:.4f}")
    else:
        print("No stable threshold found")
    print(f"At scale 1.0: final distance = {final_distances[-1]:.8f}")

    # --- Plot ---
    os.makedirs("figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: final distance vs scale
    ax1.plot(scales, final_distances, color="steelblue", linewidth=1.5)
    ax1.axhline(baseline_final, color="crimson", linestyle="--",
                linewidth=1, label=f"Baseline = {baseline_final:.4f}")
    ax1.axhline(success_threshold, color="green", linestyle=":",
                linewidth=1, label=f"Success threshold = {success_threshold}")
    if true_effective_scale:
        ax1.axvline(true_effective_scale, color="gold", linewidth=2,
                    linestyle="--",
                    label=f"Critical threshold = {true_effective_scale:.4f}")
    ax1.set_ylabel("Final Divergence")
    ax1.set_title("Perturbation Cost Study — Critical Intervention Threshold")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.set_yscale("log")

    # Bottom: improvement % vs scale
    ax2.plot(scales, improvement_pcts, color="green", linewidth=1.5)
    ax2.axhline(99, color="gold", linestyle="--",
                linewidth=1, label="99% improvement threshold")
    ax2.axhline(0, color="crimson", linestyle=":", linewidth=1)
    if true_effective_scale:
        ax2.axvline(true_effective_scale, color="gold",
                    linewidth=2, linestyle="--",
                    label=f"Critical threshold = {true_effective_scale:.4f}")

    # Shade the three regimes
    ax2.axvspan(0, true_effective_scale if true_effective_scale else 0.45,
                alpha=0.08, color="red", label="Chaotic response zone")
    ax2.axvspan(true_effective_scale if true_effective_scale else 0.45, 1.0,
                alpha=0.08, color="green", label="Stable recovery zone")

    ax2.set_xlabel("Perturbation Scale")
    ax2.set_ylabel("Improvement %")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("figures/perturbation_study.png", dpi=150)
    plt.show()
    print("Figure saved -> figures/perturbation_study.png")

    # --- Log ---
    logger = ExperimentLogger()
    logger.save("perturbation_study", {
        "r": r,
        "x0_safe": x0_safe,
        "delta": delta,
        "n_iter": n_iter,
        "baseline_final_distance": float(baseline_final),
        "true_effective_scale": float(true_effective_scale) if true_effective_scale else None,
        "chaotic_response_zone": [0.0, float(true_effective_scale) if true_effective_scale else None],
        "stable_recovery_zone": [float(true_effective_scale) if true_effective_scale else None, 1.0],
        "scale_at_100pct": 1.0,
        "final_distance_at_100pct": float(final_distances[-1]),
        "n_scales_tested": len(scales)
    })

    return scales, final_distances, improvement_pcts, true_effective_scale
