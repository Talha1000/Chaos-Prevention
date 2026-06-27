import numpy as np
import matplotlib.pyplot as plt
import os
from src.chaos.logistic import LogisticMap, LogisticParameters
from src.utils.logger import ExperimentLogger


class ReactiveSupervsor:
    """
    Mode 1 - Reactive Supervisor.

    Watches two trajectories in real time.
    When divergence enters the optimal window,
    applies a minimal perturbation to redirect
    the unsafe trajectory toward the safe one.
    """

    def __init__(
        self,
        r=4.0,
        detection_threshold=1e-7,   # window opens
        saturation_threshold=0.1,   # window closes
        # how hard to push (0=nothing, 1=full correction)
        perturbation_scale=0.5,
    ):
        self.r = r
        self.detection_threshold = detection_threshold
        self.saturation_threshold = saturation_threshold
        self.perturbation_scale = perturbation_scale
        self.params = LogisticParameters(r=r)
        self.lmap = LogisticMap(self.params)

    def run(self, x0_safe=0.3123, delta=1e-9, n_iter=100):
        """
        Run supervised vs unsupervised comparison.

        - Safe trajectory: reference, never touched
        - Unsafe trajectory: starts delta away, gets supervised
        - Baseline trajectory: starts delta away, never touched
        """

        # --- Trajectories ---
        traj_safe = np.zeros(n_iter)
        traj_supervised = np.zeros(n_iter)
        traj_baseline = np.zeros(n_iter)

        traj_safe[0] = x0_safe
        traj_supervised[0] = x0_safe + delta
        traj_baseline[0] = x0_safe + delta

        # --- Tracking ---
        interventions = []
        perturbation_costs = []
        supervised_distance = np.zeros(n_iter)
        baseline_distance = np.zeros(n_iter)

        intervened = False

        for t in range(1, n_iter):

            # Step all trajectories
            traj_safe[t] = self.lmap.step(traj_safe[t - 1])
            traj_supervised[t] = self.lmap.step(traj_supervised[t - 1])
            traj_baseline[t] = self.lmap.step(traj_baseline[t - 1])

            # Compute distances
            supervised_distance[t] = abs(traj_safe[t] - traj_supervised[t])
            baseline_distance[t] = abs(traj_safe[t] - traj_baseline[t])

            # --- Supervisor logic ---
            d = supervised_distance[t]

            if self.detection_threshold < d < self.saturation_threshold:
                correction = traj_safe[t] - traj_supervised[t]
                perturbation = self.perturbation_scale * correction
                traj_supervised[t] += perturbation
                cost = abs(perturbation)
                interventions.append(t)
                perturbation_costs.append(cost)
                supervised_distance[t] = abs(traj_safe[t] - traj_supervised[t])
                print(f"  [Supervisor] Intervened at t={t} | "
                      f"distance={d:.2e} | cost={cost:.2e}")

        # --- Results ---
        final_supervised = supervised_distance[-1]
        final_baseline = baseline_distance[-1]
        improvement = (final_baseline - final_supervised) / \
            (final_baseline + 1e-12)

        print(f"\n  Final distance (supervised): {final_supervised:.6f}")
        print(f"  Final distance (baseline):   {final_baseline:.6f}")
        print(f"  Improvement:                 {improvement * 100:.1f}%")

        # --- Plot ---
        os.makedirs("figures", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Top: trajectories
        ax1.plot(traj_safe, color="green",
                 linewidth=1.5, label="Safe Trajectory")
        ax1.plot(traj_supervised, color="steelblue", linewidth=1.2,
                 linestyle="--", label="Supervised Trajectory")
        ax1.plot(traj_baseline, color="crimson", linewidth=1.0,
                 linestyle=":", alpha=0.7, label="Baseline (no supervision)")

        for t_int in interventions:
            ax1.axvline(t_int, color="gold", linewidth=2,
                        linestyle="--", label=f"Intervention (t={t_int})")

        ax1.set_ylabel("State x(t)")
        ax1.set_title("Reactive Supervisor — Trajectory Comparison")
        ax1.legend(fontsize=8)
        ax1.grid(True, linestyle="--", alpha=0.4)

        # Bottom: divergence comparison
        ax2.semilogy(baseline_distance, color="crimson", linewidth=1.2,
                     label="Baseline divergence")
        ax2.semilogy(supervised_distance + 1e-12, color="steelblue",
                     linewidth=1.5, label="Supervised divergence")

        for t_int in interventions:
            ax2.axvline(t_int, color="gold", linewidth=2, linestyle="--")

        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Distance (log scale)")
        ax2.legend(fontsize=8)
        ax2.grid(True, which="both", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig("figures/reactive_supervisor.png", dpi=150)
        plt.show()
        print("Figure saved -> figures/reactive_supervisor.png")

        # --- Log ---
        logger = ExperimentLogger()
        logger.save("reactive_supervisor", {
            "r": self.r,
            "x0_safe": x0_safe,
            "delta": delta,
            "n_iter": n_iter,
            "perturbation_scale": self.perturbation_scale,
            "detection_threshold": self.detection_threshold,
            "saturation_threshold": self.saturation_threshold,
            "interventions": interventions,
            "perturbation_costs": [float(c) for c in perturbation_costs],
            "final_supervised_distance": float(final_supervised),
            "final_baseline_distance": float(final_baseline),
            "improvement_percent": float(improvement * 100)
        })

        return {
            "traj_safe": traj_safe,
            "traj_supervised": traj_supervised,
            "traj_baseline": traj_baseline,
            "supervised_distance": supervised_distance,
            "baseline_distance": baseline_distance,
            "interventions": interventions,
            "improvement": improvement
        }
