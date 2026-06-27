import numpy as np
import matplotlib.pyplot as plt
import os
from src.chaos.logistic import LogisticMap, LogisticParameters
from src.utils.logger import ExperimentLogger


class PredictiveSupervisor:
    """
    Mode 2 - Predictive Supervisor.

    Monitors early warning signals to detect
    instability before divergence becomes visible.
    Intervenes earlier than the reactive supervisor
    at lower cost.

    Early warning signals:
    1. Local Lyapunov estimate spike
    2. Divergence acceleration threshold

    Cooldown prevents consecutive interventions
    on normal chaotic fluctuations.
    """

    def __init__(
        self,
        r=4.0,
        perturbation_scale=1.0,
        lyapunov_window=10,
        lyapunov_spike_factor=1.05,
        accel_threshold=1.5,
        cooldown=5,
    ):
        self.r = r
        self.perturbation_scale = perturbation_scale
        self.lyapunov_window = lyapunov_window
        self.lyapunov_spike_factor = lyapunov_spike_factor
        self.accel_threshold = accel_threshold
        self.cooldown = cooldown
        self.params = LogisticParameters(r=r)
        self.lmap = LogisticMap(self.params)

    def _local_lyapunov(self, trajectory, t):
        """Compute Lyapunov estimate over recent window."""
        start = max(0, t - self.lyapunov_window)
        window = trajectory[start:t + 1]
        derivatives = np.abs(self.r * (1 - 2 * window))
        derivatives = derivatives[derivatives > 0]
        if len(derivatives) == 0:
            return 0.0
        return np.mean(np.log(derivatives))

    def run(self, x0_safe=0.3123, delta=1e-9, n_iter=100):
        """
        Run predictive supervised vs reactive vs baseline comparison.
        """

        global_lambda = np.log(2)

        # --- Trajectories ---
        traj_safe = np.zeros(n_iter)
        traj_predictive = np.zeros(n_iter)
        traj_reactive = np.zeros(n_iter)
        traj_baseline = np.zeros(n_iter)

        traj_safe[0] = x0_safe
        traj_predictive[0] = x0_safe + delta
        traj_reactive[0] = x0_safe + delta
        traj_baseline[0] = x0_safe + delta

        # --- Tracking ---
        predictive_distance = np.zeros(n_iter)
        reactive_distance = np.zeros(n_iter)
        baseline_distance = np.zeros(n_iter)
        local_lyapunov = np.zeros(n_iter)
        divergence_accel = np.zeros(n_iter)

        predictive_interventions = []
        reactive_interventions = []
        predictive_costs = []
        reactive_costs = []

        last_intervention = -999
        reactive_threshold = 1e-7
        reactive_saturation = 0.1

        for t in range(1, n_iter):

            # Step all trajectories
            traj_safe[t] = self.lmap.step(traj_safe[t - 1])
            traj_predictive[t] = self.lmap.step(traj_predictive[t - 1])
            traj_reactive[t] = self.lmap.step(traj_reactive[t - 1])
            traj_baseline[t] = self.lmap.step(traj_baseline[t - 1])

            # Compute distances
            predictive_distance[t] = abs(traj_safe[t] - traj_predictive[t])
            reactive_distance[t] = abs(traj_safe[t] - traj_reactive[t])
            baseline_distance[t] = abs(traj_safe[t] - traj_baseline[t])

            # --- Early warning signals ---
            local_lyapunov[t] = self._local_lyapunov(traj_safe, t)

            if t >= 2 and predictive_distance[t - 1] > 0:
                divergence_accel[t] = (
                    predictive_distance[t] - predictive_distance[t - 1]
                ) / (predictive_distance[t - 1] + 1e-12)

            # --- Predictive supervisor logic ---
            lyapunov_spike = (
                local_lyapunov[t] > global_lambda * self.lyapunov_spike_factor
            )
            accel_spike = divergence_accel[t] > self.accel_threshold
            cooldown_clear = (t - last_intervention) > self.cooldown

            if (lyapunov_spike or accel_spike) and cooldown_clear:
                correction = traj_safe[t] - traj_predictive[t]
                p_cost = abs(self.perturbation_scale * correction)
                traj_predictive[t] += self.perturbation_scale * correction
                # clamp to valid range
                traj_predictive[t] = np.clip(traj_predictive[t], 0.0, 1.0)
                predictive_distance[t] = abs(traj_safe[t] - traj_predictive[t])
                predictive_interventions.append(t)
                predictive_costs.append(p_cost)
                last_intervention = t
                print(f"  [Predictive] Intervened at t={t} | "
                      f"distance={predictive_distance[t]:.2e} | "
                      f"cost={p_cost:.2e} | "
                      f"trigger={'λ-spike' if lyapunov_spike else 'accel'}")

            # --- Reactive supervisor logic ---
            d_r = reactive_distance[t]
            if reactive_threshold < d_r < reactive_saturation:
                correction = traj_safe[t] - traj_reactive[t]
                r_cost = abs(correction)
                traj_reactive[t] += correction  # full correction
                # clamp to valid range
                traj_reactive[t] = np.clip(traj_reactive[t], 0.0, 1.0)
                reactive_distance[t] = abs(traj_safe[t] - traj_reactive[t])
                reactive_interventions.append(t)
                reactive_costs.append(r_cost)

        # --- Results ---
        final_predictive = predictive_distance[-1]
        final_reactive = reactive_distance[-1]
        final_baseline = baseline_distance[-1]

        improvement_predictive = (
            final_baseline - final_predictive
        ) / (final_baseline + 1e-12) * 100
        improvement_reactive = (
            final_baseline - final_reactive
        ) / (final_baseline + 1e-12) * 100
        total_predictive_cost = sum(predictive_costs)
        total_reactive_cost = sum(reactive_costs)

        print(f"\n  Final distance (predictive):  {final_predictive:.8f}")
        print(f"  Final distance (reactive):    {final_reactive:.8f}")
        print(f"  Final distance (baseline):    {final_baseline:.8f}")
        print(f"  Improvement (predictive):     {improvement_predictive:.1f}%")
        print(f"  Improvement (reactive):       {improvement_reactive:.1f}%")
        print(
            f"  Predictive interventions:     {len(predictive_interventions)}")
        print(f"  Reactive interventions:       {len(reactive_interventions)}")
        print(f"  Predictive total cost:        {total_predictive_cost:.2e}")
        print(f"  Reactive total cost:          {total_reactive_cost:.2e}")

        # --- Plot ---
        os.makedirs("figures", exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

        # Top: trajectories
        axes[0].plot(traj_safe, color="green",
                     linewidth=1.5, label="Safe")
        axes[0].plot(traj_predictive, color="steelblue",
                     linewidth=1.2, linestyle="--", label="Predictive")
        axes[0].plot(traj_reactive, color="orange",
                     linewidth=1.0, linestyle="-.", label="Reactive")
        axes[0].plot(traj_baseline, color="crimson",
                     linewidth=0.8, linestyle=":", alpha=0.6, label="Baseline")
        for t_int in predictive_interventions:
            axes[0].axvline(t_int, color="steelblue",
                            linewidth=0.8, linestyle="--", alpha=0.5)
        axes[0].set_ylabel("State x(t)")
        axes[0].set_title(
            "Predictive Supervisor — Mode 2 vs Mode 1 vs Baseline")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, linestyle="--", alpha=0.4)

        # Middle: divergence comparison
        axes[1].semilogy(baseline_distance + 1e-12, color="crimson",
                         linewidth=1.2, label="Baseline")
        axes[1].semilogy(reactive_distance + 1e-12, color="orange",
                         linewidth=1.2, linestyle="-.", label="Reactive")
        axes[1].semilogy(predictive_distance + 1e-12, color="steelblue",
                         linewidth=1.5, linestyle="--", label="Predictive")
        for t_int in predictive_interventions:
            axes[1].axvline(t_int, color="steelblue",
                            linewidth=0.8, linestyle="--", alpha=0.5)
        axes[1].set_ylabel("Distance (log scale)")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, which="both", linestyle="--", alpha=0.4)

        # Bottom: early warning signals
        axes[2].plot(local_lyapunov, color="purple",
                     linewidth=1.2, label="Local λ estimate")
        axes[2].axhline(
            global_lambda * self.lyapunov_spike_factor,
            color="gold", linestyle="--", linewidth=1,
            label=f"Spike threshold (λ × {self.lyapunov_spike_factor})"
        )
        axes[2].axhline(
            global_lambda, color="green", linestyle=":",
            linewidth=1, label=f"Global λ = {global_lambda:.4f}"
        )
        for t_int in predictive_interventions:
            axes[2].axvline(t_int, color="steelblue",
                            linewidth=0.8, linestyle="--", alpha=0.5)
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Local λ")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig("figures/predictive_supervisor.png", dpi=150)
        plt.show()
        print("Figure saved -> figures/predictive_supervisor.png")

        # --- Log ---
        logger = ExperimentLogger()
        logger.save("predictive_supervisor", {
            "r": self.r,
            "x0_safe": x0_safe,
            "delta": delta,
            "n_iter": n_iter,
            "perturbation_scale": self.perturbation_scale,
            "lyapunov_window": self.lyapunov_window,
            "lyapunov_spike_factor": self.lyapunov_spike_factor,
            "accel_threshold": self.accel_threshold,
            "cooldown": self.cooldown,
            "predictive_interventions": predictive_interventions,
            "predictive_total_cost": float(total_predictive_cost),
            "reactive_interventions": reactive_interventions,
            "reactive_total_cost": float(total_reactive_cost),
            "final_predictive_distance": float(final_predictive),
            "final_reactive_distance": float(final_reactive),
            "final_baseline_distance": float(final_baseline),
            "improvement_predictive_pct": float(improvement_predictive),
            "improvement_reactive_pct": float(improvement_reactive)
        })

        return {
            "predictive_distance": predictive_distance,
            "reactive_distance": reactive_distance,
            "baseline_distance": baseline_distance,
            "predictive_interventions": predictive_interventions,
            "reactive_interventions": reactive_interventions,
            "improvement_predictive": improvement_predictive,
            "improvement_reactive": improvement_reactive,
            "total_predictive_cost": total_predictive_cost,
            "total_reactive_cost": total_reactive_cost
        }
