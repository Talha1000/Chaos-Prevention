import numpy as np
import matplotlib.pyplot as plt
import os
from src.chaos.henon import HenonMap, HenonParameters
from src.utils.logger import ExperimentLogger


def compute_henon_lyapunov(a=1.4, b=0.3, x0=0.1, y0=0.1, n_iter=10000):
    """
    Compute Lyapunov exponent for Henon map.
    Uses the standard method of tracking
    the growth of a tangent vector.
    """
    params = HenonParameters(a=a, b=b)
    hmap = HenonMap(params)

    x, y = x0, y0
    lam_sum = 0.0

    # Tangent vector
    dx, dy = 1.0, 0.0

    for _ in range(n_iter):
        # Jacobian of Henon map:
        # J = [[-2ax, 1],
        #      [b,    0]]
        dx_new = -2 * a * x * dx + dy
        dy_new = b * dx

        x, y = hmap.step(x, y)

        # Norm of tangent vector
        norm = np.sqrt(dx_new ** 2 + dy_new ** 2)
        if norm > 0:
            lam_sum += np.log(norm)
            dx = dx_new / norm
            dy = dy_new / norm

    return lam_sum / n_iter


def predict_intervention_window(lam, delta=1e-9,
                                detection=1e-7,
                                saturation=0.1):
    """Predict intervention window from Lyapunov exponent."""
    t_open = np.log(detection / delta) / lam
    t_close = np.log(saturation / delta) / lam
    return t_open, t_close


def run_henon_experiment(a=1.4, b=0.3,
                         x0=0.1, y0=0.1,
                         delta=1e-9, n_iter=200):
    """
    Full BIA experiment on Henon map.
    1. Compute λ
    2. Predict intervention window
    3. Run divergence experiment
    4. Run IWA to find observed window
    5. Run reactive supervisor
    6. Compare prediction to observation
    """

    params = HenonParameters(a=a, b=b)
    hmap = HenonMap(params)

    # --- Step 1: Compute Lyapunov ---
    lam = compute_henon_lyapunov(a=a, b=b, x0=x0, y0=y0)
    print(f"Henon Lyapunov exponent: {lam:.6f}")
    print(f"Theoretical value:       ~0.418000")

    # --- Step 2: Predict window ---
    t_open, t_close = predict_intervention_window(lam)
    print(f"\nPredicted intervention window:")
    print(f"  Opens at:  t = {t_open:.1f}")
    print(f"  Closes at: t = {t_close:.1f}")

    # --- Step 3: Divergence experiment ---
    traj_ax, traj_ay = hmap.simulate(x0, y0, n_iter)
    traj_bx, traj_by = hmap.simulate(x0 + delta, y0, n_iter)

    distance = np.sqrt(
        (traj_ax - traj_bx) ** 2 +
        (traj_ay - traj_by) ** 2
    )

    # --- Step 4: Find observed window ---
    detection_threshold = 1e-7
    saturation_threshold = 0.5

    observed_open = None
    observed_close = None

    for t in range(1, n_iter):
        if observed_open is None and distance[t] > detection_threshold:
            observed_open = t
        if observed_open is not None and observed_close is None:
            if distance[t] > saturation_threshold:
                observed_close = t
                break

    print(f"\nObserved intervention window:")
    print(f"  Opens at:  t = {observed_open}")
    print(f"  Closes at: t = {observed_close}")

    # --- Step 5: Reactive supervisor ---
    traj_sx, traj_sy = np.zeros(n_iter), np.zeros(n_iter)
    traj_ux, traj_uy = np.zeros(n_iter), np.zeros(n_iter)

    traj_sx[0], traj_sy[0] = x0, y0
    traj_ux[0], traj_uy[0] = x0 + delta, y0

    supervised_distance = np.zeros(n_iter)
    interventions = []
    intervened = False

    for t in range(1, n_iter):
        traj_sx[t], traj_sy[t] = hmap.step(traj_sx[t-1], traj_sy[t-1])
        traj_ux[t], traj_uy[t] = hmap.step(traj_ux[t-1], traj_uy[t-1])

        d = np.sqrt(
            (traj_sx[t] - traj_ux[t]) ** 2 +
            (traj_sy[t] - traj_uy[t]) ** 2
        )
        supervised_distance[t] = d

        if not intervened and detection_threshold < d < saturation_threshold:
            # Full correction on x component
            traj_ux[t] = traj_sx[t]
            traj_uy[t] = traj_sy[t]
            supervised_distance[t] = 0.0
            interventions.append(t)
            intervened = True
            print(f"\n  [Supervisor] Intervened at t={t} | "
                  f"distance={d:.2e}")

    final_supervised = supervised_distance[-1]
    final_baseline = distance[-1] if distance[-1] < 10 else 1.0
    improvement = (final_baseline - final_supervised) / (
        final_baseline + 1e-12) * 100

    print(f"  Final supervised distance: {final_supervised:.6f}")
    print(f"  Final baseline distance:   {final_baseline:.6f}")
    print(f"  Improvement:               {improvement:.1f}%")

    # --- Step 6: Comparison ---
    print(f"\n{'='*45}")
    print(f"PREDICTION vs OBSERVATION")
    print(f"{'='*45}")
    print(f"Window open:  predicted={t_open:.1f}  "
          f"observed={observed_open}")
    print(f"Window close: predicted={t_close:.1f}  "
          f"observed={observed_close}")

    # --- Plot ---
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Top: Henon attractor trajectory (x component)
    axes[0].plot(traj_ax, color="green", linewidth=0.8,
                 label="Safe trajectory (x)")
    axes[0].plot(traj_bx, color="crimson", linewidth=0.8,
                 linestyle=":", alpha=0.7, label="Baseline (x)")
    axes[0].plot(traj_ux, color="steelblue", linewidth=0.8,
                 linestyle="--", label="Supervised (x)")
    for t_int in interventions:
        axes[0].axvline(t_int, color="gold", linewidth=2,
                        linestyle="--", label=f"Intervention t={t_int}")
    axes[0].set_ylabel("x(t)")
    axes[0].set_title("BIA on Henon Map — Generalization Experiment")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Middle: divergence
    axes[1].semilogy(distance + 1e-12, color="crimson",
                     linewidth=1.2, label="Baseline divergence")
    axes[1].semilogy(supervised_distance + 1e-12, color="steelblue",
                     linewidth=1.5, linestyle="--",
                     label="Supervised divergence")
    if observed_open:
        axes[1].axvline(observed_open, color="green", linewidth=1,
                        linestyle=":", label=f"Observed open t={observed_open}")
    if observed_close:
        axes[1].axvline(observed_close, color="orange", linewidth=1,
                        linestyle=":", label=f"Observed close t={observed_close}")
    axes[1].axvline(t_open, color="green", linewidth=1.5,
                    linestyle="--", label=f"Predicted open t={t_open:.1f}")
    axes[1].axvline(t_close, color="orange", linewidth=1.5,
                    linestyle="--", label=f"Predicted close t={t_close:.1f}")
    axes[1].set_ylabel("Distance (log scale)")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)

    # Bottom: Henon attractor shape
    axes[2].scatter(traj_ax[:500], traj_ay[:500],
                    s=0.5, color="steelblue", alpha=0.6)
    axes[2].set_xlabel("Iteration / x")
    axes[2].set_ylabel("y(t) / Attractor")
    axes[2].set_title("Henon Attractor (first 500 points)")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("figures/henon_experiment.png", dpi=150)
    plt.show()
    print("Figure saved -> figures/henon_experiment.png")

    # --- Log ---
    logger = ExperimentLogger()
    logger.save("henon_experiment", {
        "a": a, "b": b,
        "lyapunov_computed": float(lam),
        "lyapunov_theoretical": 0.418,
        "predicted_window_open": float(t_open),
        "predicted_window_close": float(t_close),
        "observed_window_open": int(observed_open) if observed_open else None,
        "observed_window_close": int(observed_close) if observed_close else None,
        "interventions": interventions,
        "improvement_pct": float(improvement),
        "final_supervised_distance": float(final_supervised),
        "final_baseline_distance": float(final_baseline)
    })

    return {
        "lam": lam,
        "predicted": (t_open, t_close),
        "observed": (observed_open, observed_close),
        "improvement": improvement
    }
