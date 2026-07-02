import numpy as np
import matplotlib.pyplot as plt
import os
from src.chaos.lorenz import LorenzSystem, LorenzParameters
from src.utils.logger import ExperimentLogger


def compute_lorenz_lyapunov():
    """Compute largest Lyapunov exponent for Lorenz system."""
    params = LorenzParameters()
    lsys = LorenzSystem(params)
    return lsys.compute_lyapunov()


def predict_intervention_window(lam, delta=1e-6,
                                detection=1e-5,
                                saturation=1.0):
    t_open = np.log(detection / delta) / lam
    t_close = np.log(saturation / delta) / lam
    return t_open, t_close


def run_lorenz_experiment(x0=1.0, y0=1.0, z0=1.0,
                          delta=1e-6, n_iter=3000):
    """
    Full BIA experiment on Lorenz system.
    1. Compute λ
    2. Predict intervention window
    3. Run divergence experiment
    4. Find observed window
    5. Run reactive supervisor
    6. Compare prediction to observation
    """

    params = LorenzParameters()
    lsys = LorenzSystem(params)

    # --- Step 1: Compute Lyapunov ---
    print("Computing Lorenz Lyapunov exponent...")
    lam = compute_lorenz_lyapunov()
    print(f"Lorenz Lyapunov exponent: {lam:.6f}")
    print(f"Theoretical value:        ~0.906000")

    # --- Step 2: Predict window ---
    t_open, t_close = predict_intervention_window(
        lam,
        delta=1e-6,
        detection=1e-5,
        saturation=1.0
    )
    print(f"\nPredicted intervention window:")
    print(f"  Opens at:  t = {t_open:.1f}")
    print(f"  Closes at: t = {t_close:.1f}")

    # --- Step 3: Divergence experiment ---
    ax, ay, az = lsys.simulate(x0, y0, z0, n_iter)
    bx, by, bz = lsys.simulate(x0 + delta, y0, z0, n_iter)

    distance = np.sqrt(
        (ax - bx)**2 +
        (ay - by)**2 +
        (az - bz)**2
    )

    # --- Debug: print distance at key points ---
    print(f"\nDistance samples:")
    for t in [1, 5, 10, 15, 20, 25, 30, 50, 100, 200]:
        if t < n_iter:
            print(f"  t={t:3d}: distance = {distance[t]:.4e}")
    print(f"  Max distance: {np.max(distance):.4e}")
    print(f"  Min distance: {np.min(distance[1:]):.4e}")

    # --- Step 4: Find observed window ---
    detection_threshold = 1e-5
    saturation_threshold = 1.0

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
    sx, sy, sz = np.zeros(n_iter), np.zeros(n_iter), np.zeros(n_iter)
    ux, uy, uz = np.zeros(n_iter), np.zeros(n_iter), np.zeros(n_iter)

    sx[0], sy[0], sz[0] = x0, y0, z0
    ux[0], uy[0], uz[0] = x0 + delta, y0, z0

    supervised_distance = np.zeros(n_iter)
    interventions = []
    intervened = False

    for t in range(1, n_iter):
        sx[t], sy[t], sz[t] = lsys.step(sx[t-1], sy[t-1], sz[t-1])
        ux[t], uy[t], uz[t] = lsys.step(ux[t-1], uy[t-1], uz[t-1])

        d = np.sqrt(
            (sx[t] - ux[t])**2 +
            (sy[t] - uy[t])**2 +
            (sz[t] - uz[t])**2
        )
        supervised_distance[t] = d

        if not intervened and detection_threshold < d < saturation_threshold:
            ux[t] = sx[t]
            uy[t] = sy[t]
            uz[t] = sz[t]
            supervised_distance[t] = 0.0
            interventions.append(t)
            intervened = True
            print(f"\n  [Supervisor] Intervened at t={t} | "
                  f"distance={d:.2e}")

    final_supervised = supervised_distance[-1]
    final_baseline = float(np.clip(distance[-1], 0, 100))
    improvement = (final_baseline - final_supervised) / (
        final_baseline + 1e-12) * 100

    print(f"  Final supervised distance: {final_supervised:.6f}")
    print(f"  Final baseline distance:   {final_baseline:.6f}")
    print(f"  Improvement:               {improvement:.1f}%")

    # --- Step 6: Comparison ---
    print(f"\n{'='*50}")
    print(f"PREDICTION vs OBSERVATION — LORENZ")
    print(f"{'='*50}")
    print(f"Window open:  predicted={t_open:.1f}  "
          f"observed={observed_open}")
    print(f"Window close: predicted={t_close:.1f}  "
          f"observed={observed_close}")

    # --- Plot ---
    os.makedirs("figures", exist_ok=True)
    fig = plt.figure(figsize=(14, 11))

    # Top: x trajectory
    ax1 = fig.add_subplot(3, 2, (1, 2))
    ax1.plot(ax, color="green", linewidth=0.8, label="Safe (x)")
    ax1.plot(bx, color="crimson", linewidth=0.8,
             linestyle=":", alpha=0.7, label="Baseline (x)")
    ax1.plot(ux, color="steelblue", linewidth=0.8,
             linestyle="--", label="Supervised (x)")
    for t_int in interventions:
        ax1.axvline(t_int, color="gold", linewidth=2,
                    linestyle="--", label=f"Intervention t={t_int}")
    ax1.set_ylabel("x(t)")
    ax1.set_title("BIA on Lorenz System — Generalization Experiment")
    ax1.legend(fontsize=7)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Middle: divergence
    ax2 = fig.add_subplot(3, 2, (3, 4))
    ax2.semilogy(distance + 1e-12, color="crimson",
                 linewidth=1.2, label="Baseline divergence")
    ax2.semilogy(supervised_distance + 1e-12, color="steelblue",
                 linewidth=1.5, linestyle="--",
                 label="Supervised divergence")
    ax2.axvline(t_open, color="green", linewidth=1.5,
                linestyle="--",
                label=f"Predicted open t={t_open:.1f}")
    ax2.axvline(t_close, color="orange", linewidth=1.5,
                linestyle="--",
                label=f"Predicted close t={t_close:.1f}")
    if observed_open:
        ax2.axvline(observed_open, color="green", linewidth=1,
                    linestyle=":",
                    label=f"Observed open t={observed_open}")
    if observed_close:
        ax2.axvline(observed_close, color="orange", linewidth=1,
                    linestyle=":",
                    label=f"Observed close t={observed_close}")
    ax2.set_ylabel("Distance (log scale)")
    ax2.legend(fontsize=7)
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)

    # Bottom left: Lorenz attractor XZ plane
    ax3 = fig.add_subplot(3, 2, 5)
    ax3.plot(ax[:500], az[:500], linewidth=0.4,
             color="steelblue", alpha=0.7)
    ax3.set_xlabel("x")
    ax3.set_ylabel("z")
    ax3.set_title("Lorenz Attractor (XZ plane)")
    ax3.grid(True, linestyle="--", alpha=0.3)

    # Bottom right: Lorenz attractor XY plane
    ax4 = fig.add_subplot(3, 2, 6)
    ax4.plot(ax[:500], ay[:500], linewidth=0.4,
             color="crimson", alpha=0.7)
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_title("Lorenz Attractor (XY plane)")
    ax4.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/lorenz_experiment.png", dpi=150)
    plt.show()
    print("Figure saved -> figures/lorenz_experiment.png")

    # --- Log ---
    logger = ExperimentLogger()
    logger.save("lorenz_experiment", {
        "sigma": 10.0,
        "rho": 28.0,
        "beta": float(8/3),
        "lyapunov_computed": float(lam),
        "lyapunov_theoretical": 0.906,
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
