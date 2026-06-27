import numpy as np
from src.experiments.divergence import run_divergence_experiment
from src.chaos.lyapunov import compute_lyapunov
from src.analysis.bsa import run_bsa
from src.analysis.iwa import run_iwa
from src.supervisor.reactive import ReactiveSupervsor
from src.utils.logger import ExperimentLogger

if __name__ == "__main__":

    # Experiment 1 — Divergence
    run_divergence_experiment()

    # Experiment 2 — Lyapunov
    lam = compute_lyapunov()
    print(f"\nLyapunov Exponent: {lam:.6f}")
    print(f"Theoretical (ln2): {np.log(2):.6f}")
    print(f"Difference:        {abs(lam - np.log(2)):.6f}")
    logger = ExperimentLogger()
    logger.save("lyapunov", {
        "computed": lam,
        "theoretical": float(np.log(2)),
        "difference": float(abs(lam - np.log(2)))
    })

    # Experiment 3 — BSA
    print("\nStarting BSA...")
    x0_vals, lam_vals = run_bsa()

    # Experiment 4 — IWA
    print("\nStarting IWA...")
    distance, rate, peak, opt_start, opt_end = run_iwa()
    print(f"Optimal intervention window: iterations {opt_start} to {opt_end}")
    print(f"Peak leverage point: iteration {peak}")

    # Experiment 5 — Reactive Supervisor
    print("\nStarting Reactive Supervisor...")
    supervisor = ReactiveSupervsor(
        r=4.0,
        detection_threshold=1e-7,
        saturation_threshold=0.1,
        perturbation_scale=1.0
    )
    results = supervisor.run(x0_safe=0.3123, delta=1e-9, n_iter=100)
