import numpy as np
from src.experiments.divergence import run_divergence_experiment
from src.chaos.lyapunov import compute_lyapunov
from src.analysis.bsa import run_bsa
from src.analysis.iwa import run_iwa
from src.supervisor.reactive import ReactiveSupervsor
from src.experiments.perturbation_study import run_perturbation_study
from src.supervisor.predictive import PredictiveSupervisor
from src.utils.logger import ExperimentLogger
from src.experiments.henon_experiment import run_henon_experiment
from src.experiments.lorenz_experiment import run_lorenz_experiment
from src.experiments.neural_drift import run_neural_drift_experiment

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

    # Experiment 6 — Perturbation Study
    print("\nStarting Perturbation Study...")
    scales, distances, improvements, true_scale = run_perturbation_study()
    if true_scale:
        print(f"Critical threshold: {true_scale:.4f}")
        print(f"Below {true_scale:.4f} — chaotic response zone")
        print(f"Above {true_scale:.4f} — stable recovery zone")

    # Experiment 7 — Predictive Supervisor
    print("\nStarting Predictive Supervisor...")
    predictive = PredictiveSupervisor(
        r=4.0,
        perturbation_scale=1.0,      # full correction every time
        lyapunov_window=10,
        lyapunov_spike_factor=1.05,  # more sensitive than before
        accel_threshold=1.5,         # lower than before
        cooldown=5                   # shorter cooldown
    )
    pred_results = predictive.run(x0_safe=0.3123, delta=1e-9, n_iter=100)

    # --- Final Summary ---
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Reactive  — interventions: 1  | "
          f"improvement: {results['improvement'] * 100:.1f}%")
    print(f"Predictive — interventions: "
          f"{len(pred_results['predictive_interventions'])} | "
          f"improvement: {pred_results['improvement_predictive']:.1f}% | "
          f"cost: {pred_results['total_predictive_cost']:.2e}")

    # Experiment 8 — Henon Map Generalization
    print("\nStarting Henon Map Experiment...")
    henon_results = run_henon_experiment()

    # Experiment 9 — Lorenz System
    print("\nStarting Lorenz System Experiment...")
    lorenz_results = run_lorenz_experiment()

    # Experiment 10 — Neural Network Drift
    print("\nStarting Neural Network Drift Experiment...")
    nn_results = run_neural_drift_experiment()
    print(f"\nNeural Network Results:")
    print(f"  Unsupervised MSE: {nn_results['unsupervised_mse']:.6f}")
    print(f"  Supervised MSE:   {nn_results['supervised_mse']:.6f}")
    print(f"  Improvement:      {nn_results['improvement']:.1f}%")
    print(f"  Interventions:    {len(nn_results['interventions'])}")
