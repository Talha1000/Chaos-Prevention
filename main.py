import numpy as np

from src.chaos.lyapunov import compute_lyapunov
from src.experiments.divergence import run_divergence_experiment
from src.utils.logger import ExperimentLogger

if __name__ == "__main__":
    run_divergence_experiment()

    # Lyapunov exponent
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
