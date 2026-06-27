import numpy as np
from src.chaos.logistic import LogisticMap, LogisticParameters


def compute_lyapunov(r=4.0, x0=0.3123, n_iter=1000):
    """
    Compute the Lyapunov exponent for the logistic map.
    λ = (1/N) * Σ log|f'(x)|
    For logistic map: f'(x) = r(1 - 2x)
    """
    params = LogisticParameters(r=r)
    lmap = LogisticMap(params)

    trajectory = lmap.simulate(x0=x0, iterations=n_iter)

    # Derivative of logistic map at each point
    derivatives = np.abs(r * (1 - 2 * trajectory))

    # Avoid log(0)
    derivatives = derivatives[derivatives > 0]

    lyapunov = np.mean(np.log(derivatives))

    return lyapunov
