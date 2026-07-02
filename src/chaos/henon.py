from dataclasses import dataclass
import numpy as np


@dataclass
class HenonParameters:
    a: float = 1.4
    b: float = 0.3


class HenonMap:
    """
    Henon Map — 2D chaotic system.
    x(t+1) = 1 - a*x(t)^2 + y(t)
    y(t+1) = b*x(t)

    Classic parameters a=1.4, b=0.3
    produce chaotic attractor.
    Lyapunov exponent ≈ 0.418
    """

    def __init__(self, params: HenonParameters):
        self.params = params

    def step(self, x: float, y: float):
        x_new = 1 - self.params.a * x ** 2 + y
        y_new = self.params.b * x
        return x_new, y_new

    def simulate(self, x0: float, y0: float, iterations: int):
        trajectory_x = np.zeros(iterations)
        trajectory_y = np.zeros(iterations)
        trajectory_x[0] = x0
        trajectory_y[0] = y0

        for i in range(1, iterations):
            trajectory_x[i], trajectory_y[i] = self.step(
                trajectory_x[i - 1],
                trajectory_y[i - 1]
            )

        return trajectory_x, trajectory_y
