from dataclasses import dataclass
import numpy as np


@dataclass
class LogisticParameters:
    r: float = 4.0


class LogisticMap:

    def __init__(self, params: LogisticParameters):
        self.params = params

    def step(self, x: float):

        return self.params.r * x * (1.0 - x)

    def simulate(
            self,
            x0: float,
            iterations: int):

        trajectory = np.empty(iterations)

        trajectory[0] = x0

        for i in range(1, iterations):

            trajectory[i] = self.step(
                trajectory[i - 1]
            )

        return trajectory
