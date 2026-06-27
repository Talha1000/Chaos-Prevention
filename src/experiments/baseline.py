from src.chaos.logistic import LogisticMap
from src.chaos.logistic import LogisticParameters
from src.utils.logger import ExperimentLogger
from src.chaos.metrics import ChaosMetrics

from src.visualization.plots import Plotter


params = LogisticParameters(r=4.0)

system = LogisticMap(params)

trajectory = system.simulate(
    x0=0.3123,
    iterations=5000
)

print("Mean:", ChaosMetrics.mean(trajectory))

print("Variance:", ChaosMetrics.variance(trajectory))

print("Entropy:", ChaosMetrics.entropy(trajectory))

Plotter.trajectory(trajectory)


logger = ExperimentLogger()

logger.save(
    "baseline",
    {
        "mean": ChaosMetrics.mean(trajectory),
        "variance": ChaosMetrics.variance(trajectory),
        "entropy": ChaosMetrics.entropy(trajectory),
        "iterations": len(trajectory),
        "r": 4.0,
        "x0": 0.3123
    }
)
