import numpy as np


class ChaosMetrics:

    @staticmethod
    def variance(x):

        return np.var(x)

    @staticmethod
    def mean(x):

        return np.mean(x)

    @staticmethod
    def maximum(x):

        return np.max(x)

    @staticmethod
    def minimum(x):

        return np.min(x)

    @staticmethod
    def energy(x):

        return np.sum(np.square(x))

    @staticmethod
    def entropy(x, bins=50):

        hist, _ = np.histogram(x, bins=bins)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return -np.sum(prob * np.log2(prob))
