import numpy as np
from scipy.stats import entropy


class SemanticEntropyCalculator:
    """
    Calculates entropy to measure the diversity of clusters.
    """

    def calculate_entropy(self, labels):
        label_counts = np.bincount(labels)
        probabilities = label_counts / len(labels)
        return entropy(probabilities, base=2)