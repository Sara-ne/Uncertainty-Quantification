import numpy as np
from scipy.stats import entropy

class SemanticEntropyCalculator:
    """
    Calculates entropy based on how images are distributed across clusters.
    High entropy = more diversity, Low entropy = less diversity.
    """
    def calculate_entropy(self, labels):
        label_counts = np.bincount(labels)
        probabilities = label_counts / len(labels)
        return entropy(probabilities, base=2)

