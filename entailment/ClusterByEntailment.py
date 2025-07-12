import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster


class ClusterByEntailment:
    """
    Clusters images based on textual entailment of their descriptions.
    Uses hierarchical clustering.
    """
    def __init__(self, entailment_checker, threshold=0.7):
        self.entailment_checker = entailment_checker
        self.threshold = threshold

    def cluster_images(self, descriptions):
        """
        Clusters images based on how much their descriptions entail each other.
        :param descriptions: List of text descriptions.
        :return: List of cluster labels.
        """
        num_images = len(descriptions) # count number of images
        entailment_matrix = np.zeros((num_images, num_images)) # create empty square matrix

        # Compute pairwise entailment scores
        for i in range(num_images):
            for j in range(i + 1, num_images):
                # Check both directions of entailment
                score_ij = self.entailment_checker.check_entailment(descriptions[i], descriptions[j])
                score_ji = self.entailment_checker.check_entailment(descriptions[j], descriptions[i])

                # Only store scores if both directions of entailment are positive
                if score_ij > 0 and score_ji > 0:
                    entailment_matrix[i, j] = score_ij
                    entailment_matrix[j, i] = score_ji  # Ensure symmetry

        # Perform hierarchical clustering (using 1 - entailment scores)
        linkage_matrix = linkage(1 - entailment_matrix, method='average')
        labels = fcluster(linkage_matrix, t=self.threshold, criterion='distance')
        return labels


def print_cluster_descriptions(descriptions, labels):
    """
    Prints the text descriptions of images grouped by their cluster labels.
    :param descriptions: List of image descriptions.
    :param labels: Cluster labels for each description.
    """
    cluster_dict = {} # store clusters

    # Loop through descriptions and their corresponding labels
    for desc, label in zip(descriptions, labels):
        if label not in cluster_dict:
            cluster_dict[label] = [] # If not, create empty list for label
        cluster_dict[label].append(desc) # append description (desc) to the list of label

    # Loop through each cluster (cluster) and its corresponding descriptions (desc)
    # Print cluster number and the descriptions that belong to that cluster
    for cluster, descs in cluster_dict.items():
        print(f"\nCluster {cluster}:")
        for d in descs:
            print(f"  - {d}")