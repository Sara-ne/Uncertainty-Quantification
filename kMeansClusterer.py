import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KMeansClusterer:
    """
    Uses KMeans clustering to group images based on CLIP embeddings.
    """

    def determine_optimal_k(self, embeddings, k_range=(2, 10)):
        """
        Finds the best number of clusters using the silhouette score.
        """
        n_samples = embeddings.shape[0]
        min_k = max(2, min(k_range[0], n_samples - 1))
        max_k = min(k_range[1], n_samples - 1)

        if min_k >= max_k:
            raise ValueError(f"Not enough samples ({n_samples}) for clustering with k in range ({k_range})")

        best_k = min_k
        best_score = -1
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
            if len(set(kmeans.labels_)) < 2:
                continue  # Skip invalid k-values
            score = silhouette_score(embeddings, kmeans.labels_)
            if score > best_score:
                best_k, best_score = k, score
        return best_k

    def cluster_images(self, embeddings):
        """
        Clusters images using KMeans.
        :param embeddings: Image embeddings.
        :return: Cluster labels, optimal K, and KMeans model.
        """
        optimal_k = self.determine_optimal_k(embeddings)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42).fit(embeddings)
        return kmeans.labels_, optimal_k, kmeans
