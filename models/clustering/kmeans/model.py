import random
import statistics
from typing import List, Optional, Union

from models.clustering.cluster import _Cluster
from models.utils.linear_alg import Matrix, euclidean, Vector


class KMeans:
    def __init__(self, *, n_clusters: int = 3, n_iterations: int = 500, strategy: str = 'mean',
                 n_fixed_points: int = 0):
        if n_clusters < n_fixed_points:
            raise ValueError("the number of fixed samples exceeds the number of clusters")

        if strategy not in ('mean', 'median'):
            raise ValueError(f'invalid clustering strategy: {strategy}')

        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.strategy = strategy
        self.n_fixed_points = n_fixed_points

        self.labels_: Optional[Vector] = None
        self.cluster_centers_: Optional[Matrix] = None
        self._X: Optional[Matrix] = None

    def fit(self, X: Matrix) -> 'KMeans':
        self._X = X
        strategy = statistics.mean if self.strategy == 'mean' else statistics.median

        fixed_samples: Matrix = X[:self.n_fixed_points]
        n_remaining: int = self.n_clusters - self.n_fixed_points

        centroids: Matrix = fixed_samples
        if n_remaining > 0:
            centroids.extend(random.sample(X[self.n_fixed_points:], n_remaining))

        clusters: List[_Cluster] = []
        for _ in range(self.n_iterations):
            clusters = [_Cluster(center) for center in centroids]
            for sample in X:
                cluster = min(clusters, key=lambda c: euclidean(sample, c.center))
                cluster.add(sample)

            for i, cluster in enumerate(clusters[self.n_fixed_points:], start=self.n_fixed_points):
                if len(cluster) == 1:
                    continue

                centroid = [strategy(dimension) for dimension in zip(*cluster)]
                centroids[i] = centroid

        self.cluster_centers_ = [cluster.center for cluster in clusters]
        self.labels_ = []
        for sample in X:
            cluster_label = next(i for i, cluster in enumerate(clusters) if sample in cluster)
            self.labels_.append(cluster_label)

        return self

    def predict(self, X: Union[Vector, Matrix]) -> Union[float, Vector]:
        """
        Prediz os clusters dos dados fornecidos.

        :param X: as características das amostras a serem previstas.
        :return: o(s) índice(s) do(s) cluster(s) das amostras fornecidas.
        """
        if self._X is None:
            raise ValueError("you must call 'fit' before calling 'predict'")

        is_vector = all(isinstance(v, (float, int)) for v in X)
        X = [X] if is_vector else X

        y = []
        for sample in X:
            distances = [(i, euclidean(sample, center)) for i, center in enumerate(self.cluster_centers_)]
            cluster_label = min(distances, key=lambda x: x[1])[0]
            if is_vector:
                return cluster_label
            y.append(cluster_label)

        return y
