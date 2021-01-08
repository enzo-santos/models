import random
import statistics
from typing import List, Sequence, Callable

from classifiers.clustering.cluster import Cluster
from classifiers.clustering.datapoint import DataPoint
from classifiers.clustering.position import Position


def find(
        points: Sequence[DataPoint],
        *,
        n_clusters: int,
        strategy: str = 'mean',
        n_iterations: int = 500,
        should_fix_point: Callable[[DataPoint], bool] = None,
) -> List[Cluster]:
    """
    Encontra clusters em um conjunto de dados usando k-means ou k-medians.

    :param points: pontos do conjunto de dados.
    :param n_clusters: número de clusters a serem encontrados.
    :param strategy: define a estratégia de clusterização; 'mean' para k-means, 'median' para k-medians
    :param n_iterations: número de iterações do algoritmo; padrão 500
    :param should_fix_point: função que define quais pontos devem ser considerados fixos como centróides durante a
                                execução. Isso é útil se houver alguns centróides predefinidos no conjunto de dados que
                                devem ser levados em consideração. Se `None`, nenhum ponto será fixo.
    :raise ValueError: se o argumento 'strategy' for inválido ou se o número de pontos fixos for maior que o número de
                        clusters a serem encontrados.
    :return: os clusters encontrados.
    """
    if strategy not in ('mean', 'median'):
        raise ValueError(f'invalid strategy: {strategy}')
    strategy = statistics.mean if strategy == 'mean' else statistics.median

    fixed_points: List[DataPoint] = []
    non_fixed_points: List[DataPoint] = []
    if should_fix_point is None:
        non_fixed_points.extend(points)
    else:
        for point in points:
            if should_fix_point(point):
                fixed_points.append(point)
            else:
                non_fixed_points.append(point)

    n_fixed_points: int = len(fixed_points)
    n_remaining: int = n_clusters - n_fixed_points
    if n_remaining < 0:
        raise ValueError('the number of fixed points exceeds the number of clusters')

    starting_points: List[DataPoint] = fixed_points
    if n_remaining > 0:
        starting_points.extend(random.sample(non_fixed_points, n_remaining))

    centroids: List[Position] = [point.position for point in starting_points]
    clusters: List[Cluster] = []
    for _ in range(n_iterations):
        clusters = [Cluster(position) for position in centroids]
        for point in points:
            distances: List[float] = [point.distance_to(cluster.position) for cluster in clusters]
            i: int = distances.index(min(distances))
            clusters[i].add(point)

        for i, cluster in enumerate(clusters[n_fixed_points:], start=n_fixed_points):
            if len(cluster) == 1:
                continue

            x_m: float = strategy(point.position.x for point in cluster)
            y_m: float = strategy(point.position.y for point in cluster)
            centroids[i] = Position(x_m, y_m)

    return clusters
