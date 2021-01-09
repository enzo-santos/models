import random
import unittest
import unittest.mock
from typing import List, Sequence, Optional

from models.clustering.cluster import Cluster
from models.clustering.datapoint import DataPoint
from models.clustering.kmeans.model import find
from models.clustering.position import Position


class ClusterTestCase(unittest.TestCase):
    @staticmethod
    def _mock_CentroidSelection(num_fixed_points: int):
        """
        Mocka os centróides iniciais do algoritmo.

        Este mock garante que cada centroide inicial fará parte de um grupo distinto de pontos no conjunto de dados
        original. Considere um algoritmo de clusterização que deve encontrar 3 clusters em um conjunto de dados. O
        conjunto de dados é composto por 3 grupos distintos de pontos: A, B e C. Em uma execução do algoritmo, dois
        centróides iniciais caem no grupo A e um cai no grupo C. Esses dois centróides no grupo A ficarão presos
        durante da execução e, ao término, haverão 3 clusters: um contendo ambos os grupos B e C, um contendo uma metade
        do grupo A e o outro contendo a outra metade do grupo A. Este mock evita isso escolhendo os centróides iniciais
        de cada grupo de pontos no conjunto de dados original.

        :param num_fixed_points: o número de pontos fíxos.
        :return: o mock simulando a função `random.sample`.
        """

        def mock(sequence: Sequence[DataPoint], n: int):
            points = []
            for i in range(num_fixed_points, n + num_fixed_points):
                grouped_points = [point for point in sequence if point.id.endswith(f"_{i}")]
                points.append(random.choice(grouped_points))
            return points

        return mock

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import os.path
        self._root, _ = os.path.split(__file__)
        self._root += os.sep

    def _get_path(self, filename: str) -> str:
        return self._root + filename

    def test_Init(self):
        self.assertRaises(
            ValueError,
            lambda: find(
                [DataPoint("a", Position(1, 2)), DataPoint("b", Position(3, 4)), DataPoint("c", Position(5, 6))],
                n_clusters=2,
                strategy='',
            ),
        )

        self.assertRaises(
            ValueError,
            lambda: find(
                [DataPoint("a", Position(1, 2)), DataPoint("b", Position(3, 4)), DataPoint("c", Position(5, 6))],
                n_clusters=2,
                should_fix_point=lambda x: True,
            )
        )

    def _test_ClusteringOnSample(
            self,
            filename: str,
            strategy: str,
            n_clusters: int,
            fixed_points_id: Optional[List[str]] = None,
    ) -> List[Cluster]:
        fixed_points_id = [] if fixed_points_id is None else fixed_points_id
        points: List[DataPoint] = []

        # Carrega o arquivo
        ranges = {}
        with open(self._get_path(filename)) as f:
            for i, line in enumerate(f):
                line = line.strip()
                x, y, c = line.split(",")
                x = float(x)
                y = float(y)
                c = int(c)

                # Calcula uma margem de erro aceitável para cada centroide
                class_ranges = ranges.setdefault(c, [float('+inf'), float('+inf'), float('-inf'), float('-inf')])
                class_ranges[0] = x if x < class_ranges[0] else class_ranges[0]  # x_min
                class_ranges[1] = y if y < class_ranges[1] else class_ranges[1]  # y_min
                class_ranges[2] = x if x > class_ranges[2] else class_ranges[2]  # x_max
                class_ranges[3] = y if y > class_ranges[3] else class_ranges[3]  # y_max

                points.append(DataPoint(f"{i}_{c}", Position(x, y)))

        # Aplica o algoritmo
        clusters: List[Cluster]
        with unittest.mock.patch("random.sample", side_effect=self._mock_CentroidSelection(len(fixed_points_id))):
            clusters = find(
                points,
                n_clusters=n_clusters,
                strategy=strategy,
                should_fix_point=None if not fixed_points_id else (lambda point: point.id in fixed_points_id),
            )

        # Verifica se os centroides gerados estão dentro da margem de erro aceitável
        for i, cluster in enumerate(clusters):
            position = cluster.position
            x_min, y_min, x_max, y_max = ranges[i]
            self.assertLessEqual(x_min, position.x)
            self.assertLessEqual(position.x, x_max)
            self.assertLessEqual(y_min, position.y)
            self.assertLessEqual(position.y, y_max)

        return clusters

    def test_KMeans_3Clusters(self):
        self._test_ClusteringOnSample("sample1.csv", strategy='mean', n_clusters=3)

    def test_KMedians_3Clusters(self):
        self._test_ClusteringOnSample("sample2.csv", strategy='median', n_clusters=3)

    def test_KMeans_2Clusters(self):
        self._test_ClusteringOnSample("sample3.csv", strategy='mean', n_clusters=2)

    def test_KMeans_WithOutlier(self):
        # Outlier na linha 1
        clusters = self._test_ClusteringOnSample("sample4.csv", strategy='mean', n_clusters=3)
        self.assertIn(Position(-10, -10), [cluster.position for cluster in clusters])
        cluster = next(cluster for cluster in clusters if cluster.position == Position(-10, -10))
        self.assertEqual(1, len(cluster))

    def test_KMeans_FixedPoints(self):
        # Ponto fixo na linha 1
        clusters = self._test_ClusteringOnSample("sample5.csv", strategy='mean', n_clusters=3, fixed_points_id=["0_0"])
        self.assertIn(Position(5, 7), [cluster.position for cluster in clusters])
