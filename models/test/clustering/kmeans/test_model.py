import collections
import random
import unittest
import unittest.mock

from models.clustering.kmeans.model import KMeans
from models.utils.linear_alg import Matrix, Vector


class ClusterTestCase(unittest.TestCase):
    @staticmethod
    def _mock_CentroidSelection(y: Vector, num_fixed_points: int):
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

        def mock(X: Matrix, n: int):
            points = []
            for i in range(num_fixed_points, n + num_fixed_points):
                grouped_samples = [sample for j, sample in enumerate(X) if y[num_fixed_points:][j] == i]
                points.append(random.choice(grouped_samples))

            return points

        return mock

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        import os.path
        self._root, _ = os.path.split(__file__)
        self._root += os.sep

    def _get_path(self, filename: str) -> str:
        return self._root + filename

    def _test_ClusteringOnSample(
            self,
            filename: str,
            strategy: str,
            n_clusters: int,
            n_fixed_points: int = 0,
    ) -> KMeans:
        # Carrega o arquivo
        X: Matrix = []
        y: Vector = []
        ranges = {}
        with open(self._get_path(filename)) as f:
            for i, line in enumerate(f):
                x_, y_, c = line.strip().split(",")
                x_ = float(x_)
                y_ = float(y_)
                c = int(c)

                # Calcula uma margem de erro aceitável para cada centroide
                class_ranges = ranges.setdefault(c, [float('+inf'), float('+inf'), float('-inf'), float('-inf')])
                class_ranges[0] = min(x_, class_ranges[0])  # x_min
                class_ranges[1] = min(y_, class_ranges[1])  # y_min
                class_ranges[2] = max(x_, class_ranges[2])  # x_max
                class_ranges[3] = max(y_, class_ranges[3])  # y_max

                X.append([x_, y_])
                y.append(c)

        # Aplica o algoritmo
        clf: KMeans = KMeans(n_clusters=n_clusters, strategy=strategy, n_fixed_points=n_fixed_points)
        with unittest.mock.patch("random.sample", side_effect=self._mock_CentroidSelection(y, n_fixed_points)):
            clf.fit(X)

        # Verifica se os centroides gerados estão dentro da margem de erro aceitável
        for i, centroid in enumerate(clf.cluster_centers_):
            x_, y_ = centroid
            x_min, y_min, x_max, y_max = ranges[i]
            self.assertLessEqual(x_min, x_)
            self.assertLessEqual(x_, x_max)
            self.assertLessEqual(y_min, y_)
            self.assertLessEqual(y_, y_max)

        return clf

    def test_Init(self):
        self.assertRaises(ValueError, lambda: KMeans(strategy=''))
        self.assertRaises(ValueError, lambda: KMeans(n_clusters=4, n_fixed_points=5))

    def test_KMeans_3Clusters(self):
        self._test_ClusteringOnSample("sample1.csv", strategy='mean', n_clusters=3)

    def test_KMedians_3Clusters(self):
        self._test_ClusteringOnSample("sample2.csv", strategy='median', n_clusters=3)

    def test_KMeans_2Clusters(self):
        self._test_ClusteringOnSample("sample3.csv", strategy='mean', n_clusters=2)

    def test_KMeans_WithOutlier(self):
        # Outlier na linha 1
        clf: KMeans = self._test_ClusteringOnSample("sample4.csv", strategy='mean', n_clusters=3)
        self.assertIn([-10, -10], [centroid for centroid in clf.cluster_centers_])
        self.assertTrue(any(count == 1 for _, count in collections.Counter(clf.labels_).items()))

    def test_KMeans_FixedPoints(self):
        # Ponto fixo na linha 1
        clf = self._test_ClusteringOnSample("sample5.csv", strategy='mean', n_clusters=3, n_fixed_points=1)
        self.assertIn([5, 7], [centroid for centroid in clf.cluster_centers_])

    def test_Prediction(self):
        clf: KMeans = KMeans()
        self.assertRaises(ValueError, lambda: clf.predict([[1, 2], [3, 4]]))

        clf.fit([[1, 1], [5, 5], [10, 10]])
        c0, c1, c2 = clf.predict([[1, 1], [5, 5], [10, 10]])
        self.assertEqual({0, 1, 2}, {c0, c1, c2})

        self.assertEqual(c0, clf.predict([1, 1]))
        self.assertEqual(c1, clf.predict([5, 5]))
        self.assertEqual(c2, clf.predict([10, 10]))

        self.assertEqual(c0, clf.predict([0, 0]))
        self.assertEqual(c1, clf.predict([4, 4]))
        self.assertEqual(c2, clf.predict([11, 11]))
