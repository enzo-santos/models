import collections
import random

from models.clustering.kmeans.model import KMeans
from models.utils.linear_alg import Matrix, Vector


def main():
    LABELS = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
    # Remover a linha abaixo caso queira gerar diferentes resultados a cada execução
    random.seed(0)

    # Lê o arquivo e preenche os vetores de características e classes
    X: Matrix = []
    y: Vector = []
    with open('iris.csv') as f:
        for line in f:
            # Considera apenas as colunas 3, 4
            _, _, f0, f1, c = line.strip().split(',')
            X.append([float(f0), float(f1)])
            y.append(LABELS.index(c))

    # Aplica o algoritmo de clusterização sobre os dados
    clf: KMeans = KMeans(n_clusters=3).fit(X)

    counters = [collections.Counter(clf.labels_[i:i + 50]) for i in range(0, 150, 50)]
    print("Clusters encontrados:")
    for i, sample in enumerate(clf.cluster_centers_):
        n_points = sum(count for _, count in counters[i].items())
        print(f"    Cluster {i + 1}")
        print(f"    Centróide: {sample}")
        print(f"    Número de pontos: {n_points}")
        print(f"    Acurácia: {100 * max(count for _, count in counters[i].items()) / n_points:.2f}%")
        print()


if __name__ == '__main__':
    main()
