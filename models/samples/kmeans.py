import random
from typing import List

from models.clustering.cluster import Cluster
from models.clustering.datapoint import DataPoint
from models.clustering.kmeans.model import find
from models.clustering.position import Position
from models.utils.linear_alg import Matrix, Vector


def main():
    labels = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
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
            y.append(labels.index(c))

    # Transforma os dados em pontos de dado
    data_points: List[DataPoint] = []
    for i, (row, c) in enumerate(zip(X, y)):
        # O ID de cada ponto será "L_C", onde L é a sua linha do arquivo `iris.csv` e
        # C é a sua classe, sendo 0 para setosa, 1 para versicolor e 2 para virginica
        data_points.append(DataPoint(f"{i}_{c}", Position(row[0], row[1])))

    # Aplica o algoritmo de clusterização sobre os dados
    clusters: List[Cluster] = find(data_points, n_clusters=3)

    print("Clusters encontrados:")
    for i, cluster in enumerate(clusters, start=1):
        print(f"    Cluster {i}")
        print(f"    Centróide: {cluster.position}")

        # Verifica pelo ID de cada ponto se estão agrupados por classe
        # Cada chave de `classes` é a classe C de um ponto (0 para
        # setosa, 1 para versicolor e 2 para virginica) e cada valor
        # são os pontos da classe C presentes nesse cluster
        classes = {}
        for point in cluster:
            _, c = point.id.split("_")
            group = classes.setdefault(int(c), [])
            group.append(point)

        # A classe desse cluster é a do grupo que possui o maior número
        # de pontos da mesma classe.
        c = max(classes, key=lambda k: len(classes[k]))
        total = len(cluster)
        print(f"    Classe: {labels[c]}")
        print(f"    Número de pontos: {total}")
        print(f"    Acurácia: {100 * len(classes[c]) / total:.2f}%")
        print()


if __name__ == '__main__':
    main()
