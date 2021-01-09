import random
import statistics
from typing import List

from models.classification.knn.model import KNearestClassifier
from models.classification.utils import split_into_train_test, k_fold
from models.utils.linear_alg import Matrix, Vector


def main():
    # Remover a linha abaixo caso queira gerar diferentes resultados a cada execução
    random.seed(0)

    # Lê o arquivo e preenche os vetores de características e classes
    X: Matrix = []
    y: Vector = []
    with open('iris.csv') as f:
        labels = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
        for line in f:
            # Considera apenas as colunas 3, 4 e 5
            _, _, f0, f1, c = line.strip().split(',')

            X.append([float(f0), float(f1)])
            y.append(labels.index(c))

    # Divide as amostras entre treino e teste
    X_train, y_train, X_test, y_test = split_into_train_test(X, y, test_sampling=0.33)

    clf: KNearestClassifier = KNearestClassifier(k=5)
    clf.fit(X_train, y_train)

    # Testa o algoritmo para uma amostra desconhecida
    i = random.choice(range(len(X_test)))
    f = X_test[i]
    c_true = y_test[i]
    c_pred = clf.predict(f)
    print(f'Amostra {f} é da classe {c_true}, foi classificada como {c_pred}')

    # Testa a acurácia do algoritmo para amostras desconhecidas
    accuracy = clf.score(X_test, y_test)
    print(f'Acurácia do classificador para amostras desconhecidas: {accuracy * 100:.2f}%')

    # Testa a acurácia do algoritmo para diferentes amostras desconhecidas
    accuracies: List[float] = k_fold(clf, X, y, n_folds=5)
    acc_mean = statistics.mean(accuracies)
    acc_stdev = statistics.stdev(accuracies)
    print(f'Acurácia do classificador para diferentes amostras: {acc_mean * 100:.2f}% (+/- {acc_stdev * 100:.2f}%)')


if __name__ == '__main__':
    main()
