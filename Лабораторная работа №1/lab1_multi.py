import numpy as np
from matplotlib import pyplot as plt


def compute_hypothesis(X, theta):
    return X @ theta


def compute_cost(X, y, theta):
    m = X.shape[0]  # количество примеров в выборке
    # ВАШ КОД ЗДЕСЬ

    return 1 / (2 * m) * sum((compute_hypothesis(X, theta) - y) ** 2)

    # ==============


def gradient_descend(X, y, theta, alpha, num_iter):
    history = list()
    m = X.shape[0]  # количество примеров в выборке
    n = X.shape[1]  # количество признаков с фиктивным

    for i in range(num_iter):

        # ВАШ КОД ЗДЕСЬ

        theta_temp = theta.copy()
        for column in range(n):
            theta_temp[column] = theta_temp[column] - alpha * (compute_hypothesis(X, theta) - y).dot(X[:, column]) / m
        theta = theta_temp

        # =====================

        history.append(compute_cost(X, y, theta))
    return history, theta


def scale_features(X):
    # ВАШ КОД ЗДЕСЬ

    m = X.shape[0]  # количество примеров в выборке
    mean_X = 1 / m * sum(X)
    std = np.sqrt(1 / (m - 1) * sum((X - mean_X) ** 2)) # ср. кв. откл.

    X_0 = np.ones(m)[:, np.newaxis] # приведение к вектору-столбцу

    X_scale = np.concatenate([X_0, np.divide((X[:, 1:] - mean_X[1:]), std[1:])], axis=1)
    return X_scale
    # =====================


def normal_equation(X, y):
    # ВАШ КОД ЗДЕСь

    return np.linalg.pinv(X.T @ X) @ X.T @ y

    # =====================


def load_data(data_file_path):
    with open(data_file_path) as input_file:
        X = list()
        y = list()
        for line in input_file:
            *row, label = map(float, line.split(','))
            X.append([1] + row)
            y.append(label)
        return np.array(X, float), np.array(y, float)


X, y = load_data('lab1data2.txt')

history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.01, 1500)

plt.title('График изменения функции стоимости от номера итерации до нормализации')
plt.plot(range(len(history)), history)
plt.show()

X = scale_features(X)

history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.01, 1500)

plt.title('График изменения функции стоимости от номера итерации после нормализации')
plt.plot(range(len(history)), history)
plt.show()

theta_solution = normal_equation(X, y)
print(f'theta, посчитанные через градиентный спуск: {theta}, через нормальное уравнение: {theta_solution}')
