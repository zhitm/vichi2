import random
from math import sqrt

import numpy as np
import pandas as pd
from tabulate import tabulate


# l2 norm
def get_norm(mat):
    acc = 0
    for row in mat:
        for el in row:
            acc += el * el
    return sqrt(acc)


def get_random_variation_A(matrix, delta):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            a = random.uniform(-delta, delta)
            matrix[i][j] += a
    return matrix


def get_random_variation_B(matrix, delta):
    for i in range(len(matrix)):
        a = random.uniform(-delta, delta)
        matrix[i] += a
    return matrix


# спектральное число
def spectral_number(matrix):
    inv_matrix = np.linalg.inv(np.mat(matrix)).tolist()
    return get_norm(matrix) * get_norm(inv_matrix)


# объемный критерий
def volume_number(matrix):
    number = 0
    for row in matrix:
        sum = 0
        for el in row:
            sum += el * el
        number += sqrt(sum)
    number /= np.linalg.det(matrix)
    return number


# угловой критерий
def ang_number(matrix):
    inv_matrix = np.linalg.inv(np.mat(matrix)).transpose().tolist()
    cond = 0
    for i in range(len(matrix)):
        v1 = matrix[i]
        v2 = inv_matrix[i]
        a = np.linalg.norm(v1)
        b = np.linalg.norm(v2)
        if a * b > cond:
            cond = a * b
    return cond


def solve_linear_system(A, B):
    a = np.array(A)
    b = np.array(B)
    return np.linalg.solve(a, b)


def compare_solutions(A, B, delta):
    x = solve_linear_system(A, B)
    A1 = get_random_variation_A(A, delta)
    B1 = get_random_variation_B(B, delta)
    x0 = solve_linear_system(A1, B1)
    return np.linalg.norm(np.mat(x) - np.mat(x0))


def get_hilbert_matrix(n):
    return [[1 / (i + j + 1) for j in range(n)] for i in range(n)]


def random_good_mat(n, min, max):
    mat = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        mat[i][i] = random.randint(min, max)
    for i in range(1, n):
        mat[i][i - 1] = random.randint(-1, 1)
        mat[i - 1][i] = random.randint(-1, 1)
    return mat


vols = []
spec = []
ang = []
diff_3 = []
diff_5 = []
diff_10 = []


def process(matrix):
    size = len(matrix)
    B = [1 for i in range(size)]
    d1 = 10 ** (-3)
    d2 = 10 ** (-5)
    d3 = 10 ** (-10)
    vols.append(volume_number(matrix))
    spec.append(spectral_number(matrix))
    ang.append(ang_number(matrix))
    diff_3.append(compare_solutions(matrix, B, d1))
    diff_5.append(compare_solutions(matrix, B, d2))
    diff_10.append(compare_solutions(matrix, B, d3))


if (__name__ == "__main__"):
    process(get_hilbert_matrix(4))
    process(get_hilbert_matrix(7))
    process(get_hilbert_matrix(10))

    min = 5
    max = 10
    process(random_good_mat(4, min, max))
    process(random_good_mat(7, min, max))
    process(random_good_mat(10, min, max))

    df = pd.DataFrame(
        {'Matr descr': ['Hilbert 4x4', 'Hilbert 7x7', 'Hilbert 10x10', '3diag 4x4', '3diag 7X7', '3diag 10x10'],
         'Volume number': vols,
         'Spectral number': spec,
         'Angular number': ang,
         'diff delta=10^(-3)': diff_3,
         'diff delta=10^(-5)': diff_5,
         'diff delta=10^(-10)': diff_10
         },
        )

    print(tabulate(df, headers='keys', tablefmt='psql'))
