import numpy as np
from vichi2.task4.task4 import get_big_mat
import pandas as pd
from tabulate import tabulate
def get_rotation_mat(i, j, cos, sin, n):
    T = np.eye(n)
    T[i][i] = cos
    T[i][j] = -sin
    T[j][j] = cos
    T[j][i] = sin
    return T

def get_max(A):
    n = len(A)
    best_i = 0
    best_j = 1
    max = float("-inf")
    for i in range(0, n):
        for j in range(i+1, n):
            if abs(A[i][j]) > max:
                max = abs(A[i][j])
                best_j = j
                best_i = i
    return [best_i, best_j, max]
def rotate(A, i, j):
    n = len(A)
    x = A[i][j]
    y = A[i][i]-A[j][j]
    if y == 0:
        cos = 1/np.sqrt(2)
        sin = 1/np.sqrt(2)
        t = get_rotation_mat(i, j, cos, sin, n)
        return t.T.dot(A.dot(t))

    else:
        d = np.sqrt(4*x**2+y**2)
        cos = np.sqrt(0.5 * (1 + abs(y)/d))
        sin = np.sign(x*y) * np.sqrt(0.5 * (1-abs(y)/d))
        t = get_rotation_mat(i, j, cos, sin, n)
        return t.T.dot(A.dot(t))


def jacobi_rotations(A, eps):
    n = len(A)
    arr = []
    it = 0
    it_max = 1000*n
    run = True
    while run and it < it_max:
        it+=1
        run = True
        i, j, val = get_max(A)
        if abs(val) > eps:
            A = rotate(A, i, j)
        else:
            run = False

    for i in range(n):
        arr.append(A[i][i])
    return np.array(arr, dtype='d'), it

def get_diff(n, eps_pow):
    A = get_big_mat(n)
    eigen_values = jacobi_rotations(A, 10 ** (-eps_pow))
    it = eigen_values[1]
    my_eigs = np.array(sorted(eigen_values[0]), dtype='d')
    eigs = np.array(sorted(np.linalg.eig(A)[0]), dtype='d')
    diff = max(my_eigs - eigs, key=lambda x: abs(x))
    return diff, it
def compute_and_show():
    diffs = []
    sizes = []
    n_iter = []
    for i in range(1, 10):
        t = i*3
        sizes.append(t)
        eps = 5
        diff, it = get_diff(t, eps)
        diffs.append(diff)
        n_iter.append(it)

    for i in range(1, 10):
        t = i * 3
        sizes.append(t)
        eps = 10
        diff, it = get_diff(t, eps)
        diffs.append(diff)
        n_iter.append(it)

    for i in range(1, 10):
        t = i*3
        sizes.append(t)
        eps = 15
        diff, it = get_diff(t, eps)
        diffs.append(diff)
        n_iter.append(it)
    df = pd.DataFrame(
        {'Matix size': sizes,
         'Итераций' : n_iter,
         'diff' : diffs,
         },
    )

    print(tabulate(df, headers='keys', tablefmt='psql'))


if __name__=='__main__':
    compute_and_show()