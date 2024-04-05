import numpy as np
import pandas as pd
from tabulate import tabulate

from vichi2.task1.cond_nums import volume_number, get_hilbert_matrix, random_good_mat


def get_rot_m(mat, i, j):
    n = len(mat)
    rot_mat = np.array([[0 for _ in range(n)] for _ in range(n)], dtype='d')
    for a in range(n):
        rot_mat[a][a] = 1

    k = mat[i][i]
    l = mat[j][i]
    c = k / np.sqrt(k * k + l * l)
    s = l / np.sqrt(k * k + l * l)
    rot_mat[i][i] = c
    rot_mat[j][j] = c
    rot_mat[i][j] = s
    rot_mat[j][i] = -s
    return rot_mat


# returns R и Q^TB
def transform_equation(mat, b):
    n = len(mat)
    r = mat.copy()

    for i in range(n):
        for j in range(i + 1, n):
            rot_m = get_rot_m(r, i, j)
            r = np.matmul(rot_m, r)
            b = np.matmul(rot_m, b)
    return r, b


def compare_solutions(A, B):
    x1 = solve_by_qr(A, B)
    x2 = np.linalg.solve(A, B)
    diff = x1 - x2
    return np.linalg.norm(diff)



def solve(A, B):
    n = len(A)
    ans = np.array([0 for _ in range(n)], dtype='d')
    for i in range(n):
        if abs(A[n - i - 1][n - i - 1]) < 10 ** (-18):
            ans[n - i - 1] = 1
        else:
            ans[n - i - 1] = B[n - i - 1] / A[n - i - 1][n - i - 1]
        for j in range(0, n - i - 1):
            B[j] -= ans[n - i - 1] * A[j][n - i - 1]
    return ans


def solve_by_qr(A, B):
    A1, B1 = transform_equation(A, B)
    return solve(A1, B1)


arr_a = []
arr_b = []
arr_qr_solutions = []
arr_solutions = []
arr_diff = []
vols = []
vols_after = []


def fill_arr(A, B):
    arr_a.append(np.array2string(A))
    arr_b.append(np.array2string(B))
    arr_qr_solutions.append(solve_by_qr(A, B))
    arr_solutions.append(np.linalg.solve(A, B))
    arr_diff.append(compare_solutions(A, B))
    vols.append(volume_number(A))
    vols_after.append(volume_number(transform_equation(A, B)[0]))

if (__name__ == "__main__"):


    spec = []
    ang = []
    A = np.array([[1, 2, 3],
                  [3, 4, 5], [0, 2, 7]], dtype='d')

    B = np.array([1, 1, 1], dtype='d')
    fill_arr(A, B)

    a5 = get_hilbert_matrix(5)
    b5 = np.array([1 for i in range(5)], dtype='d')
    fill_arr(a5, b5)

    a7 = get_hilbert_matrix(7)
    b7 = np.array([1 for i in range(7)], dtype='d')
    fill_arr(a7, b7)

    good = random_good_mat(8, -10, 10)
    b8 = np.array([1 for i in range(8)], dtype='d')
    fill_arr(good, b8)

    df = pd.DataFrame(
        {'Matr descr': ['Mat 3x3', 'Hilbert 5x5', 'Hilbert 7x7', '3diag 8x8'],
         'A': arr_a,
         'B': arr_b,
         'QR solution': arr_qr_solutions,
         'Numpy solution': arr_solutions,
         'Norm of diff of solutions': arr_diff,
         'Volume number': vols,
         'Volume number after': vols_after
         },
        )

    print(tabulate(df, headers='keys', tablefmt='psql'))


