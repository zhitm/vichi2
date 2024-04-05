import random

import numpy as np
import pandas as pd
from tabulate import tabulate

class Linear:

    def __init__(self, A, B, eps):
        self.A = A
        self.B = B
        self.eps = eps
        self.size = len(A)

    def get_h_g_form(self):
        d = self.get_d()
        d_inv = d.copy()
        e = np.eye(self.size)
        for i in range(self.size):
            d_inv[i][i] = 1 / d[i][i]

        H = e - np.matmul(d_inv, self.A)
        g = np.matmul(d_inv, self.B)
        return [H, g]

    def get_r(self, mat):
        r = mat.copy()
        for i in range(1, self.size):
            for j in range(i):
                r[i][j] = 0
        return r

    def get_d(self):
        d = np.array([[0 for _ in range(self.size)] for _ in range(self.size)], dtype='d')
        for i in range(self.size):
            d[i][i] = self.A[i][i]
        return d

    def get_l(self, mat):
        l = mat.copy()
        for i in range(self.size):
            for j in range(i, self.size):
                l[i][j] = 0
        return l

    def solve(self):
        n_it = 0
        H, g = self.get_h_g_form()
        prev_x = np.array([0 for _ in range(self.size)], dtype='d')
        curr_x = np.array([1 for _ in range(self.size)], dtype='d')
        while np.linalg.norm(prev_x - curr_x) > self.eps or n_it == 0:
            n_it += 1
            prev_x = curr_x
            curr_x = np.matmul(H, prev_x) + g
        return curr_x, n_it

    def seidel_solve(self):
        h, g = self.get_h_g_form()
        h_l = self.get_l(h)
        h_r = self.get_r(h)
        e = np.eye(self.size)
        prev_x = np.array([0 for _ in range(self.size)], dtype='d')
        curr_x = np.array([1 for _ in range(self.size)], dtype='d')
        n_it = 0
        while np.linalg.norm(prev_x - curr_x) > self.eps or n_it == 0:
            n_it += 1
            prev_x = curr_x
            curr_x = np.matmul(np.matmul(np.linalg.inv(e-h_l), h_r), prev_x) + np.matmul(np.linalg.inv(e-h_l), g)
        return curr_x, n_it


def get_big_mat(n):
    mat = np.array([[0 for _ in range(n)] for _ in range(n)], dtype='d')
    k = 10
    for i in range(n):
        for j in range(i, n):
            if i==j:
                mat[i][j] = random.randint(k*n+1, k*2*n)
            else:
                value = random.randint(-k, k)
                mat[i][j] = value
                mat[j][i] = value
    return mat

if (__name__ == "__main__"):
    mat_size = []
    eps_arr = []
    it1_arr = []
    it2_arr = []
    diff1 = []
    diff2 = []
    for i in range(5):
        eps = 10**(-5)
        t=300
        mat_size.append(t)
        eps_arr.append(eps)
        a = get_big_mat(t)
        b = np.array([1 for _ in range(t)], dtype='d')
        solver = Linear(a, b, eps)
        solution = np.linalg.solve(a, b)
        my_solution, it = solver.solve()

        my_solution_z, it_z = solver.seidel_solve()
        it1_arr.append(it)
        it2_arr.append(it_z)
        diff1.append(np.linalg.norm(solution-my_solution))
        diff2.append(np.linalg.norm(solution-my_solution_z))

    for i in range(5):
        eps = 10**(-10)
        t=300
        mat_size.append(t)
        eps_arr.append(eps)
        a = get_big_mat(t)
        b = np.array([1 for _ in range(t)], dtype='d')
        solver = Linear(a, b, eps)
        solution = np.linalg.solve(a, b)
        my_solution, it = solver.solve()

        my_solution_z, it_z = solver.seidel_solve()
        it1_arr.append(it)
        it2_arr.append(it_z)
        diff1.append(np.linalg.norm(solution-my_solution))
        diff2.append(np.linalg.norm(solution-my_solution_z))


    for i in range(5):
        eps = 10**(-15)
        t=300
        mat_size.append(t)
        eps_arr.append(eps)
        a = get_big_mat(t)
        b = np.array([1 for _ in range(t)], dtype='d')
        solver = Linear(a, b, eps)
        solution = np.linalg.solve(a, b)
        my_solution, it = solver.solve()

        my_solution_z, it_z = solver.seidel_solve()
        it1_arr.append(it)
        it2_arr.append(it_z)
        diff1.append(np.linalg.norm(solution-my_solution))
        diff2.append(np.linalg.norm(solution-my_solution_z))

    df = pd.DataFrame(
        {'Matix size': mat_size,
         'eps': eps_arr,
         'Iteration count: simple iteration': it1_arr,
         'Iteration count: Seidel': it2_arr,
         'diff1': diff1,
         'diff2': diff2,
         },
        )

    print(tabulate(df, headers='keys', tablefmt='psql'))