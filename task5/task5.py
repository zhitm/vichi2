import numpy as np
from vichi2.task4.task4 import get_big_mat
import pandas as pd
from tabulate import tabulate

def pow_method(A, eps):
    n = len(A)
    # pow_A = A
    x_last = np.array([1. for _ in range(n)])
    x_new = x_last
    lamb = 0
    new_lamb = 1
    it = 0
    while abs(lamb-new_lamb) > eps:
        it+=1
        x_new /= np.linalg.norm(x_new)
        x_last = x_new
        x_new = A.dot(x_new)
        lamb = new_lamb
        new_lamb = x_new[0] / x_last[0]
    return new_lamb, it

def scalar_method(A, eps):
    n = len(A)
    x_last = np.array([1. for _ in range(n)])
    x_new = x_last
    lamb = 0
    new_lamb = 1
    it=0
    while abs(lamb-new_lamb) > eps:
        it+=1
        x_new /= np.linalg.norm(x_new)
        x_last = x_new
        x_new = A.dot(x_new)
        lamb = new_lamb
        new_lamb = A.dot(x_last).dot(x_new) / x_last.dot(x_new)
    return new_lamb, it


mat_size = []
eps_arr = []
eig_1 = []
eig_2 = []
it1 = []
it2 = []
np_eig = []
diff1 = []
diff2 = []
np.set_printoptions(precision=20, floatmode='fixed')
def process(eps_pow):
    np.set_printoptions(precision=20, floatmode='fixed')

    for i in range(1, 10):
        eps = 10 ** (-eps_pow)
        t = 3*i
        mat_size.append(t)
        eps_arr.append(eps)
        a = get_big_mat(t)
        pow, it_pow = pow_method(a, eps)
        scal, it_scal = scalar_method(a, eps)
        eig_1.append(pow)
        eig_2.append(scal)
        it1.append(it_pow)
        it2.append(it_scal)
        eig3=max(np.linalg.eig(a)[0])
        np_eig.append(max(np.linalg.eig(a)[0]))
        diff1.append(abs(pow-eig3))
        diff2.append(abs(scal-eig3))


process(5)
process(10)
process(15)
process(18)


pd.set_option("display.precision", 10)

df = pd.DataFrame(
    {'Matix size': mat_size,
     'Степенной метод' : eig_1,
     'Число итераций, степенной' : it1,
     'Скалярный метод' : eig_2,
     'Число итераций, скалярный': it2,
     'eps': eps_arr,
    'numpy': np_eig,
     'd1' : diff1,
     'd2': diff2
     },
)

print(tabulate(df, headers='keys', tablefmt='psql'))