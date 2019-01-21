#spectral differentiation on chebyshev nodes
#http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.473.7647&rep=rep1&type=pdf
import numpy as np
from misc import spectral as spectral
def cheb_diff(n, a=-1, b=1):
        x = spectral.chebNodes(n, a, b)
        X = np.ones([n, n], dtype=np.float)
        X = ((X.T)*x).T
        dX = X - X.T
        C = np.append([2], np.ones(n - 2))
        C = np.append(C, [2])
        C *= (-1)**np.arange(0, n)
        C = np.reshape(np.kron(C, 1/C), newshape=[n, n])
        D = C/(dX + np.eye(n))
        D = D - np.diag(np.sum(D, axis=1))
        return D

def TTdiff(u, axis, n=1, a=-1, b=1):
    cores = u.copy()
    D = cheb_diff(u[axis].shape[1], a, b)
    for i in range(n - 1):
        D = np.dot(D, D)

    i = axis
    shape = cores[i].shape
    cores[i] = np.swapaxes(cores[i], 0, 1)
    shape1, shape2 = cores[i].shape[1], cores[i].shape[2]
    cores[i] = np.reshape(cores[i], [shape[1], shape1 * shape2])
    I = np.dot(D, cores[i])

    I = np.reshape(I, [u[i].shape[1], shape1, shape2])
    I = np.swapaxes(I, 0, 1)
    cores[i] = I
    return cores
