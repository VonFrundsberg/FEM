import numpy as np
import scipy.linalg as sp_linalg
import matplotlib.pyplot as plt
from misc import spectral
from misc import diff
import scipy.interpolate as sp_interp
import scipy.special as special

def decay_smth(f, tol=1e-6):
    shape = f.shape
    ls = len(shape)
    f = f.copy()
    sigma = 1
    psigma = 1
    cores = []
    for i in range(len(shape) - 1):

        f = np.reshape(f, [np.prod(shape[:-i - 1]), shape[-1 - i]*sigma])

        u, s, v = sp_linalg.svd(f, full_matrices=False)

        psigma = sigma
        sigma = max(1, np.size(s[np.abs(s) > tol]))

        f = np.dot(u[:, :sigma], np.diag(np.sqrt(s[:sigma])))
        v = np.dot(np.diag(np.sqrt(s[:sigma])), v[:sigma, :])

        v = np.reshape(v, [sigma, shape[ls - i - 1], psigma])

        cores.append(v)

        f = np.reshape(f, [*shape[:-1 - i], sigma])
    
    f = np.reshape(f, [1, *shape[:-1 - i], sigma])
    cores.append(f)
    return cores[::-1]
    # for i in range(len(cores)):
    #     print(cores[i].shape)
def matrixTT(A, shape, tol=1e-6, f=None):
    # plt.spy(A)
    # plt.show()
    # shape = shape[::-1]
    A = np.reshape(A, [*shape, *shape])
    # odd = np.arange(2*shape.size)[::2]
    # even = np.arange(2*shape.size)[1::2]
    first = np.arange(shape.size)
    last = np.arange(shape.size, 2*shape.size)
    newAxes = np.zeros(2*shape.size)
    newAxes[::2] = first; newAxes[1::2] = last
    newAxes = np.array(newAxes, dtype=np.int)
    A = np.transpose(A, newAxes)
    A = np.reshape(A, shape**2)
    # A = np.reshape(A, [12, 12])
    # A = np.reshape(A, [shape[0]**2*shape[1]**2, shape[2]**2])
    # print(A.shape)
    A = decay_smth(A, tol)
    for i in range(len(A)):
        a = A[i].shape[0]; b = A[i].shape[2]
        A[i] = np.reshape(A[i], [a, shape[i], shape[i], b])
    # for it in A:
    #     print(it.shape)
    # u, s, v = sp_linalg.svd(A, full_matrices=False)
    # print(s)
    # sigma = np.size(s[s > tol])
    # A = u[:, :sigma]; v = np.dot(np.diag(s[:sigma]), v[:sigma, :])
    # v = np.reshape(v, [sigma, shape[1], shape[1]])
    # A = np.reshape(A, [sigma, shape[0], shape[0]])
    # print(v.shape, A.shape)
    # meh = np.outer(np.dot(A[0], F[0]), np.dot(v[0], F[1]))+\
    #     np.outer(np.dot(A[1], F[0]), np.dot(v[1], F[1]))# + np.outer(A[1], v[1])
    # print(v.shape, F[1].shape)
    # res1 = np.squeeze(np.einsum('ijk,mjl->mjl', A, F[0]))
    # res2 = np.squeeze(np.einsum('ijk,mjl->mjl', v, F[1]))
    # res = np.outer(res1[:, 0], res2[0, :]) + np.outer(res1[:, 1], res2[1, :])
    # print('first')
    # print(v[0])
    # print('and then')
    # print(v[1])
    # res1 = np.squeeze(np.dot(A[0].T, F[0]))
    # res2 = np.squeeze(np.dot(v[0], F[1]))

    # res3 = np.squeeze(np.dot(A[1].T, F[0]))
    # res4 = np.squeeze(np.dot(v[1], F[1]))
    # plt.plot(res1)
    # plt.show()
    # r1 = np.outer(res1[:, 0], res2[:, 0]) + np.outer(res1[:, 1], res2[:, 1])
    # r2 = np.outer(res3[:, 0], res4[:, 0]) + np.outer(res3[:, 1], res4[:, 1])
    # print(r1.shape, r2.shape)
    # print(res.shape)
    # print(meh.shape)
    # plt.imshow(r1 + r2)
    # plt.show()
    return A
import time as time
def vecTTMul(A, u):
    Y = []
    for k in range(len(A)):
        M = A[k]; x = u[k]
        y = []
        for i in range(M.shape[1]):
            G = np.sum(M[:, np.newaxis, i, :, :, np.newaxis] * x[np.newaxis, :, :, np.newaxis, :], axis=2)
            G = np.reshape(G, [G.shape[0]*G.shape[1], G.shape[2]*G.shape[3]])
            y.append(G)
        y = np.array(y)
        y = np.swapaxes(y, 0, 1)
        Y.append(y)
    return Y
def vecRound(u, tol=1e-6):
    size = len(u)
    for i in range(size - 1, 1, -1):
        a, n, b = u[i].shape
        # R, Q = sp_linalg.rq(np.reshape(u[i], [a, n*b]))
        R, Q = sp_linalg.qr(np.reshape(u[i], [a, n * b]))
        u[i] = np.reshape(Q, [a, n, 1])
        # print('r', R.shape)
        # print(u[i - 1].shape, Q.shape, R.shape)
        u[i - 1] = np.tensordot(u[i - 1], R, axes=(2, 0))
        # print(Q.shape, R.shape)
    for i in range(size - 1):
        a, n, b = u[i].shape
        # print(a, n, b)
        U, S, V = sp_linalg.svd(np.reshape(u[i], [a*n, b]))
        # print(i, S)
        sigma = max(1, np.size(S[S > tol]))
        u[i] = U[:, :sigma]; V = np.dot(np.diag(S[:sigma]), V[:sigma, :])
        # print(V.shape, u[i + 1].shape)
        u[i + 1] = np.tensordot(V, u[i + 1], axes=(1, 0))
        u[i] = np.reshape(u[i], [a, n, sigma])

    # a, n, b = u[-1].shape
    # U, S, V = sp_linalg.svd(np.reshape(u[-1], [a * n, b]))
    # sigma = np.size(S[S > tol])
    # print(a, n, b)
    # u[i] = U[:, :sigma];
    # V = np.dot(np.diag(S[:sigma]), V[:sigma, :])
    # # print(V.shape, u[i + 1].shape)
    # u[i + 1] = np.tensordot(V, u[i + 1], axes=(1, 0))
    # u[i] = np.reshape(u[i], [a, n, sigma])
    return u
    # for it in u:
    #     print(it.shape)
def sum(u, v):
    c = []
    for i in range(len(u)):
        if i == 0:
            core = np.hstack(u[i], v[i])
        elif i == len(u) - 1:
            core = np.hstack(u[i], v[i])
        else:
            core = sp_linalg.block_diag([u[i], v[i]])
    c.append(core)
def TTtoVec(u):
    G = u[0]
    for k in range(len(u) - 1):
        G = np.tensordot(G, u[k + 1], axes=1)
    G = np.squeeze(G)
    return G
def spectralTTInterpolation(v, box, x):
    xx = x.copy()
    cores = v.copy()
    x_shape = xx.shape
    xx = np.reshape(xx, [len(v), int(np.prod(x_shape)/len(v))])
    Is = []
    for i in range(len(v)):
        shape = cores[i].shape
        cores[i] = np.swapaxes(cores[i], 0, 1)
        shape1, shape2 = cores[i].shape[1], cores[i].shape[2]
        cores[i] = np.reshape(cores[i], [shape[1], shape1 * shape2])

        nodes = spectral.chebNodes(shape[1], box[i, 0], box[i, 1])

        I = sp_interp.barycentric_interpolate(nodes, cores[i], xx[i])

        I = np.reshape(I, [xx.shape[1], shape1, shape2])
        I = np.swapaxes(I, 0, 1)
        Is.append(I)

    G = np.reshape(Is[0], [Is[0].shape[1], Is[0].shape[2]])
    for i in range(len(Is) - 1):
        G = np.einsum('jk,kjm->jm', G, Is[i + 1])
    G = np.squeeze(G)
    G = np.reshape(G, x_shape[1:])
    return G
def showCores(v):
    for it in v:
        print(it.shape)
def mul(u, v):
    uv = []
    for i in range(len(u)):
        A = u[i]; B = v[i]
        C = []
        for i in range(A.shape[1]):
            C.append(np.kron(A[:, i, :], B[:, i, :]))
        C = np.array(C)
        C = np.swapaxes(C, 0, 1)
        uv.append(C)
    return uv
def TTmerge(u, v):
    return u + v
def matrixTT1d(A_list):
    res = []
    for it in A_list:
        res.append(np.reshape(it, [1, *it.shape, 1]))
    return res
def meshgrid(*arg1):
    return np.meshgrid(*arg1, indexing='ij')
def ttFourierLegendre(fx, k, a=-1, b=1, axis=0):
    cores = fx.copy()

    i = axis
    shape = cores[i].shape
    cores[i] = np.swapaxes(cores[i], 0, 1)
    shape1, shape2 = cores[i].shape[1], cores[i].shape[2]
    cores[i] = np.reshape(cores[i], [shape[1], shape1 * shape2])
    t = time.time()
    print(cores[i].shape)
    I = spectral.legendreFT(cores[i], y=k, a=a, b=b)
    print(time.time() - t)

    I = np.reshape(I, [k.size, shape1, shape2])
    I = np.swapaxes(I, 0, 1)
    cores[i] = I

    return cores

# x = spectral.chebNodes(15, a=-1, b=1)
# grid = meshgrid(x, x, x, x, x, x)
# f = lambda x, x1, y, y1, z, z1: np.exp(-np.sqrt((x - x1)**2 + (y - y1)**2 + (z - z1)**2))#np.exp(-x**2)*np.exp(-y**2)#*np.exp(-z**2)
# fx = f(*grid)
# tt_fx = decay_smth(fx)
# print('first')
# for it in tt_fx:
#     print(it.shape)
#
# print('next')
# x = spectral.chebNodes(15, a=-1, b=1)
# grid = meshgrid(x, x, x)
# f = lambda x, y, z: np.exp(-np.sqrt((x)**2 + (y)**2 + (z)**2))#np.exp(-x**2)*np.exp(-y**2)#*np.exp(-z**2)
# fx = f(*grid)
# tt_fx = decay_smth(fx)
# for it in tt_fx:
#     g = lambda x, y: spectral.chebEval(spectral.chebTransform(it), x - y)
#     g(*meshgrid(x, x), )
# for it in tt_fx:
#     print(it.shape)

# plt.imshow(fx)
# plt.show()

# import scipy.special as special
# d = 12
# r = spectral.chebNodes(2**d)
# f = lambda x: np.exp(-x**2)*np.cos((20*(x - 0.5)**2) + 0.5)# + np.sqrt(1 - x**2)*np.exp(-2*x)
# f = lambda x: special.j0(10*(x + 1))
# plt.plot(r, f(r))
# plt.show()
# for i in range(4, 12):
#     d = i
#     r = spectral.chebNodes(2**d)
#     # f = lambda x: np.exp(-x**2)*np.cos((20*(x - 0.5)**2) + 0.5) + np.sqrt(1 - x**2)*np.exp(-2*x)
#     fx = f(r)
#     cfx = spectral.chebTransform(fx)
#
#     acfx = np.abs(cfx)
#     dimcfx = np.reshape(acfx, np.array(2*np.ones(d), np.int))
#     print('d = ', d)
#     # print(dimcfx.shape)
#     tt = decay_smth(dimcfx)
#     for it in tt:
#         print(it.shape)
#     estimated = TTtoVec(tt)
#
#
#     plt.loglog(np.abs(estimated.flatten()), '--')
#     plt.loglog(np.abs(cfx))
#     plt.show()
# import scipy.special as spec
# # r = spectral.chebNodes(50, 0, 1)
# # # z = spectral.chebNodes(50, 0, 10)
# # # t = spectral.chebNodes(50, 0, 2*np.pi)
# # # f = lambda r, z, t: np.exp(-(r**2 + z**2))*np.sin(t/2)
# # f = lambda x, k: spec.i0(x*k)
# #
# # rr, kk = meshgrid(r, r)
# # fx = f(rr, kk)
# # # print(fx)
# # ttfx = decay_smth(fx)
# # for it in ttfx:
# #     print(it.shape)
# # cfx = spectral.chebTransform(fx)
