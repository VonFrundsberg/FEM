import numpy as np
import scipy.linalg as sp_linalg
import scipy.special as special
from new_misc import spectral as spectral
from new_misc import approx as approx
import numpy.polynomial.chebyshev as np_cheb
import matplotlib.pyplot as plt
import time as time


def reg_32_ndim(f, box, n):
    grids = [];
    weights = []
    for i in range(n.size):
        ws, ns = reg_32_wn(a=box[i, 0], b=box[i, 1], n=n[i], merged=True)
        grids.append(ns)
        weights.append(ws)
    mshgrid = np.array(np.meshgrid(*grids, indexing='ij'))
    # wghtgrid = np.array(np.outer(*weights))
    # print(*weights)
    wghtgrid = np.prod(np.array(np.ix_(*weights)))
    fx = f(mshgrid)
    nans = np.isnan(fx)
    fx[nans] = 0
    I = np.sum(np.array(fx) * np.array(wghtgrid))
    return I
def reg_32(f, a=-1, b=1, n=20, precalc=False):

        x = np.zeros(16)
        v = np.zeros(16)
        x[0] = 5.89955061432525 * 0.01
        x[1] = 3.08275706222781 * 0.1
        x[2] = 7.463707253079130 * 0.1
        x[3] = 1.35599372649466
        x[4] = 2.112943217346336
        x[5] = 2.98724149654594
        x[6] = 3.944798920961176
        x[7] = 4.950269202842798
        x[8] = 5.972123043117706
        x[9] = 6.989783558137742
        x[10] = 7.997673019512965
        x[11] = 8.999694932747039
        x[12] = 9.999979225211805
        x[13] = 1.099999938266130 * 10
        x[14] = 1.19999999946207 * 10
        x[15] = 13

        v[0] = 1.511076023874179 * 0.1
        v[1] = 3.459395921169090 * 0.1
        v[2] = 5.273502805146873 * 0.1
        v[3] = 6.878444094543021 * 0.1
        v[4] = 8.210319140034114 * 0.1
        v[5] = 9.218382875515803 * 0.1
        v[6] = 9.873027487553060 * 0.1
        v[7] = 1.018251913441155
        v[8] = 1.021933430349293
        v[9] = 1.012567983413513
        v[10] = 1.004052289554521
        v[11] = 1.000713413344501
        v[12] = 1.000063618302950
        v[13] = 1.000002486385216
        v[14] = 1.000000030404477
        v[15] = 1.000000000020760
        n = n
        A = 14
        h = 1 / (n + 2 * A - 1)
        points = np.append((b - a) * x * h + a, (b - a) * (A * h + np.arange(n) * h) + a)
        points = np.append(points, ((b - a) * (1 - x * h) + a)[::-1])
        S = f(points)

        if len(S.shape) > 1:
            return (b - a) * h * (
                    np.dot(S[:, 0: 16], v) + np.dot(S[:, 16 + n:32 + n], v[::-1]) + np.sum(S[:, 16: n + 16], axis=1))
        else:
            return (b - a) * h * (np.dot(S[0: 16], v) + np.dot(S[16 + n:32 + n], v[::-1]) + np.sum(S[16: n + 16]))
def reg_32_wn(a=-1, b=1, n=20, merged=False):
    x = np.zeros(16)
    w = np.zeros(16)
    A = 14
    #n = n - 2*(A + 2)
    h = 1 / (n + 2 * A - 1)
    x[0] = 5.89955061432525 * 0.01;
    w[0] = 1.511076023874179 * 0.1
    x[1] = 3.08275706222781 * 0.1;
    w[1] = 3.459395921169090 * 0.1
    x[2] = 7.463707253079130 * 0.1;
    w[2] = 5.273502805146873 * 0.1
    x[3] = 1.35599372649466;
    w[3] = 6.878444094543021 * 0.1
    x[4] = 2.112943217346336;
    w[4] = 8.210319140034114 * 0.1
    x[5] = 2.98724149654594;
    w[5] = 9.218382875515803 * 0.1
    x[6] = 3.944798920961176;
    w[6] = 9.873027487553060 * 0.1
    x[7] = 4.950269202842798;
    w[7] = 1.018251913441155
    x[8] = 5.972123043117706;
    w[8] = 1.021933430349293
    x[9] = 6.989783558137742;
    w[9] = 1.012567983413513
    x[10] = 7.997673019512965;
    w[10] = 1.004052289554521
    x[11] = 8.999694932747039;
    w[11] = 1.000713413344501
    x[12] = 9.999979225211805;
    w[12] = 1.000063618302950
    x[13] = 1.099999938266130 * 10;
    w[13] = 1.000002486385216
    x[14] = 1.19999999946207 * 10;
    w[14] = 1.000000030404477
    x[15] = 13;
    w[15] = 1.000000000020760
    weigths = np.append(w, np.ones(n))
    weigths = np.append(weigths, w[::-1])

    weigths = weigths
    left_points = (b - a) * x * h + a
    mid_points = (b - a) * (A * h + np.arange(n) * h) + a
    right_points = ((b - a) * (1 - x * h) + a)[::-1]

    left_weights = w * (b - a) * h
    mid_weights = np.ones(n) * (b - a) * h
    right_weights = w[::-1] * (b - a) * h
    if merged == False:
        return A, [left_points, mid_points, right_points], [left_weights, mid_weights, right_weights]
    if merged == True:
        points = np.array([left_points, mid_points, right_points])
        weights = np.array([left_weights, mid_weights, right_weights])
        points = np.concatenate(points).ravel()
        weights = np.concatenate(weights).ravel()
        return weights, points
def log_16(f=None, a=0, b=1, n=20, wn=False):
    v = np.zeros(15)
    u = np.zeros(15)

    x = np.zeros(8)
    w = np.zeros(8)
    A = 10
    B = 7
    h = 1 / (n + A + B - 1)
    v[0] = 8.371529832014113 * 0.0001;
    u[0] = 3.190919086626234 * 0.001
    v[1] = 1.239382725542637 * 0.01;
    u[1] = 2.423621380426338 * 0.01
    v[2] = 6.009290785739468 * 0.01;
    u[2] = 7.740135521653088 * 0.01
    v[3] = 1.805991249601928 * 0.1;
    u[3] = 1.704889420286369 * 0.1
    v[4] = 4.142832599028031 * 0.1;
    u[4] = 3.029123478511309 * 0.1
    v[5] = 7.964747731112430 * 0.1;
    u[5] = 4.652220834914617 * 0.1
    v[6] = 1.348993882467059;
    u[6] = 6.401489637096768 * 0.1
    v[7] = 2.073471660264395;
    u[7] = 8.051212946181061 * 0.1
    v[8] = 2.947904939031494;
    u[8] = 9.362411945698647 * 0.1
    v[9] = 3.928129252248612;
    u[9] = 1.014359775369075
    v[10] = 4.957203086563112;
    u[10] = 1.035167721053657
    v[11] = 5.986360113977494;
    u[11] = 1.020308624984610
    v[12] = 6.997957704791519;
    u[12] = 1.004798397441514
    v[13] = 7.999888757524622;
    u[13] = 1.000395017352309
    v[14] = 8.999998754306120;
    u[14] = 1.000007149422537

    x[0] = 9.919337841451028 * 0.01;
    w[0] = 2.528198928766921 * 0.1
    x[1] = 5.076592669645529 * 0.1;
    w[1] = 5.550158230159486 * 0.1
    x[2] = 1.184972925827278;
    w[2] = 7.852321453615224 * 0.1
    x[3] = 2.047493467134072;
    w[3] = 9.245915673876714 * 0.1
    x[4] = 3.007168911869310;
    w[4] = 9.839350200445296 * 0.1
    x[5] = 4.000474996776184;
    w[5] = 9.984463448413151 * 0.1
    x[6] = 5.000007879022339;
    w[6] = 9.999592378464547 * 0.1
    x[7] = 6.000000000000000;
    w[7] = 9.999999686258662 * 0.1

    points = np.append((b - a) * v * h + a, (b - a) * (A * h + np.arange(n) * h) + a)
    points = np.append(points, ((b - a) * (1 - x * h) + a)[::-1])

    weights = np.append(u, np.ones(n))
    weights = np.append(weights, w[::-1])

    if wn == True:
        return (b - a) * h * weights, points

    S = f(points)
    return (b - a) * h * np.dot(S, weights)
def trapz(f, a=-1, b=1, n=20):
    points = (b - a) * (np.arange(n) / (n - 1)) + a
    S = f(points)
    S[0] /= 2
    S[-1] /= 2
    return (b - a) / (n - 1) * np.sum(S, axis=0)
def __clenshawOld(self, f, a=-1, b=1, n=10, weight=None, precalc=False):
    if precalc == False:
        nodes = self.chebNodes(a, b, n)
        f = f(nodes)
    else:
        n = f.shape[0]
    f_cheb = np.real(self.chebTransform(f.T))
    if weight == None:
        i = np.arange(0, n + 1)
        M = -(np.cos(np.pi * i) + 1) / (i ** 2 - 1)
        Minfs = np.isinf(M)
        Mnans = np.isnan(M)
        M[Minfs] = 0
        M[Mnans] = 0
        q = (b - a) / 2
        return q * ((M.dot(f_cheb))).T
    if weight == 'x':
        q = (b - a) / 2
        p = (b + a) / 2
        i = np.arange(0, n + 1)
        M1 = (q * (-1 + np.cos(i * np.pi))) / (-4 + i ** 2)
        Minfs = np.isinf(M1)
        Mnans = np.isnan(M1)
        M1[Minfs] = 0
        M1[Mnans] = 0
        M2 = -(p * (1 + np.cos(i * np.pi))) / (-1 + i ** 2)
        Minfs = np.isinf(M2)
        Mnans = np.isnan(M2)
        M2[Minfs] = 0
        M2[Mnans] = 0
        M = M1 + M2
        return q * ((M.dot(f_cheb))).T
    else:
        w_cheb = weight(nodes)
        w_cheb = self.chebTransform(w_cheb)

        i = np.arange(0, n + 1)
        j = np.arange(0, n + 1)
        ii, jj = np.meshgrid(i, j)
        # t = time.time()
        # M = -(((-1 + ii**2 + jj**2)*(1 + np.cos(ii*np.pi)*np.cos(jj*np.pi)) + 2*ii*jj*np.sin(ii*np.pi)*np.sin(jj*np.pi))/((-1 + ii**2)**2 - 2*(1 + ii**2)*jj**2 + jj**4))
        # M = -(ii**2+jj**2-1)*((-1)**(ii+jj)+1)/(ii**4-2*ii**2*jj**2+jj**4-2*ii**2-2*jj**2+1)
        M = -(((1 + (-1) ** (ii + jj)) * (-1 + ii ** 2 + jj ** 2)) / (
                ii ** 4 + (-1 + jj ** 2) ** 2 - 2 * ii ** 2 * (1 + jj ** 2)))
        # print(time.time() - t)

        Minfs = np.isinf(M)
        Mnans = np.isnan(M)
        M[Minfs] = 0
        M[Mnans] = 0

        q = (b - a) / 2
        p = (b + a) / 2

        return q * ((w_cheb.T).dot(M.dot(f_cheb))).T
def clenshaw(f, a=-1, b=1, n=10, weight=None, precalc=False, chebForm=False):
    if precalc == False:
        nodes = spectral.chebNodes(a=a, b=b, n=n)
        f = f(nodes)
    else:
        n = f.shape[0]

    if chebForm == False:
        f_cheb = np.real(spectral.chebTransform(f))
    else:
        f_cheb = f

    if weight is None:
        i = np.arange(0, n)
        M = -((-1) ** i + 1) / (i ** 2 - 1)
        M[1] = 0
        q = (b - a) / 2
        # print(f_cheb.shape, M.shape)
        # np.dot(f_cheb, M)
        # return q*(f_cheb.T).dot(M)
        # if np.size(f_cheb.shape) > 1:
        # print(axis)
        # print(M.shape, f_cheb.shape)
        # return q*np.tensordot(M, f_cheb, axes=(0, axis))
        # else:
        return q * M.dot(f_cheb)
    else:
        if precalc == False:
            w_cheb = weight(nodes)
        else:
            w_cheb = weight
        if chebForm == False:
            w_cheb = spectral.chebTransform(w_cheb)
        else:
            w_cheb = weight

        F = spectral.cmul(f_cheb, w_cheb)

        n = F.shape[0]
        i = np.arange(0, n)
        M = -((-1) ** i + 1) / (i ** 2 - 1)
        M[1] = 0
        q = (b - a) / 2

        return q * ((np.tensordot(M, F, (0, 0)))).T
def fourier_inf(f, k, a=1, n=10):
    roots = special.roots_laguerre(n)[0]
    vv, rr = np.meshgrid(roots, np.arange(n))
    A = vv ** rr * np.exp(-vv)
    sol = sp_linalg.solve(A, special.factorial(np.arange(n)))
    w = 1j / k * sol
    x = 1j / k * roots
    g = lambda x: np.exp(1j * k * x) * f(x)
    return np.dot(w, g(a + x))
def tanh(f, h=1e-1, n=50):
    gg = lambda x: np.exp(np.pi / 2 * np.sinh(x))
    dg = lambda x: np.pi / 2 * np.cosh(x) * np.exp(0.5 * np.pi * np.sinh(x))
    x = np.linspace(-h * n, h * n, 2 * n)
    plt.plot(x, f(gg(x)) * dg(x))
    plt.show()
    return reg_32(lambda x: f(gg(x)) * dg(x), a=-h * n, b=h * n, n=2 * n)
def pClenshaw(f, g, a=-1, b=1, n=10, weight=None, precalc=False, chebForm=False):
    if precalc == True:
        if chebForm == False:
            f_cheb = np.real(spectral.chebTransform(f.T))
            g_cheb = np.real(spectral.chebTransform(g.T))
        else:
            f_cheb = f.copy()
            g_cheb = g.copy()
        nf = f_cheb.shape[0]
        ng = g_cheb.shape[0]

    if weight == None:
        i = np.arange(0, nf)
        j = np.arange(0, ng)
        ii, jj = np.meshgrid(i, j)
        M = -(((1 + (-1) ** (ii + jj)) * (-1 + ii ** 2 + jj ** 2)) / (
                ii ** 4 + (-1 + jj ** 2) ** 2 - 2 * ii ** 2 * (1 + jj ** 2)))
        Minfs = np.isinf(M)
        Mnans = np.isnan(M)
        M[Minfs] = 0
        M[Mnans] = 0
        q = (b - a) / 2
        return q * ((g_cheb.T).dot(M.dot(f_cheb))).T
    if weight == 'x':
        M1 = (2 * (-4 + ii ** 2 + jj ** 2) * (-1 + np.cos(ii * np.pi) * np.cos(jj * np.pi)) + 4 * ii * jj * np.sin(
            ii * np.pi) * np.sin(jj * np.pi)) / (2. * (ii ** 4 + (-4 + jj ** 2) ** 2 - 2 * ii ** 2 * (4 + jj ** 2)))
        M1 *= q
        Minfs = np.isinf(M1)
        Mnans = np.isnan(M1)
        M1[Minfs] = 0
        M1[Mnans] = 0
        M2 = -p * (((-1 + ii ** 2 + jj ** 2) * (1 + np.cos(ii * np.pi) * np.cos(jj * np.pi)) + 2 * ii * jj * np.sin(
            ii * np.pi) * np.sin(jj * np.pi)) / ((-1 + ii ** 2) ** 2 - 2 * (1 + ii ** 2) * jj ** 2 + jj ** 4))
        Minfs = np.isinf(M2)
        Mnans = np.isnan(M2)
        M2[Minfs] = 0
        M2[Mnans] = 0
        M = M1 + M2
        return q * ((g_cheb.T).dot(M.dot(f_cheb))).T
    else:
        if precalc == True:
            w_cheb = np.real(Fourier.chebTransform(weight.T))
            M = np.tensordot(self.M, w_cheb, axes=(2, 0))
            return q * ((g_cheb.T).dot(M.dot(f_cheb))).T
        w_cheb = np.real(spectral.chebTransform(weight.T))
        ii, jj, kk = np.meshgrid(i, i, i)
        M = ((-(ii ** 6 + (-jj ** 2 - kk ** 2 - 3) * ii ** 4 + (
                -jj ** 4 + (10 * kk ** 2 - 2) * jj ** 2 - kk ** 4 - 2 * kk ** 2 + 3) * ii ** 2 + (
                        jj ** 2 + kk ** 2 - 1) * (jj + kk + 1) * (jj - kk + 1) * (jj + kk - 1) * (
                        jj - kk - 1)) * np.cos(np.pi * kk) * np.cos(np.pi * jj) + 6 * (
                      ii ** 4 + (-0.2e1 / 0.3e1 * jj ** 2 - 0.2e1 / 0.3e1 * kk ** 2 - 0.2e1 / 0.3e1) * ii ** 2 - (
                      jj + kk + 1) * (jj - kk + 1) * (jj + kk - 1) * (jj - kk - 1) / 3) * kk * np.sin(
            np.pi * kk) * jj * np.sin(np.pi * jj)) * np.cos(np.pi * ii) - 2 * ii * kk * np.sin(np.pi * kk) * np.sin(
            np.pi * ii) * (ii ** 4 + (2 * jj ** 2 - 2 * kk ** 2 - 2) * ii ** 2 - 3 * jj ** 4 + (
                2 * kk ** 2 + 2) * jj ** 2 + kk ** 4 - 2 * kk ** 2 + 1) * np.cos(np.pi * jj) - 2 * ii * jj * np.sin(
            np.pi * ii) * (ii ** 4 + (-2 * jj ** 2 + 2 * kk ** 2 - 2) * ii ** 2 + jj ** 4 + (
                2 * kk ** 2 - 2) * jj ** 2 - 3 * kk ** 4 + 2 * kk ** 2 + 1) * np.sin(np.pi * jj) * np.cos(
            np.pi * kk) - ii ** 6 + (jj ** 2 + kk ** 2 + 3) * ii ** 4 + (
                     jj ** 4 + (-10 * kk ** 2 + 2) * jj ** 2 + kk ** 4 + 2 * kk ** 2 - 3) * ii ** 2 - (
                     jj ** 2 + kk ** 2 - 1) * (jj + kk + 1) * (jj - kk + 1) * (jj + kk - 1) * (jj - kk - 1)) / (
                    ii + jj + kk + 1) / (ii + jj + kk - 1) / (ii - jj + kk + 1) / (ii - jj + kk - 1) / (
                    ii + jj - kk + 1) / (ii + jj - kk - 1) / (ii - jj - kk + 1) / (ii - jj - kk - 1)
        Minfs = np.isinf(M)
        Mnans = np.isnan(M)
        M[Minfs] = 0
        M[Mnans] = 0
        self.M = M.copy()
        M = np.tensordot(M, w_cheb, axes=(2, 0))

        return q * ((g_cheb.T).dot(M.dot(f_cheb))).T
def clenshawTT(f, box, n, tt_tol=1e-6):
    grids = []
    for i in range(n.size):
        grids.append(spectral.chebNodes(a=box[i, 0], b=box[i, 1], n=n[i]))
    mshgrid = np.array(np.meshgrid(*grids, indexing='ij'))
    fx = f(mshgrid)
    nans = np.isnan(fx)
    fx[nans] = 0
    from new_misc import approx
    cores = approx.decay_smth(fx, tt_tol)
    Is = []
    for i in range(n.size):
        # print('hey')
        cores[i] = np.swapaxes(cores[i], 0, 1)
        shape1, shape2 = cores[i].shape[1], cores[i].shape[2]
        cores[i] = np.reshape(cores[i], [n[i], shape1 * shape2])
        # print(cores[i][0, :])
        I = clenshaw(f=cores[i], a=box[i, 0], b=box[i, 1], precalc=True)
        I = np.reshape(I, [shape1, shape2])
        Is.append(I)
    I = Is[0]
    for i in range(1, n.size):
        I = np.dot(I, Is[i])
    return I
def vectorizedClenshawTT(tt_fx, box):
    # grids = []
    # for i in range(n.size):
    #     grids.append(self.chebNodes(a=box[i, 0], b=box[i, 1], n=n[i]))
    # mshgrid = np.meshgrid(*grids, indexing='ij')
    # nans = np.isnan(fx)
    # fx[nans] = 0
    from new_misc import approx
    # cores = approx.decay_smth(fx, tt_tol)
    cores = tt_fx.copy()
    Is = []
    for i in range(len(tt_fx)):
        # print('hey')
        ni = cores[i].shape[1]
        cores[i] = np.swapaxes(cores[i], 0, 1)
        shape1, shape2 = cores[i].shape[1], cores[i].shape[2]
        cores[i] = np.reshape(cores[i], [ni, shape1 * shape2])
        I = clenshaw(f=cores[i], a=box[i, 0], b=box[i, 1], precalc=True)
        I = np.reshape(I, [shape1, shape2])
        Is.append(I)
    I = Is[0]
    for i in range(1, len(tt_fx)):
        I = np.dot(I, Is[i])
    return I
def gaussLegendre(f, a=-1, b=1, n=30):
    x, w = special.roots_legendre(n)
    q = (b - a) / 2
    p = (b + a) / 2
    return q * w, x * q + p
    return q * np.dot(f(x * q + p), w)
def chebChunkIntegration(f_cheb, partition, a=-1, b=1, n=9, axis=0, w_cheb=None):
    x, w = special.roots_legendre(n)
    q = np.diff(partition) / 2;
    p = (partition[: -1] + partition[1:]) / 2
    x, qq = np.meshgrid(x, q)
    grid = ((qq * x).T + p).T
    if w_cheb is None:
        vals = spectral.chebEval(f_cheb, grid, a, b)
        I = np.tensordot(vals, w, axes=((-1), (0)))
        return I * q
    else:
        F = spectral.cmul(f_cheb, w_cheb)
        vals = spectral.chebEval(F, grid, a, b)
        I = np.tensordot(vals, w, axes=((-1), (0)))
        return I * q

def clenshaw3D(f, g, w, a=-1, b=1, chebForm=True):
    if chebForm == False:
        f = self.chebTransform(f=f);
        g = self.chebTransform(f=g);
        w = self.chebTransform(f=w)

    nf = f.shape[0];
    ng = g.shape[0];
    nw = w.shape[0]
    i = np.arange(0, nf);
    j = np.arange(0, ng);
    k = np.arange(0, nw)
    ii, jj, kk = np.meshgrid(i, j, k)
    # M = ((ii**6 + (-1 + jj - kk)*(1 + jj - kk)*(-1 + jj + kk)*(1 + jj + kk)*(-1 + jj**2 + kk**2) - ii**4*(3 + jj**2 + kk**2) -
    #       ii**2*(-3 + jj**4 + 2*kk**2 + kk**4 + jj**2*(2 - 10*kk**2)))*(1 + np.cos((ii + jj + kk)*np.pi)))/\
    #      -  ((1 + ii - jj - kk)*(-1 + ii + jj - kk)*(1 + ii + jj - kk)*(-1 + ii - jj + kk)*(1 + ii - jj + kk)*
    #          (1 - ii + jj + kk)*(-1 + ii + jj + kk)*(1 + ii + jj + kk))
    # print(nw)
    # Minfs = np.isinf(M)
    # Mnans = np.isnan(M)
    # M[Minfs] = 0
    # M[Mnans] = 0
    M = ((-(ii ** 6 + (-jj ** 2 - kk ** 2 - 3) * ii ** 4 + (
            -jj ** 4 + (10 * kk ** 2 - 2) * jj ** 2 - kk ** 4 - 2 * kk ** 2 + 3) * ii ** 2 + (
                    jj ** 2 + kk ** 2 - 1) * (jj + kk + 1) * (jj - kk + 1) * (jj + kk - 1) * (
                    jj - kk - 1)) * np.cos(np.pi * kk) * np.cos(np.pi * jj) + 6 * (
                  ii ** 4 + (-0.2e1 / 0.3e1 * jj ** 2 - 0.2e1 / 0.3e1 * kk ** 2 - 0.2e1 / 0.3e1) * ii ** 2 - (
                  jj + kk + 1) * (jj - kk + 1) * (jj + kk - 1) * (jj - kk - 1) / 3) * kk * np.sin(
        np.pi * kk) * jj * np.sin(np.pi * jj)) * np.cos(np.pi * ii) - 2 * ii * kk * np.sin(np.pi * kk) * np.sin(
        np.pi * ii) * (ii ** 4 + (2 * jj ** 2 - 2 * kk ** 2 - 2) * ii ** 2 - 3 * jj ** 4 + (
            2 * kk ** 2 + 2) * jj ** 2 + kk ** 4 - 2 * kk ** 2 + 1) * np.cos(np.pi * jj) - 2 * ii * jj * np.sin(
        np.pi * ii) * (ii ** 4 + (-2 * jj ** 2 + 2 * kk ** 2 - 2) * ii ** 2 + jj ** 4 + (
            2 * kk ** 2 - 2) * jj ** 2 - 3 * kk ** 4 + 2 * kk ** 2 + 1) * np.sin(np.pi * jj) * np.cos(
        np.pi * kk) - ii ** 6 + (jj ** 2 + kk ** 2 + 3) * ii ** 4 + (
                 jj ** 4 + (-10 * kk ** 2 + 2) * jj ** 2 + kk ** 4 + 2 * kk ** 2 - 3) * ii ** 2 - (
                 jj ** 2 + kk ** 2 - 1) * (jj + kk + 1) * (jj - kk + 1) * (jj + kk - 1) * (jj - kk - 1)) / (
                ii + jj + kk + 1) / (ii + jj + kk - 1) / (ii - jj + kk + 1) / (ii - jj + kk - 1) / (
                ii + jj - kk + 1) / (ii + jj - kk - 1) / (ii - jj - kk + 1) / (ii - jj - kk - 1)
    Minfs = np.isinf(M)
    Mnans = np.isnan(M)
    M[Minfs] = 0
    M[Mnans] = 0
    M = np.tensordot(M, w, axes=(2, 0))
    q = (b - a) / 2
    return q * ((g.T).dot(M.dot(f)))
