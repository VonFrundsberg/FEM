import numpy as np
import scipy.special as special
import scipy.linalg as sp_lin
import scipy.interpolate as sp_interp
import numpy.linalg as np_lin
import matplotlib.pyplot as plt
import time as time
import scipy.fftpack as dft
import numpy.polynomial.chebyshev as np_cheb
import scipy.optimize as sp_opt
from scipy.optimize import linprog


def __chebTransformOld(self, f, axis=0):
    n = f.shape[axis]
    iii = np.arange(0, n)
    nn1, nn2 = np.meshgrid(iii[::-1], iii)
    T = (np.cos(nn2 * nn1 * np.pi / (n - 1)))
    f[0] /= 2; f[-1] /= 2
    # np.set_printoptions(precision=4, suppress=True)
    a = 2 / (n - 1) * T.dot(f)
    a[0] /= 2; a[-1] /= 2
    f[0] *= 2; f[-1] *= 2
    return a

def chebTransform(f, axis=0):
    n = f.shape[axis]
    a = 1 / (n - 1) * dft.dct(f[::-1], type=1, axis=axis)
    a[0] /= 2; a[-1] /= 2
    return a

def bary(f, x, a=-1, b=1):
    if len(f.shape) > 1:
        cx = chebNodes(f.shape[0], a=a, b=b)
    else:
        cx = chebNodes(f.size, a=a, b=b)
    args0 = np.argwhere((x < a) | (x > b))
    res = sp_interp.barycentric_interpolate(cx, f, x, axis=0)
    res[args0] = 0
    return res


def chebNodes(n, a=-1, b=1):
    nodes = (np.cos(np.arange(0, n) * np.pi / (n - 1))[::-1]) * (b - a) / 2
    nodes -= nodes[0]
    nodes += a
    return nodes


def laguerreRoots(self, n):
    return special.roots_laguerre(n)


def hermiteRoots(self, n):
    return special.roots_hermite(n)


def lagrangeToLegendre(t, axis=0):
    n = t.shape[axis]
    points = chebNodes(n=n)
    LegendreM = np.zeros([n, n])
    for i in range(n):
        LegendreM[:, i] = special.eval_legendre(i, points)
    if axis == 0:
        return sp_lin.solve(LegendreM, t)
    else:
        return (sp_lin.solve(LegendreM, t.T))


def lagrangeToJacobi(t, a, b, axis=0):
    n = t.shape[axis]
    points = chebNodes(n=n)
    JacobiM = np.zeros([n, n])
    for i in range(n):
        JacobiM[:, i] = special.eval_jacobi(i, a, b, points)
    if axis == 0:
        return sp_lin.solve(JacobiM, t)
    else:
        return (sp_lin.solve(JacobiM, t.T))


def integrate(self, f, w=None):
    if w is None:
        i = np.arange(0, f.shape[0])
        M = -((-1) ** i + 1) / (i ** 2 - 1)
        M[1] = 0
        return ((M.dot(f))).T
    else:
        F = self.cmul(f, w)
        n = F.shape[0]
        i = np.arange(0, n)
        M = -(np.cos(np.pi * i) + 1) / (i ** 2 - 1)
        Minfs = np.isinf(M)
        Mnans = np.isnan(M)
        M[Minfs] = 0
        M[Mnans] = 0
        return ((M.dot(F))).T


def legendreToLagrange(self, t, axis=0):
    n = t.shape[axis]
    points = self.chebNodes(n=n)
    LegendreM = np.zeros([n, n])
    for i in range(n):
        LegendreM[:, i] = special.eval_legendre(i, points)
    if axis == 0:
        return LegendreM.dot(t)
    else:
        return LegendreM.dot(t.T)


def legendreFT(fx, y, a=-1, b=1, axis=0):
    fx = np.array(fx)
    coeff = lagrangeToLegendre(fx, axis)
    plt.loglog(np.abs(coeff))
    plt.show()
    arr = np.arange(fx.shape[axis])
    q = (b - a) / 2
    p = (b + a) / 2
    xx, yy = np.meshgrid(arr, y)
    # Y = special.jn(xx + 0.5, yy * q)
    # t = time.time()
    # newy = chebNodes(int(q*(y[-1] - y[0]))*2, a=q*y[0], b=q*y[-1])
    # #
    # xx, nyy = np.meshgrid(arr, newy)
    # nY = special.jn(xx + 0.5, nyy)
    # cnY = chebEval(chebTransform(nY), y*q, a=q*y[0], b=q*y[-1])
    # np.set_printoptions(precision=4, suppress=False)
    # print(np.max(np.abs(Y - cnY.T), axis=0))
    # print('bessel eval', time.time() - t)

    FM = 1j ** (xx) * np.sqrt(2 * np.pi / yy / q) * special.jn(xx + 0.5, yy * q) * (q * np.exp(1j * yy * p))
    # print(FM.shape)
    # print(np_lin.cond(FM))
    return np.conj(FM.dot(coeff))


def legendreFTM(self, y, a=-1, b=1, axis=0):
    arr = np.arange(y.size)
    q = (b - a) / 2
    p = (b + a) / 2
    xx, yy = np.meshgrid(arr, y)
    FM = 1j ** (xx) * np.sqrt(2 * np.pi / yy / q) * special.jn(xx + 0.5, yy * q) * (q * np.exp(1j * yy * p))
    return FM


def CLegendreFT(self, t, a=-1, b=1, axis=0):
    t = np.array(t)
    coeff = self.lagrangeToLegendre(t, axis)
    arr = np.arange(t.shape[axis])
    q = (b - a) / 2
    p = (b + a) / 2

    def FM(arr, y):
        xx, yy = np.meshgrid(arr, y)
        return 1j ** (arr) * np.sqrt(2 * np.pi / yy / q) * special.jn(arr + 0.5, yy * q) * (q * np.exp(1j * yy * p))

    return lambda y: np.conj(FM(arr, y).dot(coeff))


def productIntegrate(self, f, a=-1, b=1):
    f_cheb = np.real(self.ChebTransform(f.T))
    i = np.arange(0, f.shape[0])
    j = np.arange(0, f.shape[0])
    ii, jj = np.meshgrid(i, j)
    M = -(((-1 + ii ** 2 + jj ** 2) * (1 + np.cos(ii * np.pi) * np.cos(jj * np.pi)) + 2 * ii * jj * np.sin(
        ii * np.pi) * np.sin(jj * np.pi)) / ((-1 + ii ** 2) ** 2 - 2 * (1 + ii ** 2) * jj ** 2 + jj ** 4))
    Minfs = np.isinf(M)
    Mnans = np.isnan(M)
    M[Minfs] = 0
    M[Mnans] = 0
    q = (b - a) / 2
    return q * ((f_cheb.T).dot(M.dot(f_cheb))).T


def meshL2Norm(self, x, y):
    x = self.ChebTransform(x)
    y = self.ChebTransform(y)
    if x.size >= y.size:
        subs = x - np.hstack((y, np.zeros(x.size - y.size)))
    if x.size < y.size:
        subs = y - np.hstack((x, np.zeros(y.size - x.size)))
    return self.product_integrate(subs)


def chebPlot(self, A, a=-1, b=1, n=1000):
    x = np.linspace(a, b, n)
    T = self.chebEval(A, x, a, b)
    plt.plot(x, T)
    # plt.show()


def plot(self, fx, a=-1, b=1, n=1000):
    x = np.linspace(a, b, n)
    T = sp_interp.barycentric_interpolate(self.chebNodes(n=fx.size, a=a, b=b), fx, x)
    plt.plot(x, T)


def Add(self, lhs, rhs):
    lhs = self.chebTransform(lhs);
    rhs = self.chebTransform(rhs)
    return np_cheb.chebadd(lhs, rhs)


def chebL2Norm(self, cfx):
    return np.sqrt(self.integrate(self.cmul(cfx, cfx)))


def chebEval(A, x, a=-1, b=1):
    q = (b - a) / 2;
    p = (a + b) / 2
    return np_cheb.chebval((x - p) / q, A, tensor=True)


def chebRoots(A, a=-1, b=1, ImTol=1e-6, im=False):

    D = np_cheb.chebcompanion(A)
    eigs = sp_lin.eigvals(D)
    if im == False:
        eigs = np.real(eigs[np.abs(np.imag(eigs)) < ImTol])
        eigs = eigs[eigs >= -1]
        eigs = np.sort(eigs[eigs <= 1])
    else:
        eigs = eigs[eigs.real >= -1]
        eigs = eigs[eigs.real <= 1]

    q = (b - a) / 2;
    p = (a + b) / 2
    return eigs * q + p


def indefInt(A, s=1):
    return np_cheb.chebint(A, scl=s)


def chebDiff(A, m=1):
    r = chebTransform(A)
    return np_cheb.chebder(r, m=m)


def cmul(X, Y, axis=0, copy=True):
    xs, ys = X.shape[axis], Y.shape[axis]

    x = X.copy(); y = Y.copy()

    N = xs + ys; n = N
    x[0] *= 2;
    y[0] *= 2

    cx = (dft.dct(x, type=1, axis=axis, n=n) / 2);
    cy = (dft.dct(y, type=1, axis=axis, n=n) / 2)

    if len(cx.shape) > 1 and len(cy.shape) > 1:
        cc = cx[:, None, :] * cy[:, :, None]
    elif len(cx.shape) > 1:
        cc = ((cx.T) * cy).T
    else:
        cc = cx * cy

    c = (dft.dct(cc, type=1, axis=axis, n=n) / (n - 1));
    c[0] /= 2
    return c


def xmul(self, x):
    return np_cheb.chebmulx(x)


def gegenbauerReconstruction(self, A, N):
    # np.set_printoptions(precision=3, suppress=True)
    entireA = np.hstack((A[1:][::-1], A))

    n = np.size(entireA)
    x = self.chebNodes(n=N)
    k = np.arange(-A.size + 1, A.size)

    xx, kk = np.meshgrid(x, k)
    M = np.exp(1j * kk * np.pi * xx).T

    fx = M.dot(entireA)

    cfx = self.chebTransform(fx)
    # gamma = 1

    # print(gamma)
    l = N
    # l = 4.5
    xx, NN = np.meshgrid(x, np.arange(N))
    gx = special.eval_gegenbauer(NN, l, xx)
    # print(gx)
    cgx = (self.chebTransform(gx.T)).T
    # print(cgx)
    wx = np.power(1 - x * x, l - 0.5)
    cwx = self.chebTransform(wx)
    cIx = []
    for i in range(gx.shape[0]):
        cIx.append(np_cheb.chebmul(cwx, cgx[i, :]))
        # print(cgx[i, :])
    cIx = np.asarray(cIx)
    I = []
    h = np.sqrt(np.pi) * special.eval_gegenbauer(np.arange(N), l, 1) * special.gamma(l + 0.5) / special.gamma(l) / (
            N + l)
    for it in cIx:
        I.append(np.real(self.integrate(it, cfx)))
    I /= h
    # print(np.array(I))
    # print(gx)
    plt.plot(x, (gx.T).dot(I))
    plt.show()


def chunk(self, f, a=-1, b=1, n=31, tol=1e-15):
    x = self.chebNodes(a, b, n)
    fx = f(x)
    cfxl = self.chebTransform(fx)
    x = np.linspace(a, b, 10000)
    fxl = self.chebEval(cfxl, x, a, b)
    # print(cfxl)
    # plt.loglog(np.abs(cfxl))
    # plt.show()
    x = self.chebNodes(a, b, n)
    fx = f(g(x))
    # nn = np.arange(fx.size)
    cfxh = self.chebTransform(fx)
    x = np.linspace(a, b, 10000)
    fxh = self.chebEval(cfxh, x, a, b)
    # plt.plot(x, fxh)
    # plt.plot(x, fxl)
    # plt.show()
    print(int)
    cf = np_cheb.chebsub(cfxh, cfxl)
    # cdf = self.chebDiff(cfxh, 3)
    x = np.linspace(a, b, 10000)
    df = self.chebEval(cf, x, a, b)
    # plt.loglog(np.abs(cf))
    # plt.show()
    plt.plot(x, df)
    plt.show()
    roots = self.chebRoots(cdf, a, b)

    sf = self.chebEval(cf, roots, a, b)
    dp = roots[np.argmax(np.abs(sf))]
    # print('root', dp, 'value', np.max(np.abs(fxh - fxl)))
    # plt.plot(x, fxh - fxl)
    # plt.show()
    return dp, np.max(np.abs(sf))


def adapt(self, f, n=23, d=4):
    x = self.chebNodes(n=n)
    fx = f(x)
    cfx = self.chebTransform(fx)
    c = np.zeros(d)
    # c[1] = 1; c[3] = -1;
    # c[-1] = 1/(2*n - 1)
    # A_eq = np.zeros([d, d])
    # first = (-1)**np.arange(d)
    # A_eq[-1, :] = np.ones(d)
    #
    # A_ub = -np.eye(d)
    # b_ub = np.zeros(d); b_ub[0] = 1
    # cons = ({'type': 'eq',
    #  'fun' : lambda x: np.dot(first, x)},
    #         {'type': 'eq',
    #  'fun' : lambda x: np.sum(x)})
    # np.set_printoptions(precision=3, suppress=True)
    J = 0

    def b(c):
        co = -np.sum(c[::2]);
        ce = -np.sum(c[1::2])
        c = np.hstack((c, co, ce))
        g = lambda x: x + self.chebEval(c, x)
        x, w = np_cheb.chebgauss(n * 6)

        dg1 = self.chebDiff(c)
        dg2 = self.chebDiff(dg1)
        js = (-1) ** np.arange(dg1.size)
        if np.sum(np.abs(dg2) > 1e-8):
            dgroots = self.chebRoots(dg2)
            # print(dgroots)
            if dgroots.size > 0:
                if np.min(self.chebEval(dg1, dgroots)) <= -1 + 1e-9 or np.dot(js, dg1) <= -1 + 1e-9 or np.sum(
                        dg1) <= -1 + 1e-9:
                    return 1 + np.random.rand(1)
                else:
                    if np.dot(js, dg1) <= -1 + 1e-9 or np.sum(dg1) <= -1 + 1e-9:
                        return 1
        # print(dg)
        # if np.sum(dg) < -1:
        #     return 1
        dg = self.chebDiff(c)
        if np.random.rand(1) > 0.9:
            # plt.plot(x, self.chebEval(dg, x))

            # plt.plot(x, g(x), '--')
            # plt.show()
            # plt.plot(x, f(g(x)))
            # plt.show()
            nodes = self.chebNodes(n=n)
            plt.loglog(np.abs(self.chebTransform(f(g(nodes)))), 'ro')
            plt.show()

        # F = lambda x, n: f(g(x))*special.eval_chebyt(n, x)
        # I = np.abs(np.dot(w, F(x, int(4*n/5))))
        nodes = self.chebNodes(n=n)
        I = np.max(np.abs(self.chebTransform(f(g(x)))[-2:]))
        # print(I)
        print(I, c)
        return I

    # bounds = .05*np.ones([8, 2])
    # bounds[:, 0] *= -1
    # print(bounds)
    # print(sp_opt.differential_evolution(b, bounds, maxiter=int(1e10)))
    #     options={'maxfev': 1e10, 'maxiter': 1e10}
    sol = sp_opt.minimize(fun=b, x0=np.zeros(4), method='BFGS')

    # A = np.zeros([d, 2])
    # A[1, 0] = 1
    # A[:, 1] = 90
    #
    # sol = sp_opt.minimize(fun=b, x0=c, constraints=cons, bounds=A, method='SLSQP')
    # print(sol)


def chebDiffM(n, a, b):
    x = chebNodes(n, a, b)
    X = np.ones([n, n], dtype=np.float)
    X = ((X.T) * x).T
    dX = X - X.T
    c = np.append([2], np.ones(n - 2))
    c = np.append(c, [2])
    c *= (-1) ** np.arange(0, n)
    c = np.reshape(np.kron(c, 1 / c), newshape=[n, n])
    d = c / (dX + np.eye(n))
    d = d - np.diag(np.sum(d, axis=1))
    return d


def deAdaptivity(f, n=30):
    x = chebNodes(n=n)
    fx = f(x)
    cfx = chebTransform(fx)

    D = chebDiffM(n=n)
    D2 = D.dot(D)
    D2[0, :] *= 0;
    D2[-1, :] *= 0
    D2[0, 0] = 1;
    D2[-1, -1] = 1
    fx[0] = -1;
    fx[-1] = 1
    sol = sp_lin.solve(D2, fx)
    plt.plot(x, sol)
    plt.show()


def draw(f, a=-1, b=1, n=31, tol=1e-14):
    dp, err = chunk(f, a, b, n)
    part = np.array([[a, dp], [dp, b]])
    ldp, lerr = chunk(f, a, dp, n)
    rdp, rerr = chunk(f, dp, b, n)

    errs = np.array([lerr, rerr])
    arg_errs = (np.argwhere(errs > tol)).flatten()

    for it in arg_errs:
        dp, err = chunk(f, a, b, n)

        ldp, lerr = chunk(f, it[0], dp, n)
        rdp, rerr = chunk(f, dp, b, n)


def BesselSeries(f, n, L):
    cf = chebTransform(f)
    k = special.jn_zeros(n)

def trig_interp(fx, x):
    n = fx.size
    def ift(x):
        k0 = np.arange(int((n + 1)/2))
        k1 = np.arange(-int((n + 1) / 2), 0)
        k = np.hstack([k0, k1])
        nn, xx = np.meshgrid(k, x)
        return np.exp(1j*(nn)*(xx))/n

    ffx = dft.fft(fx)
    M = ift(x)
    ifx = (M).dot(ffx)

    return np.real(ifx)


# f = lambda x: np.sqrt(1-x**2)
# # # f = lambda x: x
# g = lambda x: np.cos(x)
# # np.set_printoptions(suppress=True)
# for i in range(31, 100, 2):
#     N = i
#     x = chebNodes(N + 1, -1, 1)
#     # print(x)
#     # print(x)
#     fx = f(x)
#     # plt.plot(x, fx, '--')
#     # plt.show()
#     cfx = chebTransform(fx)
#     x = chebNodes(1000, -1, 1)
#     cifx = chebEval(cfx, x, -1, 1)
#     # plt.plot(x, cifx - f(x))
#     # plt.plot(x, f(x), '--')
#     # plt.show()
#     # print('cheb', )
#     x1 = np.max(np.abs(cifx - f(x)))
#     # plt.plot(x, cifx - f(x), '--')
#
#     # plt.plot(cifx)
#     # plt.plot(x, cifx - f(x))
#     # plt.show()
#
#     # x = np.linspace(0, 2*np.pi, 2*N + 1)
#     x = np.arange(0, 2*N)*2*np.pi/(2*N)
#     # print(g(x))
#     # time.sleep(500)
#     # print(g(x)[:N + 1][::-1])
#     fx = f(g(x))
#     # plt.plot(x, fx)
#     # plt.show()
#     # fx[N + 1:] /= -1
#     # fx[N + 1:] += fx[N]*2
#
#     # fx = np.hstack([fx, -fx[1:] + fx[-1]*2])
#     # plt.plot(fx)
#     # plt.show()
#     # plt.show()
#     # plt.plot(x, fx, 'o')
#     # plt.show()
#     x = np.arccos(chebNodes(1000))
#     # x = np.hstack([x,x])
#     # time.sleep(500)
#     ifx = trig_interp(fx, x)
#     # plt.plot(np.imag(ifx))
#     # plt.show()
#     # plt.plot(ifx[1000:] - cifx, '--')
#     # plt.plot(x[:500], ifx[:500])
#     # plt.plot(x[:500], f(np.cos(x))[:500], '-.')
#     # plt.plot(x, np.real(f(g(x))))
#     # plt.plot(x, np.imag(f(g(x))), '--')
#     # plt.show()
#     # plt.plot(x, ifx, '--')
#     # plt.plot(x, f(g(x)))
#     # plt.show()
# #
#     x = chebNodes(1000)
#     # plt.plot(x, ifx - f(x))
#     # plt.show()
#     x2 = np.max(np.abs(ifx - f(x)))
#     # plt.plot(cifx - ifx[:1000][::-1])
#     # plt.show()
#     # plt.plot(, '--')
#     # plt.show()
#     # plt.show()
# #     # plt.plot(x, f(np.cos(x)), '-.')
# #     # x = np.linspace(-1, 1, 1000)
# #     # cifx = chebEval(cfx, x, -1, 1)
# #     # plt.plot(np.arccos(x), cifx, '--')
# #     # plt.show()
#     print(i, x1, x2)
# #
# # g = lambda x: np.exp(np.cos(x) + np.sin(2*x))
# # for i in range(5, 71, 4):
# #     x = np.arange(0, i)*2*np.pi/i
# #     gx = g(x)
# #     x = np.linspace(0, 2*np.pi, 500)
# #     igx = trig_interp(gx, x)
# #     print(i, np.max(np.abs(igx - g(x))))
# #     plt.plot(x, igx - g(x))
# #     plt.show()
#     plt.show()
#



# np.set_printoptions(precision=3, suppress=True)
# obj = spectral()
# v = np.random.rand(3);  u = np.random.rand(3)
# v = obj.chebTransform(v); v[-1] = 0.25; v[0] = v[-1]; v[1] = 0.5; u = obj.chebTransform(u); u = v.copy()
# x = np.linspace(-1, 1, 1000)
#
# # plt.show()
# w = obj.cmul(v, u)
# plt.plot(x, obj.chebEval(v,  x)*obj.chebEval(u, x))
# x = np.linspace(-1, 1, 50)
# #
# plt.plot(x, (obj.chebEval(w, x)).T, 'ro')
# plt.show()

# s = lambda x: np.sqrt(1 - (x)**2)
# ns = lambda x: np.abs(x - 0.3) + 3*np.abs(x + 0.8)
# rs = lambda x: np.exp(-9*x)#*np.sin(19*x) + x
# a = 0; b = 1;
# g = lambda x, c: special.erf(c*x)/special.erf(c) - special.erf(0)
# # dp = obj.chunk(lambda x: s(x), a=a, b=b)
# dp = obj.deAdaptivity(s)
# print(dp)
# a = dp
# k = obj.chebNodes(a=-1, b=1, n=9)
# k = np.arange(8)
# v = lambda k: (np.sinc(1 - k) + np.sinc(1 + k))/2
# # v.
# # print(v(0))
# # print(v(k))
# # plt.plot(k, v(k))
# # plt.show()
# # k = np.linspace(-10, 10, 500)
# # plt.plot(k, v(k))
# # plt.show()
# # k = np.arange(3)
# gamma = 2*np.pi*np.exp(1)/10
# gamma = 2/5
# print(int(np.size(k)*gamma))
# obj.gegenbauerReconstruction(v(k), int(np.size(k)*gamma))
# v = obj.chebTransform(v)
# x = np.linspace(0, 10, 9000)
# np.set_printoptions(precision=4, suppress=True)
# for i in range(30):
#     t = time.time()
#     # obj.chebRoots(v, a=0, b=10)
#     obj.chebEval(v, x, a=0, b=10)
#     # plt.plot(x, obj.chebEval(v, x, a=0, b=10))
#     # plt.show()
#     print(time.time() - t)
