from new_misc import diff
from new_misc import integr
from new_misc import spectral
import scipy.special as special
import numpy.linalg as np_linalg
import matplotlib.pyplot as plt
import scipy.linalg as sp_linalg
import scipy.interpolate as sp_interp
import time as time
import warnings
import numpy as np
warnings.filterwarnings('ignore')

def simplest(n):
    c = 0; m = 0.5; d = 1
    sigma = 1
    xl = spectral.chebNodes(a=c, b=m, n=n); xr = spectral.chebNodes(a=m, b=d, n=n)
    Dl = diff.cheb_diff(n=n, a=c, b=m); Dr = diff.cheb_diff(n=n, a=m, b=d)
    cDl = spectral.chebTransform(Dl); cDr = spectral.chebTransform(Dr)


    iDl = integr.pClenshaw(f=cDl.copy(), g=cDl.copy(), a=c, b=m, precalc=True, chebForm=True)
    iDr = integr.pClenshaw(f=cDr.copy(), g=cDr.copy(), a=m, b=d, precalc=True, chebForm=True)
    # print(iDl)
    # iDl[:, -1] -= Dl[-1, :]/2
    # iDl[-1, :] -= (Dl)[-1, :]/2
    # iDl[-1, -1] += sigma
    #
    # iDr[:, 0] += (Dr)[0, :]/2
    # iDr[0, :] += Dr[0, :]/2
    # iDr[0, 0] += sigma
    # print(iDr)
    # time.sleep(50)
    Id = np.eye(n)
    cId = spectral.chebTransform(Id)
    iMl = integr.pClenshaw(f=cId, g=cId, a=c, b=m, precalc=True, chebForm=True)
    iMr = integr.pClenshaw(f=cId, g=cId, a=m, b=d, precalc=True, chebForm=True)
    # print(iM)
    #
    # zeros = np.eye(2)*0
    # Idl = zeros.copy(); Idl[0, 0] = 1
    # Idr = zeros.copy(); Idr[1, 1] = 1

    # D = np.kron(Idl, iDl) + np.kron(Idr, iDr)
    D = np.zeros([2*n - 1, 2*n - 1], dtype=np.float)
    D[:n, :n] += iDl
    D[n - 1:, n - 1:] += iDr
    # print(D)

    # D[:n, n]     -= ((Dl)[-1, :]/2)
    # D[n - 1, n:] += ((Dr)[0, :]/2)
    # D[n - 1, n]  += sigma

    # D[n, :n]     += ((Dr)[0, :]/2)[::-1]
    # D[n:, n - 1] -= ((Dl)[-1, :]/2)[::-1]
    # D[n, n - 1]  += sigma
    # print(D)


    # print(np_linalg.det(D))
    # time.sleep(50)
    M = 0*D.copy()
    M[:n, :n] += iMl
    M[n - 1:, n - 1:] += iMr
    D = D[:-1, :-1]
    M = M[:-1, :-1]

    f = lambda t: np.exp(-t)
    # f = lambda t: np.heaviside(t - 0.5, 0.5)
    # fl = f(xl); fr = f(xr)[:-1]

    # F = np.hstack([fl, fr])

    # polyl = sp_interp.BarycentricInterpolator(xi=xl, yi=Dl)
    # Fl = integr.reg_32(lambda x: (polyl(x).T)*f(x), n=300, a=c, b=m)
    # Fl = polyl(0.5)/2
    # polyr = sp_interp.BarycentricInterpolator(xi=xr, yi=Dr)
    # Fr = integr.reg_32(lambda x: (polyr(x).T)*f(x), n=300, a=m, b=d)
    # Fr = polyr(0.5)/2

    polyl = sp_interp.BarycentricInterpolator(xi=xl, yi=Id)
    Fl = integr.reg_32(lambda x: (polyl(x).T)*f(x), n=300, a=c, b=m)
    # Fl = (polyl(0.5))/2#*np.exp(-.5))/2
    # polyl = sp_interp.BarycentricInterpolator(xi=xl, yi=Id)
    # Fl += -(polyl(0.5)*np.exp(-.5))/2
    polyr = sp_interp.BarycentricInterpolator(xi=xr, yi=Id)
    Fr = integr.reg_32(lambda x: -(polyr(x).T)*f(x), n=300, a=m, b=d)
    # Fr = (polyr(0.5))/2#*np.exp(-0.5))/2
    # polyr = sp_interp.BarycentricInterpolator(xi=xr, yi=Id)
    # Fr += -(polyr(0.5)*np.exp(-0.5))/2
    F = np.hstack([Fl, Fr])[:-1]

    # F = np.hstack([Fl, Fr])[:-1]
    F = np.zeros(2*n - 2)
    F[:n] = Fl
    F[n - 1:] += Fr[:-1]
    # print(F)
    # print(F)
    # print(Fl)
    # print(Fr)
    # plt.plot(F)
    # plt.show()
    xx = np.hstack([xl[:-1], xr[:-1]])
    # print(D)
    sol = sp_linalg.solve(D, F)
    sol[n:] = sol[n:]
    sol = sol
    # plt.plot(xx, ))
    asol = -1 - 1/np.exp(1) + np.exp(-xx) + xx
    asol = -1/8*(-1 + ((1 - 2*xx)**2)*f(xx))
    asol = -1/2 + (-1/2 + xx)*f(xx)
    asol = -1 + f(xx)
    asol = lambda x: (-3 + (1 + 2*x)*f(x))/2/np.sqrt(np.exp(1))
    # print(np_linalg.cond(D))
    # print(np.max(np.abs(asol(xx) - sol)))
    plt.plot(sol)
    # plt.plot(xx, asol(xx), 'ro')
    plt.show()

for n in range(3, 10):
    # n = 3
    # simplest(n)
    from new_misc import elements as elems
    u1 = elems.elem([0, 0.5], n, ['n', 'i'])
    u2 = elems.elem([0.5, 1], n, ['i', 'd'])

    inters = elems.interactions()
    K1 = inters.integr_K(u1); K2 = inters.integr_K(u2)
    # I1 = inters.integr_I(u1); I2 = inters.integr_I(u2)
    # K11 = -inters.test_bI1(u1); K22 = inters.test_bI2(u2)[:-1, :-1]
    # K1 += K11
    # K11 = inters.test_bI2(u1);

    K11 = inters.integr_bK(u1); K22 = inters.integr_bK(u2)
    K12 = inters.bK(u1, u2)
    # K2 -= K22
    # K12 = inters.bK(u1, u2)
    # print(K12)

    K1 += K11; K2 += K22
    f = lambda x: np.exp(-x)
    f1 = inters.fI(u1, f, j=lambda x: x*0 + 1)
    f2 = inters.fI(u2, f, j=lambda x: x*0 + 1)
    K = sp_linalg.block_diag(K1, K2)
    K[n:, :n] = K12
    K[:n, n:] = K12.T
    # print(K)
    # I = sp_linalg.block_diag(I1, I2)
    F = np.hstack((f1, f2))[:-1]
    # K
    sol = sp_linalg.solve(K, F)

    # plt.plot(sol)
    # plt.show()
    FEM = elems.old_fem()
    x = np.linspace(0, 1, 500)
    fx = -1 - 1/np.exp(1) + np.exp(-x) + x
    # plt.plot(x, -fx)
    FEM.plot1D([u1, u2], sol)

    # elems.fem.plot1D()
    # np.set_printoptions(suppress=True, precision=3)
    # for i in range(3, 13):
    #     simplest(i)