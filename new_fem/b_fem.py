import numpy as np
from new_fem.b_elem import *
class FEM():

    def gen_space(self, mesh):
        return 0
    def g(self, u):
        return lambda x: u.d(x)
    def i(self, u):
        fun = func()
        F = 1
        return fun.integr()

    # def A(self, a, f):
    #     a = lambda u: i(g(u), g(u), j)
    #     for it in l:
    #         K = a(u_i, u_i)
    #     for it in neigh:
    #         for every pair in it:
    #             Kij = a(u_i, u_j)
np.set_printoptions(precision=3, suppress=True)
# N= 10
# I1, I2 = [0, .5], [.5, 1.0]# [.66, 1]
# elem1 = b_elem(I1, N)
# elem2 = b_elem(I2, N, bc=[[-1, 0]])
# # elem3 = b_elem(I3, 3)
#
# fun = func()
# f = lambda x: np.exp(-x)
#
# M1 = lambda x: [elem1.d(x), elem1.d(x)]; M12 = lambda x: [elem1.d(x)/2, elem2(x)]
# M2 = lambda x: [elem2.d(x), elem2.d(x)]; M21 = lambda x: [elem1(x), elem2.d(x)/2]
# # M3 = lambda x: [elem3.d(x), elem3.d(x)]
# M11 = lambda x: [elem1.d(x)/2, elem1(x)]
# M22 = lambda x: [elem2.d(x)/2, elem2(x)]
# F1 = lambda x: [elem1(x), f(x)]
# F2 = lambda x: [elem2(x), f(x)]
#
# # F3 = lambda x: [elem3(x), f(x)]
# # f = b_f(lambda x: np.exp(x))
# # F = lambda x: [elem1(x, 1), elem2(x, 1)]
# iM1 = fun.integr(I1, M1);   iF1 = fun.integr(I1, F1)
# iM2 = fun.integr(I2, M2);   iF2 = fun.integr(I2, F2)
#
# eM12 = fun.eval(0.5, M12); eM21 = fun.eval(0.5, M21)
# eM11 = fun.eval(0.5, M11); eM22 = fun.eval(0.5, M22)
# # iM3 = fun.integr(I3, M3);   iF3 = fun.integr(I1, F1)
#
# MU = np.hstack([iM1 - eM11 - eM11.T, eM12.T - eM21.T])
# ML = np.hstack([-eM21 + eM12, iM2 + eM22 + eM22.T])
# M = np.vstack([MU, ML])
# F = np.hstack([iF1, iF2])
# print(F)
# # print(M)
# # print(F)
# sol = sp_linalg.solve(M[:-1, :-1], np.squeeze(F)[:-1])
# from new_misc import spectral as sp
# xl = sp.chebNodes(N, 0, 0.5)
# xr = sp.chebNodes(N, 0.5, 1)
# x = np.hstack([xl, xr])[:-1]
# Fq = lambda x: -(np.exp((-1 - x))*(np.exp(1) - np.exp(x) - np.exp(1 + x) + np.exp(1 + x)*x))
#
# plt.plot(x, sol - Fq(x))
#
# plt.show()

