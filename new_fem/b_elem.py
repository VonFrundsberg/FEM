import numpy as np
from new_misc import spectral as sp
from new_misc import diff as diff
from new_misc import integr as mintegr
import scipy.linalg as sp_linalg
import matplotlib.pyplot as plt
import time as time

class b_f():
    def __init__(self, f):
        self.f = f
    def p_eval(self, x):
        res = []
        for it in self.f:
            res.append(it(x))
        return res
    def ps_eval(self, x):
        return np.atleast_1d(self.f(x))

class b_elem():
    def __init__(self, I=[-1.0, 1.0], n=1, bc=[], Id=None):
        self.I = np.array(I)
        self.n = n
        self.bc = bc
        if Id == None:
            self.Id = np.eye(n)
        else:
            self.Id = Id
        self.D = diff.cheb_diff(self.n).dot(self.rp_eval())

        if I[0] == -np.inf:
            self.map = lambda x: -((1 - x) / (1 + x) - I[1])
            self.imap = lambda x: (x + 1 - I[1])/(-x + I[1] + 1)
            self.dmap = lambda x: (1 + x) ** 2 / 2
            return

        if I[1] == np.inf:
            self.map = lambda x: ((1 + x) / (1 - x) + I[0])
            self.imap = lambda x: (-x + I[0] + 1) / (-x + I[0] - 1)
            self.dmap = lambda x: (x - 1) ** 2 / 2
            return

        a = I[0]; b = I[1]
        q = (b - a) / 2; p = (b + a) / 2
        self.map = lambda x: (q * x + p)
        self.imap = lambda x: (x - p)/q

        self.dmap = lambda x: 1 / (q + x * 0)
    def rp_eval(self):
        Id = self.Id
        for it in self.bc:
            Id[it[0], it[0]] = it[1]
        return Id
    def rp_plot(self):
        P = self.Id
        x = np.linspace(-1, 1, 100)
        plt.plot(x, sp.bary(P, x))
        plt.show()
    def p_eval(self, x): ## return shape: (*x.shape, n)
        x = np.atleast_1d(x)
        P = self.rp_eval()
        xs = self.imap(x)
        res = sp.bary(P, xs)
        return res
    def xs(self):
        x = sp.chebNodes(self.n, -1, 1)
        x = self.map(x)
        return x
    def dp_eval(self, x):
        x = np.atleast_1d(x)
        P = self.D
        xs = self.imap(x)
        res = sp.bary(P, xs)*np.reshape(self.dmap(xs), (*x.shape, 1))
        return res
    def gen_func(self):
        return lambda x: self.p_eval(x)
    def gen_funcd(self):
        return lambda x: self.dp_eval(x)

    def __call__(self, x):
            return self.p_eval(x)
    def d(self, x):
            return self.dp_eval(x)

# a = b_elem([-np.inf, 0], 10)
# b = b_elem([0, np.inf], 10)
# x = np.linspace(0, 100, 30)
# print(a(-x))
# print(b(x))
# class func():
#     def __init__(self):
#         return
#
#     def integr(self, I, F, axis=0):
#         w, n = mintegr.reg_32_wn(I[0], I[1], 100, merged=True)
#         shapes = []
#         fx = F(n)
#         for it in fx:
#             if it.shape[1:] != ():
#                 shapes.append(*it.shape[1:])
#             else:
#                 shapes.append(1)
#         res = fx[0]
#         for i in range(len(fx) - 1):
#             if len(fx[i + 1].shape[1:]) != 0:
#                 res = res[:, None, :]*fx[i + 1][:, :, None]
#                 res = np.reshape(res, [n.size, np.prod(res.shape[1:])])
#             else:
#                 res = res[:, :]*fx[i + 1][:, None]
#
#         res = np.reshape(res, [n.size, *shapes[::-1]])
#         res = np.tensordot(res, w, axes=[0, 0])
#         return res
#
#     def eval(self, x, F, axis=0):
#         shapes = []
#         fx = F(x)
#         # time.sleep(500)
#         for it in fx:
#             if it.shape[1:] != ():
#                 shapes.append(*it.shape[1:])
#             else:
#                 shapes.append(1)
#         res = fx[0]
#         for i in range(len(fx) - 1):
#             if len(fx[i + 1].shape[1:]) != 0:
#                 res = res[:, None, :]*fx[i + 1][:, :, None]
#
#                 res = np.reshape(res, [1, np.prod(res.shape[1:])])
#             else:
#                 res = res[:, :]*fx[i + 1]
#         res = np.reshape(res, [*shapes][::-1])
#         return res





