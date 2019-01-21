#main dfem file
import numpy as np
import scipy.special as special
import numpy.linalg as np_linalg
import scipy.interpolate as sp_interp
from misc import diff as diff
import matplotlib.pyplot as plt
from misc import integr as integr
from misc import spectral as spectral
import scipy.linalg as sp_linalg
import scipy.sparse as sparse
import warnings
warnings.filterwarnings('ignore')
import time as time

class baseElem():
    def __init__(self, interval, n, left='i', right='i'):
        self.n = n
        self.a = interval[0]; self.b = interval[1]
        self.linfty = False; self.rinfty = False
        self.left = left; self.right = right
        flag = False
        if interval[0] == 'infty':
            self.linfty = True; flag = True
            self.map = lambda x: ((1 - x)/(1 + x) + interval[1])
            self.dmap = lambda x: (1 + x)**2/2

        if interval[1] == 'infty':
            self.rinfty = True; flag = True
            self.map = lambda x: ((1 + x)/(1 - x) + interval[0])
            self.dmap = lambda x: (x - 1) ** 2/2

        if flag == False:
            a = interval[0]; b = interval[1]
            q = (b - a)/2; p = (b + a)/2
            self.map = lambda x: (q*x + p)
            self.dmap = lambda x: 1/(q + x*0)
        self.D = diff.cheb_diff(n)
    def __copy__(self):
        return baseElem([self.a, self.b], self.n, self.left, self.right)

    def f(self):
        return np.eye(self.n)
    def j(self):
        return self.map
class elem():
    def set_n(self, n):
        self.elems[0].n = n
        self.elems[0].D = diff.cheb_diff(n)
    def __copy__(self):
       elems = []
       for it in self.elems:
           self.elems.append(it.copy())
       return elems
    def __init__(self, interval, n, boundary = None):
        self.elems = []
        n = np.atleast_1d(n)
        self.dim = n.size
        if boundary is None:
                if self.dim == 1:
                        self.elems.append(baseElem(interval, np.int(n), 'i', 'i'))
                else:
                    for i in range(self.dim):
                        self.elems.append(baseElem(interval[i], np.int(n[i]), 'i', 'i'))
        else:
            if self.dim == 1:
                for i in range(self.dim):
                        self.elems.append(baseElem(interval, np.int(n), boundary[0], boundary[1]))
            else:
                for i in range(self.dim):
                    if i in boundary[0]:
                        self.elems.append(baseElem(interval[i], np.int(n[i]), boundary[1][i][0], boundary[1][i][1]))
    def __getitem__(self, item):
        return self.elems[item]
    def __len__(self):
        return len(self.elems)
    def getN(self):
        arr = []
        for it in self.elems:
            arr.append(it.n)
        return np.array(arr, dtype=np.int)
class interactions():
    def DI(self, elem):
        j = elem.j; jd = elem.jd; jn = elem.jn; n = elem.n


        x = spectral.chebNodes(jn)
        nans = np.isnan(jdd)
        x[nans] /= 1 + 1e-8
        jdd = jd(x)
        iD = self.integr_base(elem.D, elem.D, jd(x))
        iI = self.integr_base(np.eye(elem.n), np.eye(elem.n), j(x))

        if elem.left == True and elem.right == True:
            iD[:, -1] -= jd(-1) * elem.D[-1, :] / 2
            iD[-1, :] -= jd(-1) * elem.D[-1, :] / 2

            iD[:, 0] += jd(1) * elem.D[0, :] / 2
            iD[0, :] += jd(1) * elem.D[0, :] / 2

        if elem.left == False and elem.right == True:
            iD[:, -1] -= jd(1) * elem.D[-1, :] / 2
            iD[-1, :] -= jd(1) * elem.D[-1, :] / 2

        if elem.right == False and elem.left == True:
            iD[:, 0] += jd(-1) * elem.D[0, :] / 2
            iD[0, :] += jd(-1) * elem.D[0, :] / 2

        elem.iI = iI

        return iD, iI
    def bDI(self, lhs, rhs, j):

        jdL = lambda x: j(lhs.map(x)) * lhs.dmap(x)
        jdR = lambda x: j(rhs.map(x)) * rhs.dmap(x)

        M = np.zeros([rhs.n, lhs.n])

        M[:, -1] -= jdR(-1 + 1e-8) * (rhs.D[0, :] / 2)
        M[0, :] += jdL(1 - 1e-8) * (lhs.D[-1, :] / 2)
        return M
    def fI(self, elem, f, j, method='gauss', k=10, precalc=False):
        if precalc == False:
            if method == 'gauss':
                x1 = spectral.chebNodes(k)
                elem = elem[0]
                nx = elem.map
                Id = np.eye(elem.n)
                jj = lambda x: j(nx(x))*f(nx(x))/elem.dmap(x)
                jx = jj(x1)
                res = integr.pClenshaw(Id, jx, chebForm=False, precalc=True)
                return res
        else:
            if method == 'gauss':
                elem = elem[0]
                nx = elem.map
                Id = np.eye(elem.n)
                jj = lambda x: j(nx(x))*f/elem.dmap(x)
                x1 = spectral.chebNodes(k)
                jx = jj(x1)
                res = integr.pClenshaw(Id, jx, chebForm=False, precalc=True)
                return res
    def svdTT_FI(self, u, f, j, method='gauss', tt_tol=1e-5):
        grids = []
        for i in range(len(u)):
            nx = u[i].map
            grids.append(nx(spectral.chebNodes(u[i].n)))

        mshgrid = np.meshgrid(*grids, indexing='ij')
        fx = f(*mshgrid)
        nans = np.isnan(fx)
        fx[nans] = 0

        from new_misc import approx
        cores = approx.decay_smth(f=fx, tol=tt_tol)
        Is = []


        for i in range(len(u)):
            cores[i] = np.swapaxes(cores[i], 0, 1)
            shape1, shape2 = cores[i].shape[1], cores[i].shape[2]
            cores[i] = np.reshape(cores[i], [u[i].n, shape1*shape2])

            if u[i].a != 'infty' and u[i].b != 'infty':
                ccore = spectral.chebTransform(cores[i]); cId = spectral.chebTransform(np.eye(u[i].n))
                x = spectral.chebNodes(u[i].n)
                cx = spectral.chebTransform(j[i](u[i].map(x)) / u[i].dmap(x))
                cxId = spectral.cmul(cId, cx)
                I = integr.pClenshaw(cxId, ccore, precalc=True, chebForm=True)
            else:
                I = self.integr_trapz(cores[i], np.eye(u[i].n), lambda x: j[i](u[i].map(x)) / u[i].dmap(x))
            # if int(i%1) == 0:
            #     I *= -1
            I = np.reshape(I, [u[i].n, shape1, shape2])

            if u[i].left == 'd':
                I = I[1:, :, :]
            if u[i].right == 'd':
                I = I[:-1, :, :]
            Is.append(I)

        tmp = Is[0]
        Iss = []
        # for i in range(3):
        #     # print('meh', np.squeeze(Is[i]))
        #     f = lambda x: np.exp(-x**2)
        #     a = u[i].a; b = u[i].b
        #     x = spectral.chebNodes(n=u[i].n, a=a, b=b)
        #     fx = f(x)
        #
        #     Iss.append(integr.pClenshaw(fx, np.eye(u[i].n), a=a, b=b, precalc=True)[1:-1])
        #     # print('not meh', I[1:-1] - 0*np.squeeze(Is[i]))
        # for i in range(3):
        #     return np.outer(np.outer(Iss[0], Iss[1]), Iss[2]).flatten()
        # import time as time
        # time.sleep(500)
        for i in range(1, len(u)):
            # print('before', tmp.shape, Is[i].shape)
            tmp = np.einsum('kij, ljm->klm', tmp, Is[i])
            # tmp = np.squeeze(np.dot(tmp, Is[i]))
            # print('later', tmp.shape)
            tmp = np.reshape(tmp, [tmp.shape[0]*tmp.shape[1], 1, tmp.shape[2]])
            # tmp = np.reshape(tmp, [tmp.shape[0] * tmp.shape[1], 1, tmp.shape[2]])



        # print(tmp.shape)
        return tmp.flatten()
    def K(self, elem):
        Ks = []
        for it in elem:
            Ks.append(self.DI(it))
    def iBK(self, lhs, rhs, j):

        al = lhs.a; ar = rhs.a
        bl = lhs.b; br = rhs.b
        nl = lhs.n; nr = rhs.n
        xl = spectral.chebNodes(nl, al, bl); xr = spectral.chebNodes(nr, ar, br)
        Il = np.eye(nl); Ir = np.eye(nr)
        lPoly = sp_interp.BarycentricInterpolator(xi=xl, yi=Il)
        rPoly = sp_interp.BarycentricInterpolator(xi=xr, yi=Ir)
        a = np.max([al, ar]); b = np.min([bl, br])
        w, x = integr.reg_32_wn(a=a, b=b, n=70, merged=True)
        lhs1 = ((lPoly(x).T)*lhs.J(x)).T; rhs1 = rPoly(x)
        A = lhs1[:, None, :] * rhs1[:, :, None]
        A = np.tensordot(A, w, (0, 0))
        return A
    def bK(self, lhs, rhs, j=None):
        flag = False
        if j is None:
            j = []
            for i in range(lhs.dim):
                j.append(lambda x: x*0 + 1)
        for i in range(lhs.dim):
            if rhs[i].a == lhs[i].b or rhs[i].b == lhs[i].a:
                flag = True
                ii = i

        if flag == True:
            if ii == 0:
                tmp = self.bDI(lhs[0], rhs[0], j[0])
            else:
                tmp = self.integr_base(np.eye(lhs[0].n), np.eye(rhs[0].n), j[0])

            if lhs[0].left == 'd':
                tmp = tmp[:, 1:]
            if lhs[0].right == 'd':
                tmp = tmp[:-1, :]
            if rhs[0].left == 'd':
                tmp = tmp[:, :-1]
            if rhs[0].right == 'd':
                tmp = tmp[:-1, :]

            for i in range(1, lhs.dim):
                if i != ii:

                    hey = self.integr_trapz(np.eye(lhs[i].n),
                                      np.eye(rhs[i].n),
                                      lambda x: j[i](lhs[i].map(x)) / lhs[i].dmap(x))
                    if lhs[i].left == 'd':
                        hey = hey[:, 1:]
                    if lhs[i].right == 'd':
                        hey = hey[:-1, :]
                    if rhs[i].left == 'd':
                        hey = hey[1:, :]
                    if rhs[i].right == 'd':
                        hey = hey[:, :-1]
                    # print(hey.shape)
                    tmp = sparse.kron(tmp, hey)
                else:
                    hey = self.bDI(lhs[i], rhs[i], j[i])

                    if lhs[i].left == 'd':
                        hey = hey[:, 1:]
                    if lhs[i].right == 'd':
                        hey = hey[:-1, :]
                    if rhs[i].left == 'd':
                        hey = hey[:, :-1]
                    if rhs[i].right == 'd':
                        hey = hey[:-1, :]
                    tmp = sparse.kron(tmp, hey)

            return tmp
    def integr_base(self, f, g, w):
        cf = spectral.chebTransform(f); cg = spectral.chebTransform(g); cw = spectral.chebTransform(w)
        cwf = spectral.cmul(cf, cw)
        I = integr.pClenshaw(cwf.copy(), cg.copy(), precalc=True, chebForm=True)
        return I
    def integr_trapz(self, f, g, j):
        xl = spectral.chebNodes(n=f.shape[0]); xr = spectral.chebNodes(n=g.shape[0])
        lPoly = sp_interp.BarycentricInterpolator(xi=xl, yi=f)
        rPoly = sp_interp.BarycentricInterpolator(xi=xr, yi=g)
        w, x = integr.reg_32_wn(n=70, merged=True)
        lhs1 = ((lPoly(x).T) * j(x)).T
        rhs1 = rPoly(x)
        A = lhs1[:, None, :] * rhs1[:, :, None]
        A = np.tensordot(A, w, (0, 0))
        return A
    def integr_K(self, u, j=None, k=10):
        Ds = []; Is = []
        if j is None:
            j = []
            for i in range(u.dim):
                j.append(lambda x: x*0 + 1)
        for i in range(len(u)):                               #1d_matrix_prep
            x = spectral.chebNodes(k);  x /= 1 + 1e-8
            w = j[i](u[i].map(x))*u[i].dmap(x)

            to_add = self.integr_base(u[i].D, u[i].D, w)

            if u[i].left == 'd':
                to_add = to_add[1:, 1:]
            if u[i].right == 'd':
                to_add = to_add[:-1, :-1]

            Ds.append(to_add)
        for i in range(len(u)):
            if u[i].a != 'infty' and u[i].b != 'infty':
                x = spectral.chebNodes(k)
                w = j[i](u[i].map(x))/u[i].dmap(x)
                Id = np.eye(u[i].n)
                to_add = self.integr_base(Id, Id, w)
                if u[i].left == 'd':
                    to_add = to_add[1:, 1:]
                if u[i].right == 'd':
                    to_add = to_add[:-1, :-1]
                Is.append(to_add)
            else:
                # w, x = integr.reg_32_wn(n=50, merged=True)
                # w = j[i](u[i].map(x)) / u[i].dmap(x)
                Id = np.eye(u[i].n)
                to_add = self.integr_trapz(Id, Id, lambda x: j[i](u[i].map(x)) / u[i].dmap(x))
                if u[i].left == 'd':
                    to_add = to_add[1:, 1:]
                if u[i].right == 'd':
                    to_add = to_add[:-1, :-1]
                Is.append(to_add)

        res = 0                                                #kronsum
        for i in range(len(u)):
            for j in range(len(u)):
                if j == 0 and i == 0:
                    tmp = Ds[j]
                elif j == 0 and i != 0:
                    tmp = Is[j]

                if i == j and j != 0:
                    tmp = np.kron(tmp, Ds[j])
                elif i != j and j != 0:
                    tmp = np.kron(tmp, Is[j])
            res += tmp

        # print('K time', time.time() - t)
        return res
    def integr_bK(self, u, j=None, k=10):
        Ds = []; Is = []
        import time as time
        t = time.time()
        if j is None:
            j = []
            for i in range(u.dim):
                j.append(lambda x: x*0 + 1)
        for i in range(len(u)):
            jd = lambda x: j[i](u[i].map(x))*u[i].dmap(x)
            x = spectral.chebNodes(k)
            x /= 1 + 1e-8
            if u[i].a != 'infty' and u[i].b != 'infty':
                x = spectral.chebNodes(k)
                x /= 1 + 1e-8
                w = j[i](u[i].map(x))/u[i].dmap(x)
                Id = np.eye(u[i].n)
                to_add = self.integr_base(Id, Id, w)
                if u[i].left == 'd':
                    to_add = to_add[1:, 1:]
                if u[i].right == 'd':
                    to_add = to_add[:-1, :-1]
                Is.append(to_add)
            else:
                # w, x = integr.reg_32_wn(n=50, merged=True)
                # w = j[i](u[i].map(x)) / u[i].dmap(x)
                Id = np.eye(u[i].n)
                to_add = self.integr_trapz(Id, Id, lambda x: j[i](u[i].map(x)) / u[i].dmap(x))
                if u[i].left == 'd':
                    to_add = to_add[1:, 1:]
                if u[i].right == 'd':
                    to_add = to_add[:-1, :-1]
                Is.append(to_add)

            iD = np.zeros([u[i].n, u[i].n])
            r = 1 - 1e-8;  l = -1 + 1e-8
            if u[i].left == 'i':
                iD[:, 0] += jd(l) * u[i].D[0, :] / 2
                iD[0, :] += jd(l) * u[i].D[0, :] / 2
            if u[i].right == 'i':
                iD[:, -1] -= jd(r) * u[i].D[-1, :] / 2
                iD[-1, :] -= jd(r) * u[i].D[-1, :] / 2

            if u[i].left == 'd':
                iD[:, 0] += jd(l) * u[i].D[0, :] / 2
                iD[0, :] += jd(l) * u[i].D[0, :] / 2
            if u[i].right == 'd':
                iD[:, -1] -= jd(r) * u[i].D[-1, :] / 2
                iD[-1, :] -= jd(r) * u[i].D[-1, :] / 2

            if u[i].left == 'd':
                iD = iD[1:, 1:]
            if u[i].right == 'd':
                iD = iD[:-1, :-1]

            if u[i].left == 'n':
                pass
            if u[i].right == 'n':
                pass
            Ds.append(iD)

        res = 0                                     # kronsum
        for i in range(len(u)):
            if i == 0:
                tmp = Ds[0]
            else:
                tmp = Is[0]

            for j in range(1, len(u)):
                if i == j:
                    tmp = sparse.kron(tmp, Ds[j])
                elif i != j:
                    tmp = sparse.kron(tmp, Is[j])
            res += tmp
        return res
    def integr_I(self, u, j=None, k=10):
        Is = []
        if j is None:
            j = []
            for i in range(u.dim):
                j.append(lambda x: x*0 + 1)
        for i in range(len(u)):
            if u[i].a != 'infty' and u[i].b != 'infty':
                x = spectral.chebNodes(k)
                # x /= 1 + 1e-8
                w = j[i](u[i].map(x))/u[i].dmap(x)
                Id = np.eye(u[i].n)
                to_add = self.integr_base(Id, Id, w)
                if u[i].left == 'd':
                    to_add = to_add[1:, 1:]
                if u[i].right == 'd':
                    to_add = to_add[:-1, :-1]
                Is.append(to_add)
            else:
                Id = np.eye(u[i].n)
                to_add = self.integr_trapz(Id, Id, lambda x: j[i](u[i].map(x)) / u[i].dmap(x))
                if u[i].left == 'd':
                    to_add = to_add[1:, 1:]
                if u[i].right == 'd':
                    to_add = to_add[:-1, :-1]
                Is.append(to_add)
        tmp = Is[0]
        for i in range(1, len(u)):
            tmp = sparse.kron(tmp, Is[i])
        return tmp
    def integr_bI1(self, f, g, w, axis, dim, k=10):
        tmp = self.integr_base(f[0], g[axis][0], w[0])
        for i in range(1, dim):
            tmp = sparse.kron(tmp, self.integr_base(f[i], g[axis][i], w[i]))
        return tmp
    def integr_I1(self, f, w, dim, k=10):
        Ds = [];
        Is = []
        for i in range(len(u)):  # 1d_matrix_prep
            x = spectral.chebNodes(k);
            x /= 1 + 1e-8
            w = j[i](u[i].map(x)) * u[i].dmap(x)

            to_add = self.integr_base(u[i].D, u[i].D, w)

            if u[i].left == 'd':
                to_add = to_add[1:, 1:]
            if u[i].right == 'd':
                to_add = to_add[:-1, :-1]

            Ds.append(to_add)
        for i in range(len(u)):
            if u[i].a != 'infty' and u[i].b != 'infty':
                x = spectral.chebNodes(k)
                w = j[i](u[i].map(x)) / u[i].dmap(x)
                Id = np.eye(u[i].n)
                to_add = self.integr_base(Id, Id, w)
                if u[i].left == 'd':
                    to_add = to_add[1:, 1:]
                if u[i].right == 'd':
                    to_add = to_add[:-1, :-1]
                Is.append(to_add)
            else:
                # w, x = integr.reg_32_wn(n=50, merged=True)
                # w = j[i](u[i].map(x)) / u[i].dmap(x)
                Id = np.eye(u[i].n)
                to_add = self.integr_trapz(Id, Id, lambda x: j[i](u[i].map(x)) / u[i].dmap(x))
                if u[i].left == 'd':
                    to_add = to_add[1:, 1:]
                if u[i].right == 'd':
                    to_add = to_add[:-1, :-1]
                Is.append(to_add)

        res = 0  # kronsum
        for i in range(len(u)):
            for j in range(len(u)):
                if j == 0 and i == 0:
                    tmp = Ds[j]
                elif j == 0 and i != 0:
                    tmp = Is[j]

                if i == j and j != 0:
                    tmp = np.kron(tmp, Ds[j])
                elif i != j and j != 0:
                    tmp = np.kron(tmp, Is[j])
            res += tmp
    def calcF(self, u, f):
        grids = []
        np.set_printoptions(precision=3, suppress=True)
        for i in range(len(u)):
            nx = u[i].map
            grids.append(nx(spectral.chebNodes(u[i].n)))
        mshgrid = np.meshgrid(*grids, indexing='ij')
        fx = f(*mshgrid)
        nans = np.isnan(fx)
        fx[nans] = 0
        return fx

class base_interactions():
    def elem_weight_integration(self, elem, weight, tol=1e-12, N=40, maxN = 1000):

        cheb_nodes = spectral.chebNodes(elem.n)
        Id = np.eye(elem.n)

        if elem.left == 'd':
            Id[0, 0] = 0
        if elem.right == 'd':
            Id[-1, -1] = 0
        poly = sp_interp.BarycentricInterpolator(xi=cheb_nodes, yi=Id)

        w1, n1 = integr.reg_32_wn(n=N, merged=True)
        w2, n2 = integr.reg_32_wn(n=2*N, merged=True)

        p1 = poly(n1); p2 = poly(n2)
        n1 = elem.map(n1); n2 = elem.map(n2)
        fw1 = weight(n1); fw2 = weight(n2)

        I1 = np.dot(p1.T*fw1, w1); I2 = np.dot(p2.T*fw2, w2)

        while np.max(np.abs(I2 - I1))/np.max(I2) > tol:
                I1 = I2; N *= 2
                w, n = integr.reg_32_wn(n=2*N, merged=True)
                p = poly(n)
                n = elem.map(n)
                fw = weight(n)
                I2 = np.dot(p.T*fw, w)
                if N > maxN:
                    print('not converged')
                    return False
        return I2

    def grad_weight_integration(self, elem, weight, tol=1e-12, N=40, maxN=1000):
            cheb_nodes = spectral.chebNodes(elem.n)
            Id = np.eye(elem.n)

            if elem.left == 'd':
                Id[0, 0] = 0
            if elem.right == 'd':
                Id[-1, -1] = 0
            poly = sp_interp.BarycentricInterpolator(xi=cheb_nodes, yi=Id)

            w1, n1 = integr.reg_32_wn(n=N, merged=True)
            w2, n2 = integr.reg_32_wn(n=2 * N, merged=True)

            p1 = poly(n1); p2 = poly(n2)
            n1 = elem.map(n1); n2 = elem.map(n2)
            fw1 = weight(n1); fw2 = weight(n2)

            I1 = np.dot(p1.T * fw1, w1); I2 = np.dot(p2.T * fw2, w2)

            while np.max(np.abs(I2 - I1)) / np.max(I2) > tol:
                I1 = I2;  N *= 2
                w, n = integr.reg_32_wn(n=2 * N, merged=True)
                p = poly(n)
                n = elem.map(n)
                fw = weight(n)
                I2 = np.dot(p.T * fw, w)
                if N > maxN:
                    print('not converged')
                    return False
            return I2

class space():

    def __init__(self, elems, J, dim):
        self.elemsList = []
        self.J = J
        self.dim = dim
        for it in elems:
            self.elemsList.append(it)
    def __len__(self):
        return len(self.elemsList)
    def __getitem__(self, item):
        return self.elemsList[item]


    def grad(self):
        gradList = [None] * len(self)
        for i in range(len(self)):
            gradList[i] = [None] * self.dim
        for j in range(self.dim):
            for i in range(len(self)):
                gradList[i][j] = self.elemsList[i][j].D
                gradList[i][j] = np.eye(self.elemsList[i][j].n)
        return gradList

    def lambda_grad(self):
        gradList = self.elemsList.copy()
        for i in range(self.len()):
            gradList[i][i] = gradList[i][i].D
            for j in range(self.len()):
                if i == j:
                    pass
                else:
                    gradList[i][j] = np.eye(gradList[i][j].n)


class old_fem():

    def graddot(self, u, Id=False):
        N = len(u)
        Ks = [None] * N
        for i in range(N):
            Ks[i] = [None] * N
        inters = interactions()
        Is = []
        if Id == False:
            for i in range(N):
                A = sparse.csc_matrix(inters.integr_K(u[i], u.J) + inters.integr_bK(u[i], u.J))
                Is.append(inters.integr_I(u[i], u.J))
                Ks[i][i] = A

                for j in range(i + 1, N):
                        A = inters.bK(u[i], u[j], u.J)
                        if A is None:
                            Ks[i][j] = None; Ks[j][i] = None
                        else:
                            Ks[i][j] = sparse.csc_matrix(A.T); Ks[j][i] = sparse.csc_matrix(A)
                if N > 1:
                    K = sparse.csc_matrix(sparse.bmat(Ks))

                else:
                    K = Ks[0][0]
                return K
        else:
            for i in range(N):
                A = sparse.csc_matrix(inters.integr_K(u[i], u.J) + inters.integr_bK(u[i], u.J))

                Ks[i][i] = A
                Is.append(inters.integr_I(u[i], u.J))
                dims = u[i].getN()
                dims[0] -= 1; dims[1] -= 2; dims[2] -= 2
                from new_misc import approx
                approx.matrixTT(A.toarray(), dims, tol=1e-6)
                for j in range(i + 1, N):
                    A = inters.bK(u[i], u[j], u.J)
                    if A is None:
                        Ks[i][j] = None; Ks[j][i] = None
                    else:
                        Ks[i][j] = sparse.csc_matrix(A.T); Ks[j][i] = sparse.csc_matrix(A)
            if N > 1:
                K = sparse.csc_matrix(sparse.bmat(Ks))
            else:
                K = Ks[0][0]
            return K, sparse.block_diag(Is)
    def dot(self, lhs, rhs, dim):
        tmpList = []
        for i in range(dim):
            for j in range(dim):
                tmp = integr.clenshaw(lhs[i][j], rhs[i][j])
    def fdot(self, u, f):
        Fs = np.empty([0])
        inters = interactions()
        for it in u:
            Fs = np.hstack((Fs, inters.fI(it[0], f, u.J[0])))
        return Fs
    def construct1D(self, u, f):
        L = (self.graddot(u))
        F = self.fdot(u, f)
        return L, F
    def plot1D(self, space, sol):
        pn = 0

        for i in range(len(space)):
            n = space[i][0].n; x = space[i][0].map(spectral.chebNodes(n))
            if space[i][0].right == 'd':
                x = x[:-1]
            plt.plot(x, sol[pn: pn + n])
            pn += n
        plt.show()

class fem():

    def int1(self, u, v, w):
        xl = spectral.chebNodes(n=u.shape[0]); xr = spectral.chebNodes(n=v.shape[0])
        lPoly = sp_interp.BarycentricInterpolator(xi=xl, yi=u); rPoly = sp_interp.BarycentricInterpolator(xi=xr, yi=v)
        iw, ix = integr.reg_32_wn(n=70, merged=True)
        lhs1 = ((lPoly(ix).T) * w(ix)).T
        rhs1 = rPoly(ix)
        A = lhs1[:, None, :] * rhs1[:, :, None]
        A = np.tensordot(A, iw, (0, 0))
        return A

    def grad1(self, u):
        u = (diff.cheb_diff(n=u.shape[0]).dot(u))
        return u

    def dot1(self, u, v):
        return u, v

    def fint1(self, u, f, w, k=70):
        xl = spectral.chebNodes(n=u.shape[0]); xf = spectral.chebNodes(n=k)
        lPoly = sp_interp.BarycentricInterpolator(xi=xl, yi=u); fPoly = sp_interp.BarycentricInterpolator(xi=xf, yi=f)
        iw, ix = integr.reg_32_wn(n=k, merged=True)
        lhs1 = ((lPoly(ix).T) * w(ix)).T
        rhs1 = fPoly(ix)
        A = lhs1[:, None, :] * rhs1[:, :, None]
        A = np.tensordot(A, iw, (0, 0))
        return A

    def matrix1(self, a, b, space):
        n = len(space)
        for i in range(n):
            for j in range(n):
                A = a(space[i], space[i], space[i].j())
                B = b(space[i], space[i].j())


    # def matrix(self, a, b, space):
    #     for it:












# intervals = [[0, 1], [0, 1]]
# # J = [[lambda x: x, 3], [lambda x: x*0 + 1, 3]]
# # lefts = [True, True]
# # rights = [True, True]
# # inters = interactions()
# ns = np.ones(2)*10
# # meh = elem(intervals, ns, J, lefts, rights)
#
# intervals = np.array([[1, 2], [0, 1]])
# approx_obj = approx()
# a = approx_obj.fSVD(lambda x, y: np.exp(-np.sqrt(x**2 + y**2)), intervals, ns)
# inters = interactions()
# meh = baseElem([0, 'infty'], 5, [lambda x: x*x, 3], True, True)
# fI = inters.fI(meh, lambda x: np.exp(-x))
# inters.bK(meh, meh2)

# elem = new_elem(interval=[[0, 1], [0, 1], [0, 1]], n=15*np.array([1, 1, 1]))
# inters = new_interactions()
# a = inters.svdTT_FI(elem,
#              lambda x, y, z: np.exp(-np.sqrt(x*x + y*y + z*z)),
#              j=[lambda x: x*0 + 1, lambda x: x*0 + 1, lambda x: x*0 + 1])
# print(a.size)


