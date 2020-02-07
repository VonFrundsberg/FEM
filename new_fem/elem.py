from new_fem.b_elem import *
from new_fem.mesh import *
from new_misc import integr as intg

class elem():
    def __init__(self, K, n, bc=None, Id=None):
        self.dims = []
        for i in range(len(n)):
            if bc[i] is not None:
                self.dims.append(b_elem(K[i, :], n[i], bc=bc[i]))
            else:
                self.dims.append(b_elem(K[i, :], n[i]))
    # def rp_eval(self):
    #     Id = self.Id
    #     for it in self.bc:
    #         Id[it[0], it[0]] = it[1]
    #     return Id
    # def rp_plot(self):
    #     P = self.Id
    #     x = np.linspace(-1, 1, 100)
    #     plt.plot(x, sp.bary(P, x))
    #     plt.show()
    def p_eval(self, x): ## return shape: (*x.shape, n)
        res = []
        i = 0
        for it in self.dims:
            res.append(it.p_eval(np.unique(x[i])))
            i += 1
        # for it in res:
        # for it in res:
        #     print(it.shape)
        # print(res[0].shape)
        # res = np.prod(np.array(np.ix_(*res)))
        # res = np.reshape(res, [])
        # res = np.array(res)
        # res
        return res
    def dp_eval(self, x):
        res = []
        i = 0
        for it in self.dims:
            res.append(it.dp_eval(np.unique(x[i])))
            i += 1
        # res = np.array(res)

        return res
    def bcs(self):
        lists = np.zeros(len(self.dims))
        pad = np.zeros([len(self.dims), 2], dtype=np.int)
        pad = pad.tolist()
        # pad = [[0, 0], [0, 0]]
        for i in range(len(self.dims)):

            lists[i] += len(self.dims[i].bc)
            for it in self.dims[i].bc:
                if it[0] == 0:
                    pad[i][0] = 1
                if it[0] == -1:
                    pad[i][1] = 1
        return lists, pad
    def grid(self):
        grid = []
        for i in range(len(self.dims)):
            grid.append(self.dims[i].xs())
        return grid
    def gen_func(self):
        return lambda x: self.p_eval(x)
    def gen_funcd(self):
        return lambda x: self.dp_eval(x)

    def __call__(self, x):
            return self.p_eval(x)
    def d(self, x):
            return self.dp_eval(x)

class func():
    def __init__(self):
        return
    def inner(self, g1, g2):
        res = []
        # print(g1)
        # print(len(g1))
        # print(len(g1[0]))
        if isinstance(g1[0], list):
            for i in range(len(g1)):
                tmp_res = []
                for j in range(len(g1[i])):
                    lhs = g1[i][j]
                    rhs = g2[i][j]
                    tmp = lhs[:, :, None]*rhs[:, None, :]
                    x, y, z = tmp.shape
                    tmp = np.reshape(tmp, [x, y*z])
                    tmp_res.append(tmp)
                res.append(tmp_res)
        else:

                tmp_res = []
                for j in range(len(g1)):
                    lhs = g1[j]
                    rhs = g2[j]
                    tmp = lhs[:, :, None]*rhs[:, None, :]
                    x, y, z = tmp.shape
                    tmp = np.reshape(tmp, [x, y*z])
                    tmp_res.append(tmp)
                res.append(tmp_res)
        return res
    def grad(self, elem):
        fs = lambda x: elem(x)
        ds = lambda x: elem.d(x)
        def res(x):
            dsx = ds(x)
            fsx = fs(x)
            arr = []
            for i in range(len(elem.dims)):
                tmp = []
                for j in range(len(elem.dims)):
                    if i == j:
                        tmp.append(dsx[j])
                    else:
                        tmp.append(fsx[j])
                arr.append(tmp)
            return arr
        return res
    def integr(self, K, elemF, F=None, n=500):
        grids = []
        weights = []
        for i in range(K.shape[0]):
            if K[i, 0] != K[i, 1]:
                tmp = b_elem(K[i], n)
                w, x = intg.reg_32_wn(-1, 1, n, merged=True)
                w /= tmp.dmap(x)
                x = tmp.map(x)
                grids.append(x)
                weights.append(w)
            else:
                grids.append(K[i, 0])
                weights.append(np.ones(1))
        fx = elemF(grids)
        msh = approx.meshgrid(*grids)
        if K.shape[0] > 1:
            if F is not None:
                f = F(*msh)
                u = approx.decay_smth(f)
                for i in range(len(u)):
                    u[i] *= weights[i][None, :, None]
            else:
                u = []
                for i in range(K.shape[0]):
                    u.append(weights[i][None, :, None])
            v = 0
            for it in fx:
                    v += approx.contraction(u, it)
        else:
            if F is not None:
                f = F(*msh)
            else:
                f = 1
            v = 0
            for it in fx:
                v += np.sum(f*it[0].T*weights[0], axis=1)
        return v


    def orth(self, g, K1, K2):
        meshObj = Mesh(K1.shape[0])
        b = meshObj.intersection(K1, K2)
        print(K1, K2)
        print('and then')
        for i in range(b.shape[0]):
            if b[i, 0] == b[i, 1]:
                k = i
                if np.mean(K1[k, :]) > b[i, 0]:
                    print('silence')
                    g[k][k] *= -1
                return g[k]

    def edge(self, g, K1, K2):
        flag = False
        for i in range(K1.shape[0]):
            # print('hey')
            # print(K1[i])
            # print(K2[i])
            if np.sum(np.abs(K1[i] - K2[i])) > 0:
                flag = True; break
        if flag == False:
            return g
        else:

            for i in range(K1.shape[0]):
                g[0][i] *= 1
        # print(g[0][0])
        return g





# msh = Mesh(1)
# N = 3
# msh.gen_mesh(np.array([[0, 1]]), n=[4], p=[N])
# msh.extendBox(0, 0, p=[N])
# msh = Mesh(3)
# N = 3
# msh.gen_mesh(np.array([[0, 1], [0, 1], [0, 1]]), n=[4, 4, 4], p=[N, N, N])
# msh.extendBox(1, 0, p=[N, N, N])
# msh.file_write('1.txt', '2.txt')
# msh.file_read('1.txt', '2.txt')
# it = msh.l[1]
# elem_obj = elem(K=it[1], n=it[2], bc=[[[-1, 0]],
#                                       [[-1, 0]],
#                                       [[-1, 0]]])
#
# it = msh.l[0]
# elem_obj1 = elem(K=it[1], n=it[2])
# func_obj = func()
# F = lambda x, y, z: np.exp(-np.sqrt(x**2 + y**2 + z**2))
# # I = func_obj.integr(K=np.array([[1.5, 3], [0, 2], [0.5, 4]]), elemF1=lambda x: elem_obj(x), elemF2=lambda x: elem_obj(x), F=f)
# # I = func_obj.grad(elem_obj)
# # time.sleep(500)
# # Ix = I([x, x, x])
# # print(Ix)
# np.set_printoptions(precision=3, suppress=True)
# # for i in range(len(Ix)):
# #     # print(Ix[i][2].shape)
# #     # time.sleep(500)
# #     print(str(i + 1))
# #     for j in range(len(Ix[i])):
# #         Ix[i][j] = np.array(Ix[i][j])
# #         print(Ix[i][j].shape)
# # inr = func_obj.inner(Ix, Ix)
# # F = lambda x: func_obj.inner(func_obj.grad(elem_obj)(x), func_obj.grad(elem_obj)(x))
# # I = func_obj.integr(K=np.array([[1.5, 3], [0, 2], [0.5, 4]]), elemF=F)
# FL = lambda x: func_obj.inner([(func_obj.grad(elem_obj1)(x))[0]], [elem_obj1(x)])
# # FR = lambda x: func_obj.inner([elem_obj(x)], [elem_obj(x)])
# # IL = func_obj.integr(K=np.array([[1, np.inf], [0, 1], [0, 1]]), elemF=FL)
# # IL = func_obj.integr(K=np.array([[1], [0, 1], [0, 1]]), elemF=FL)
# # IR = func_obj.integr(K=np.array([[1, np.inf], [0, 1], [0, 1]]), elemF=FR)
# rhs = func_obj.integr(K=np.array([[1, np.inf], [0, 1], [0, 1]]), elemF=lambda x: [elem_obj(x)], F=F)
# # print(tryit.shape)
# # print('done')
# from new_misc import spectral
# # time.sleep(500)
# # print()
# IL = np.reshape(IL, np.array(np.sqrt(np.repeat(IL.shape, 2)), dtype=np.int))
# # print(IL.shape)
# IL = np.swapaxes(IL, 1, 2)
# IL = np.swapaxes(IL, 2, 4)
# IL = np.swapaxes(IL, 3, 4)
#
# # IR = np.reshape(IR, np.array(np.sqrt(np.repeat(IR.shape, 2)), dtype=np.int))
# # IR = np.swapaxes(IR, 1, 2)
# # IR = np.swapaxes(IR, 2, 4)
# # IR = np.swapaxes(IR, 3, 4)
# sh = IL.shape
# IL = np.reshape(IL, [np.prod(sh[:3]), np.prod(sh[3:])])
# # print(IL)
# time.sleep(500)
# # sh = IR.shape
# # IR = np.reshape(IR, [np.prod(sh[:3]), np.prod(sh[3:])])
#
# # IR = IR[~np.all(np.abs(IR) < 1e-10, axis=1)]
# # IR = IR.T[~np.all(np.abs(IR.T) < 1e-10, axis=1)].T
# #
# rhs = np.reshape(rhs, [np.prod(rhs.shape)])
# rhs = rhs[~np.all(np.abs(IL) < 1e-10, axis=1)]
# IL = IL[~np.all(np.abs(IL) < 1e-10, axis=1)]
# IL = IL.T[~np.all(np.abs(IL.T) < 1e-10, axis=1)].T
# # print(IL.shape)
# import scipy.linalg as sp_linalg
# print()
# sol = sp_linalg.solve(IL, rhs)
#
# # args = np.argsort(np.real(sol[0]))
# # print(args)
# # print(sol[0][args[1]])
# meh = sol

# meh = np.real(np.reshape(meh, [N - 1, N - 1, N - 1]))
# plt.imshow(meh[:, 0, :])
# plt.show()
# print(IL.shape, IR.shape)
# I = func_obj.integr(K=np.array([[1.5, 3], [0, 2], [0.5, 4]]), elemF=F, F=f)
    # print(Ix.shape)
# print(Ix)

# print(I.shape)
# elem_obj = elem(K=it[1], n=it[2])
# x = 3 + np.sort(np.random.rand(500, 3), axis=0)
# res = elem_obj.dp_eval(x)
# plt.plot(x[:, 0], res[0])
# plt.show()
# msh.file_write('1.txt', '2.txt')

