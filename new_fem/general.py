from new_fem.mesh import *
from new_fem.elem import *
import scipy.sparse as sparse
import scipy.sparse.linalg as sp_linalg
class general:
    def initMesh(self, mesh):
        self.mesh = mesh
    def bilinearForm(self, a, b1, b2):
        self.iform = a
        self.bform1 = b1
        self.bform2 = b2
    def rhs(self, f):
        self.fform = f
    def initBC(self, bc):
        self.bc = bc
    def mreshape(self, s1, s2, tensor):
        shape = np.hstack([s1, s2])
        sort = 0
        if shape.size == 2:
            shape = shape[[0, 1]]
            sort = np.argsort([0, 1])
        if shape.size == 4:
            shape = shape[[0, 2, 1, 3]]
            sort = np.argsort([0, 2, 1, 3])
        if shape.size == 6:
            shape = shape[[0, 3, 1, 4, 2, 5]]
            sort = np.argsort([0, 3, 1, 4, 2, 5])

        tensor = np.reshape(tensor, shape)
        tensor = np.transpose(tensor, sort)
        tensor = np.reshape(tensor, [np.prod(s1), np.prod(s2)])
        return tensor
    def initElems(self):
        N = len(self.mesh.l)
        self.elems = [None] * N
        for i in range(N):
            # print('next elem')
            x = self.mesh[i][0]
            args = [None] * self.mesh.n
            for j in range(len(self.bc)):
                arg = np.argwhere(self.mesh[i][0][self.bc[j][0], :] == self.bc[j][1])
                if len(arg) > 0:
                    if args[self.bc[j][0]] is not None:
                        args[self.bc[j][0]].append(np.hstack([-np.squeeze(arg), self.bc[j][2]]))
                    else:
                        args[self.bc[j][0]] = [np.hstack([-np.squeeze(arg), self.bc[j][2]])]
            args = np.array(args)
            self.elems[i] = elem(*self.mesh[i], bc=args)
    def calcMatrixElems(self):
        N = len(self.mesh.l)
        self.matrixElems = [None] * N
        self.rhs = [None] * N
        for i in range(N):
            self.matrixElems[i] = [None] * N
        for i in range(N):
            tmp1 = self.iform(self.elems[i], self.elems[i], K=self.mesh[i][0])
            tmp1 = self.mreshape(self.mesh[i][1], self.mesh[i][1], tmp1)
            self.matrixElems[i][i] = tmp1
            self.rhs[i] = (self.fform(self.elems[i], K=self.mesh[i][0])).flatten()
            for it in self.mesh.neigh[i]:
                K1 = self.mesh[i][0]
                K2 = self.mesh[it[1]][0]
                self.matrixElems[i][i] += self.mreshape(self.mesh[i][1], self.mesh[i][1],
                                                        self.bform1(self.elems[i], self.elems[i], K1=K1, K2=K2))
                if i < it[1]:
                    self.matrixElems[i][it[1]] = self.mreshape(self.mesh[i][1], self.mesh[it[1]][1],
                                                            self.bform2(self.elems[i], self.elems[it[1]], K1=K1, K2=K2))
                else:
                    self.matrixElems[i][it[1]] = self.matrixElems[it[1]][i].T
    def solve(self):
        # t = time.time()
        A = sparse.bmat(self.matrixElems)
        A = sparse.csr_matrix(A)
        A = A[A.getnnz(1) > 0, :][:, A.getnnz(0) > 0]

        rhs = np.hstack(self.rhs)

        sol = sp_linalg.spsolve(A, rhs[np.nonzero(rhs)])
        sols = []
        i1 = 0
        for i in range(len(self.mesh.l)):
            ps = self.mesh[i][1]
            bcs, pad = self.elems[i].bcs()
            ps = np.array(ps - bcs, dtype=np.int)
            i2 = i1 + np.prod(ps)
            tmp = sol[i1: i2]
            tmp = np.reshape(tmp, ps)
            tmp = np.pad(tmp, pad, 'constant')
            sols.append(tmp)
            i1 = i2
        self.sol = sols

    def get1d(self):
        flatten = lambda l: [item for sublist in l for item in sublist]
        meh1 = []
        meh2 = []
        for i in range(len(self.elems)):
            meh1.append(self.elems[i].grid())
            meh2.append(self.sol[i])
        meh1 = (np.stack(flatten(meh1))).flatten()
        meh2 = np.stack(flatten(meh2))
        res = np.squeeze(np.array([meh1, meh2]))
        return res


    def plot2d(self):
        fig = plt.figure()
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.gca(projection='3d')

        for i in range(len(self.elems)):
            flag = True
            grid = self.elems[i].grid()
            x1, x2 = approx.meshgrid(*grid)
            for j in range(self.mesh.n):
                if np.size(grid[j][np.isinf(grid[j])]) != 0:
                    flag = False
            if flag == True:
                mehf = lambda x, y: np.exp(-a*((x)**2 + (y)**2))
                ax.plot_surface(x1, x2, self.sol[i][:, :] - mehf(x1, x2))

        plt.show()
    def plot3d(self):
        from mayavi import mlab
        figure = mlab.figure('DensityPlot')
        wevecome = []
        alonglong = []
        way = []
        together = []

        for i in range(len(self.elems)):
            flag = True
            grid = self.elems[i].grid()
            for j in range(self.mesh.n):
                if np.size(grid[j][np.isinf(grid[j])]) != 0:
                    flag = False
            if flag == True:
                mehf = lambda x, y, z: np.exp(-A * ((x) ** 2 + (y) ** 2 + z**2))
                msh = approx.meshgrid(*grid)
                self.sol[i] = self.sol[i] - mehf(*msh)
                a = self.sol[i].flatten()

                wevecome.append(msh[0].ravel())
                alonglong.append(msh[1].ravel())
                way.append(msh[2].ravel())
                together.append(a)

        mlab.points3d(wevecome, alonglong, way, together)
        mlab.axes()
        mlab.colorbar()
        mlab.show()

    def eval(self, x):
        return 0




gen_obj = general()
f = func()
msh = Mesh(3)
N = 7
msh.gen_mesh(np.array([[-5, 5], [-5, 5], [-5, 5]], dtype=np.float), n=[3, 3, 3], p=[N, N, N])

msh.extendBox3d([0, 1, 2], [N, N, N])

msh.file_write('1.txt', '2.txt')
msh.file_read('1.txt', '2.txt')
gen_obj.initMesh(msh)
I = lambda u, v, K: f.integr(K=K, elemF=lambda x: f.inner(f.grad(u)(x), f.grad(v)(x)))

bI1 = lambda u, v, K1, K2: -0.5*f.integr(K=msh.intersection(K1, K2), elemF=lambda x: f.inner(f.orth(f.grad(u)(x), K1, K2), v(x))) + \
                           -0.5*f.integr(K=msh.intersection(K1, K2), elemF=lambda x: f.inner(u(x), f.orth(f.grad(v)(x), K1, K2)))

bI2 = lambda u, v, K1, K2: 0.5*f.integr(K=msh.intersection(K1, K2), elemF=lambda x: f.inner(f.orth(f.grad(u)(x), K1, K2), v(x))) + \
                           -0.5*f.integr(K=msh.intersection(K1, K2), elemF=lambda x: f.inner(u(x), f.orth(f.grad(v)(x), K1, K2)))
A = 0.1
F = lambda x, y, z: -2*A*np.exp(-A*((x)**2 + (y)**2 + (z)**2))*(-3 + 2*A*(x*x + y*y + z*z))
fI = lambda v, K: f.integr(K=K, elemF=lambda x: [v(x)], F=F)
gen_obj.bilinearForm(a=I, b1=bI1, b2=bI2)
gen_obj.rhs(fI)
bc = []
bc.append([0, -np.inf, 0])
bc.append([0, np.inf, 0])
bc.append([1, -np.inf, 0])
bc.append([1, np.inf, 0])
bc.append([2, -np.inf, 0])
bc.append([2, np.inf, 0])
#
# bc.append([0, 0, 0])
# bc.append([0, 5, 0])
# bc.append([1, 0, 0])
# bc.append([1, 5, 0])

gen_obj.initBC(bc)
t1 = time.time()
gen_obj.initElems()
gen_obj.calcMatrixElems()
# gen_obj.writetxt()
print('elements calc', time.time() - t1)
t2 = time.time()
gen_obj.solve()
print('solving', time.time() - t2)
gen_obj.plot3d()


gen_obj = general()
f = func()
msh = Mesh(1)
N = 60
msh.gen_mesh(np.array([[0, 15]]), n=[3], p=[N])
msh.extendBox(1, 0, [N])
msh.file_write('1.txt', '2.txt')
msh.file_read('1.txt', '2.txt')
gen_obj.initMesh(msh)
I = lambda u, v, K: f.integr(K=K, elemF=lambda x: f.inner(f.grad(u)(x), f.grad(v)(x)), F=lambda x: x)




bI1 = lambda u, v, K1, K2: -0.5*f.integr(K=msh.intersection(K1, K2), elemF=lambda x: f.inner(f.orth(f.grad(u)(x), K1, K2), v(x)), F=lambda x: x) + \
                           -0.5*f.integr(K=msh.intersection(K1, K2), elemF=lambda x: f.inner(u(x), f.orth(f.grad(v)(x), K1, K2)), F=lambda x: x)

bI2 = lambda u, v, K1, K2: 0.5*f.integr(K=msh.intersection(K1, K2), elemF=lambda x: f.inner(f.orth(f.grad(u)(x), K1, K2), v(x)), F=lambda x: x) + \
                           -0.5*f.integr(K=msh.intersection(K1, K2), elemF=lambda x: f.inner(u(x), f.orth(f.grad(v)(x), K1, K2)), F=lambda x: x)

F = lambda x: x*x*np.exp(-x)
fI = lambda v, K: f.integr(K=K, elemF=lambda x: [v(x)], F=F)
gen_obj.bilinearForm(a=I, b1=bI1, b2=bI2)
gen_obj.rhs(fI)
bc = []
bc.append([0, np.inf, 0])
gen_obj.initBC(bc)
gen_obj.initElems()
gen_obj.calcMatrixElems()
gen_obj.solve()
sol = gen_obj.get1d()

asol = lambda x: np.exp(-x) + 2*np.exp(-x)/x - 2/x
sol[1] = np.abs(sol[1])
s1 = asol(sol[0])
s1 = np.abs(s1)
res = np.abs((sol[1] - s1))
plt.loglog(sol[0], res)
plt.show()








# msh.gen_mesh(np.array([[-5, 5], [-5, 5]], dtype=np.float), n=[3, 3], p=[N, N])
# # msh.extendBox(1, 0, [5, 5, 11])
# # msh.extendBox(-1, 0, [5, 11, 11])
# # msh.extendBox(1, 1, [11, 11, 11])
# # msh.extendBox(1, 2, [N, N])
# msh.extendBox2d([0, 1], [2*N, 2*N])
# msh.file_write('1.txt', '2.txt')
# msh.file_read('1.txt', '2.txt')
# gen_obj.initMesh(msh)
# I = lambda u, v, K: f.integr(K=K, elemF=lambda x: f.inner(f.grad(u)(x), f.grad(v)(x)))
#
# bI1 = lambda u, v, K1, K2: -0.5*f.integr(K=msh.intersection(K1, K2), elemF=lambda x: f.inner(f.orth(f.grad(u)(x), K1, K2), v(x))) + \
#                            -0.5*f.integr(K=msh.intersection(K1, K2), elemF=lambda x: f.inner(u(x), f.orth(f.grad(v)(x), K1, K2)))
#
# bI2 = lambda u, v, K1, K2: 0.5*f.integr(K=msh.intersection(K1, K2), elemF=lambda x: f.inner(f.orth(f.grad(u)(x), K1, K2), v(x))) + \
#                            -0.5*f.integr(K=msh.intersection(K1, K2), elemF=lambda x: f.inner(u(x), f.orth(f.grad(v)(x), K1, K2)))
# a = 0.1
# F = lambda x, y: -4*a*np.exp(-a*((x)**2 + (y)**2))*(-1 + a*(x*x + y*y))
# fI = lambda v, K: f.integr(K=K, elemF=lambda x: [v(x)], F=F)
# gen_obj.bilinearForm(a=I, b1=bI1, b2=bI2)
# gen_obj.rhs(fI)
# bc = []
# # bc.append([0, 0, 0])
# bc.append([0, np.inf, 0])
# bc.append([0, -np.inf, 0])
# bc.append([1, -np.inf, 0])
# bc.append([1, np.inf, 0])
# # bc.append([2, 0, 0])
# # bc.append([2, 5, 0])
# gen_obj.initBC(bc)
# t1 = time.time()
# gen_obj.initElems()
# gen_obj.calcMatrixElems()
# # gen_obj.writetxt()
# print('elements calc', time.time() - t1)
# t2 = time.time()
# gen_obj.solve()
# print('solving', time.time() - t2)
# gen_obj.plot2d()
# # gen_obj.plot3d()
print(np.log2(1e-8))