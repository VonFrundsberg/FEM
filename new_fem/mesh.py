from new_fem.b_mesh import *
import time as time
from new_misc import approx
import itertools as iter
import re
class Mesh():

    def __init__(self, n):
        self.n = n
        return
    def gen_mesh(self, K, n=10, p=3):
        n = np.atleast_1d(n)
        K = np.atleast_2d(K)
        x = []
        self.box = K.copy()
        for i in range(K.shape[0]):
            x.append(np.linspace(K[i, 0], K[i, 1], n[i]))
        l = []
        neigh = []
        ind = np.zeros(len(n), dtype=np.int)
        def get_l(ind):
            tmp = []
            for i in range(len(ind)):
                tmp.append([x[i][ind[i]], x[i][ind[i] + 1]])
            return tmp
        def intersection(e1, e2):
            start = np.max(np.vstack([e1[:, 0], e2[:, 0]]), axis=0)
            end = np.min(np.vstack([e1[:, 1], e2[:, 1]]), axis=0)

            tmp = start - end
            args2 = np.where(tmp == 0)
            args1 = np.where(tmp < 0)
            inter = np.zeros(len(n))
            inter[args1] = 1
            inter[args2] = 2
            return inter

        ns = np.product(n - 1)
        s = 0; i = 0
        while i < ns:
            if s == 0:
                for k in range(n[0] - 1):
                    ind[0] = k
                    a = get_l(ind)
                    l.append([i, np.array(a), p])
                    i += 1

                s += 1
            else:
                while(ind[s] >= n[s] - 2):
                    s += 1
                ind[s] += 1
                ind[:s] = 0
                s = 0

        l = np.array(l)
        self.l = l
        self.neigh = neigh
        for i in range(len(self.l)):
            tmp = []
            for j in range(len(self.l)):
                if i != j:
                    intr = intersection(self.l[i][1], self.l[j][1])
                    if np.size(np.where(intr == 0)) == 0 and\
                                np.size(np.where(intr == 2)) == 1:
                        if (np.size(np.where(intr == 1)) > 0 and self.n > 1) or self.n == 1:
                            tmp.append([self.l[i][0], self.l[j][0]])
            if len(tmp) > 0:
                self.neigh.append(tmp)
        self.l = np.array(self.l)
        # for it in self.neigh:
        #     print(it)

    def file_write(self, lname, nname):
        f = open(lname, "w+")

        for it in self.l:
            it = np.hstack(([it[0], it[1].flatten(), np.array(it[2])]))
            f.writelines(list(map(lambda x: str(x) + ' ', it)))
            f.write('\n')
        f.close()
        f = open(nname, "w+")
        for it in self.neigh:
            f.writelines(list(map(lambda x: str(x) + ' ', it)))
            f.write('\n')
        f.close()
        return
    def file_read(self, lname, nname):
        self.l = np.genfromtxt(lname)
        tmp = []
        for it in self.l:
            tmp.append([it[0], np.reshape(it[1:2*self.n + 1], [self.n, 2]), np.array(it[2*self.n + 1:], dtype=np.int)])
        self.l = tmp
        f = open(nname, "r+")
        self.neigh = f.readlines()
        for i in range(len(self.neigh)):
            self.neigh[i] = list(filter(None, re.split(r'\[|\]| |,', self.neigh[i])))[:-1]
            self.neigh[i] = np.array(self.neigh[i], dtype=np.int)
            self.neigh[i] = np.reshape(self.neigh[i], [int(self.neigh[i].size/2), 2])
        f.close()
        return

    def extendBox(self, s, i, p):
        l1 = np.array(self.box, dtype=np.float)
        if s == 1:
            l1[i, 0] = l1[i, 1]
            l1[i, 1] = np.inf


        if s == -1:
            l1[i, 1] = l1[i, 0]
            l1[i, 0] = -np.inf

        l2 = [self.l[-1, 0] + 1, l1, p]
        n = self.l[-1, 0] + 1
        self.l = np.vstack([self.l, np.array(l2)])
        tmp = []
        for j in range(self.l.shape[0]):
            if ((self.l[j, 1][i, 1] == l1[i, 0]) or (self.l[j, 1][i, 0] == l1[i, 1])):
                tmp.append([n, j])
                self.neigh[j].append([j, n])
        self.neigh.append(tmp)

        return

    def extendBox2d(self, i, p):
        self.extendBox(-1, i[0], p)
        self.extendBox(1, i[0], p)
        self.extendBox(-1, i[1], p)
        self.extendBox(1, i[1], p)

        l = self.l[-4:]
        tmp = []
        n = self.l[-1][0]
        tmp.append([n + 1, np.vstack([l[0][1][0], l[2][1][1]]), p])
        tmp.append([n + 2, np.vstack([l[0][1][0], l[3][1][1]]), p])
        tmp.append([n + 3, np.vstack([l[1][1][0], l[3][1][1]]), p])
        tmp.append([n + 4, np.vstack([l[1][1][0], l[2][1][1]]), p])
        for it in tmp:
            self.l = np.vstack([self.l, np.array(it)])

        def intersection(e1, e2):
            start = np.max(np.vstack([e1[:, 0], e2[:, 0]]), axis=0)
            end = np.min(np.vstack([e1[:, 1], e2[:, 1]]), axis=0)

            tmp = start - end
            args2 = np.where(tmp == 0)
            args1 = np.where(tmp < 0)
            inter = np.zeros(self.n)
            inter[args1] = 1
            inter[args2] = 2
            return inter
        self.neigh = []
        for i in range(len(self.l)):
            tmp = []
            for j in range(len(self.l)):
                if i != j:
                    intr = intersection(self.l[i][1], self.l[j][1])
                    if np.size(np.where(intr == 0)) == 0 and\
                            np.size(np.where(intr == 1)) > 0 and\
                                np.size(np.where(intr == 2)) == 1:
                        tmp.append([self.l[i][0], self.l[j][0]])
            if len(tmp) > 0:
                self.neigh.append(tmp)

        return 0


    def extendBox3d(self, i, p):
        self.extendBox(-1, i[0], p)
        self.extendBox(1, i[0], p)
        self.extendBox(-1, i[1], p)
        self.extendBox(1, i[1], p)
        self.extendBox(-1, i[2], p)
        self.extendBox(1, i[2], p)

        l = self.l[-6:]
        tmp = []
        n = self.l[-1][0]
        tmp.append([n + 1, np.vstack([l[0][1][0], l[2][1][1], l[4][1][2]]), p])
        tmp.append([n + 2, np.vstack([l[1][1][0], l[2][1][1], l[4][1][2]]), p])
        tmp.append([n + 3, np.vstack([l[0][1][0], l[3][1][1], l[4][1][2]]), p])
        tmp.append([n + 4, np.vstack([l[0][1][0], l[2][1][1], l[5][1][2]]), p])

        tmp.append([n + 5, np.vstack([l[0][1][0], l[3][1][1], l[5][1][2]]), p])
        tmp.append([n + 6, np.vstack([l[1][1][0], l[2][1][1], l[5][1][2]]), p])
        tmp.append([n + 7, np.vstack([l[1][1][0], l[3][1][1], l[4][1][2]]), p])
        tmp.append([n + 8, np.vstack([l[1][1][0], l[3][1][1], l[5][1][2]]), p])

        tmp.append([n + 9, np.vstack([l[0][1][0], l[2][1][1], self.box[2]]), p])
        tmp.append([n + 10, np.vstack([l[1][1][0], l[2][1][1], self.box[2]]), p])
        tmp.append([n + 11, np.vstack([l[0][1][0], l[3][1][1], self.box[2]]), p])
        tmp.append([n + 12, np.vstack([l[1][1][0], l[3][1][1], self.box[2]]), p])

        tmp.append([n + 13, np.vstack([l[0][1][0], self.box[1], l[4][1][2]]), p])
        tmp.append([n + 14, np.vstack([l[1][1][0], self.box[1], l[4][1][2]]), p])
        tmp.append([n + 15, np.vstack([l[0][1][0], self.box[1], l[5][1][2]]), p])
        tmp.append([n + 16, np.vstack([l[1][1][0], self.box[1], l[5][1][2]]), p])

        tmp.append([n + 17, np.vstack([self.box[0], l[2][1][1], l[4][1][2]]), p])
        tmp.append([n + 18, np.vstack([self.box[0], l[3][1][1], l[4][1][2]]), p])
        tmp.append([n + 19, np.vstack([self.box[0], l[2][1][1], l[5][1][2]]), p])
        tmp.append([n + 20, np.vstack([self.box[0], l[3][1][1], l[5][1][2]]), p])
        for it in tmp:
            self.l = np.vstack([self.l, np.array(it)])
        def intersection(e1, e2):
            start = np.max(np.vstack([e1[:, 0], e2[:, 0]]), axis=0)
            end = np.min(np.vstack([e1[:, 1], e2[:, 1]]), axis=0)

            tmp = start - end
            args2 = np.where(tmp == 0)
            args1 = np.where(tmp < 0)
            inter = np.zeros(self.n)
            inter[args1] = 1
            inter[args2] = 2
            return inter
        self.neigh = []
        for i in range(len(self.l)):
            tmp = []
            for j in range(len(self.l)):
                if i != j:
                    intr = intersection(self.l[i][1], self.l[j][1])
                    if np.size(np.where(intr == 0)) == 0 and\
                            np.size(np.where(intr == 1)) > 0 and\
                                np.size(np.where(intr == 2)) == 1:
                        tmp.append([self.l[i][0], self.l[j][0]])
            if len(tmp) > 0:
                self.neigh.append(tmp)

        return 0

    def __getitem__(self, item):
        return [self.l[item][1], self.l[item][2]]

    def intersection(self, e1, e2):
            start = np.max(np.vstack([e1[:, 0], e2[:, 0]]), axis=0)
            end = np.min(np.vstack([e1[:, 1], e2[:, 1]]), axis=0)
            res = (np.vstack([start, end]).T)
            return res


# msh = Mesh(3)
# msh.gen_mesh(np.array([[0, 3], [-2, 2], [-3, 4]]), n=[3, 3, 3], p=[3, 3, 3])
# msh.extendBox(1, 0, p=[3, 3, 3])
# msh.file_write('1.txt', '2.txt')

# msh.file_read('1.txt', '2.txt')
# msh.file_write('1.txt', '2.txt')

