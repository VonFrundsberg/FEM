import numpy as np

class Mesh():

    def __init__(self):
        return
    def gen_mesh(self, I, n=10, p=3):
        x = np.linspace(I[0], I[1], n)
        l = []
        neigh = []
        for i in range(n - 1):
            l.append([i, x[i], x[i + 1], p])
            if i > 0:
                neigh.append([i - 1, i])
        l = np.array(l)
        self.l = l
        self.neigh = neigh
    def file_write(self, lname, nname):
        np.savetxt(lname, self.l)
        f = open(nname, "w+")
        for it in self.neigh:
            f.writelines(list(map(lambda x: str(x) + ' ', it)))
            f.write('\n')
        f.close()
        return
    def file_read(self, lname, nname):
        self.l = np.genfromtxt(lname)
        f = open(nname, "r+")
        self.neigh = f.readlines()
        for i in range(len(self.neigh)):
            self.neigh[i] = self.neigh[i].split(' ')[:-1]
            self.neigh[i] = np.array(self.neigh[i], dtype=np.float)
        f.close()
        return