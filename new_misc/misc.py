import numpy as np
import time as time
def xtremes(A):
    Marg = np.argmax(A)
    marg = np.argmin(A)
    maxs = np.array([Marg])
    mins = np.array([marg])
    # print(A)
    # print(A[Marg])

    while A[Marg + 1:].size > 0:
        Marg = np.argmax(A[Marg + 1:]) + maxs[-1] + 1
        # print(Marg)
        maxs = np.append(maxs, Marg)
    while A[marg + 1:].size > 0:
        marg = np.argmin(A[marg + 1:]) + mins[-1] + 1
        mins = np.append(mins, marg)

    return np.sort(np.hstack([maxs, mins]))

def to_rgba(arr, a=1, div=15, min=2, M=30):
        max = np.max(arr)
        arr = np.atleast_1d(arr)
        low_ind = np.argwhere(arr < div); high_ind = np.argwhere(arr >= div)
        r = np.zeros(arr.size); g = np.zeros(arr.size); b = np.zeros(arr.size)
        a = np.ones(arr.size)*a
        r[low_ind] = (1 - np.abs(arr - min)**(3/3)/max)[low_ind]
        g[low_ind] = 1 - r[low_ind]
        b[high_ind] = (1 - np.abs(arr - M)**(3/3)/max)[high_ind]
        g[high_ind] = 1 - b[high_ind]

        high_ind = np.argwhere(arr > M)
        b[high_ind] = 1; g[high_ind] = 0

        low_ind = np.argwhere(arr < min)
        r[low_ind] = 1; g[low_ind] = 0

        res = np.vstack([r, g, b, a])
        return res
    # else:
    #     max = np.max(arr)
    #     if arr < div:
    #         r = (1 - np.abs(arr - 2) ** (3 / 2) / max)
    #         g = 1 - r
    #     else:
    #         g = (1 - np.abs(arr - 15) ** (3 / 2) / max)
    #         b = 1 - g
    #
    #     res = np.vstack([r, g, b, a])
    #     return res
    # print(r)


# v = np.random.rand(5)*10 + 1
# print(v)
# to_rgba(v)
# xtremes(v)