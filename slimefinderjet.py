from multiprocessing import Process,Array,cpu_count,freeze_support,sharedctypes
import numpy as np
from matplotlib import pyplot as plt

def next(seed):
    seed = (seed * 0x5deece66d + 0xb) & ((1 << 48) - 1)
    retval = seed >> (48 - 31)
    if retval & (1 << 31):
        retval -= (1 << 32)
    return retval, seed


def nextInt(n, seed):
    seed = (seed ^ 0x5deece66d) & ((1 << 48) - 1)
    retval, seed = next(seed)
    if not (n & (n - 1)):
        return (n * retval) >> 31
    else:
        bits = retval
        val = bits % n

        while (bits - val + n - 1) < 0:
            bits, seed = next(seed)
            val = bits % n
        return val


def javaInt32(val):
    return ((val + (1 << 31)) % (1 << 32)) - (1 << 31)


def javaInt64(val):
    return ((val + (1 << 63)) % (1 << 64)) - (1 << 63)


def itsASlime(cx, cz, worldseed):
    return not nextInt(10, (javaInt64(
        worldseed + javaInt32(cx * cx * 4987142) + javaInt32(cx * 5947611) + javaInt64(cz * cz * 4392871) + javaInt32(
            cz * 389711))) ^ 987234911)


def initialize(r, s, w):
    a = np.zeros((s, s), dtype=bool)
    for i in range(s):
        for j in range(s):
            a[i][j] = itsASlime(-r + j, -r + i, w)
    return a


def goDown(a, nbr, s, x, z, w):
    a = a[nbr:]
    b = np.zeros((nbr, s), dtype=bool)
    for i in range(nbr):
        for j in range(s):
            b[i][j] = itsASlime(x + j, z + s + i, w)
    return np.concatenate((a, b))


def goLeft(a, nbr, s, x, z, w):
    i = 0
    for i in range(s):
        for j in range(nbr):
            a[i] = np.concatenate((a[i][1:], [itsASlime(x + s + j, z + i, w)]))
    return a

def checkMask(mask,layer):
    return np.array_equal(mask,layer)


def oldgoleft(a, nbr, s, x, z, w):
    i=0
    for el in np.nditer(a, op_flags=['readwrite'], flags=['external_loop'], order='C'):
        b = np.zeros(nbr, dtype=bool)
        for j in range(nbr):
            b[j] = itsASlime(x + s + j, z + i, w)
            print(x + s + j, z + i)
        print("d")
        el[...] = np.concatenate((el[nbr:], b))

        i += 1

    return a


def f(common,value):

    print(value)
# size in in chunks,radius too
def main(radius, seed, size):
    processPool=[]
    core = cpu_count()
    assert size > 0
    arr=initialize(radius, size, seed)
    print(arr.dtype)
    arr=np.ctypeslib.as_ctypes(arr)
    arr=sharedctypes.RawArray(arr._type_, arr)

    for i in range(4):
        p=Process(target=f,args=(arr,i))
        p.start()
        processPool.append(p)
    for el in processPool:
        p.join()


def test():
    print(cpu_count())
    seed = 1
    l = [(9, 36), (9, 37), (8, 36)]
    j = [1, 0, 1]
    for e, el in zip(l, j):
        if not itsASlime(e[0], e[1], seed) == el:
            print(e)
    print('done, errors above')
    a = initialize(0, 10, seed)
    plt.imshow(a, interpolation='nearest')
    plt.show()
    b = goDown(a, 4, 10, 0, 0, seed)
    plt.imshow(b, interpolation='nearest')
    plt.show()
    c = goLeft(b, 2, 10, 0, 4, 1)
    plt.imshow(c, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    freeze_support()
    main(10,1,10)
