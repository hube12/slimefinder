from multiprocessing import Process, Array, cpu_count, freeze_support, Value, Queue
import numpy as np
from math import ceil, floor
from matplotlib import pyplot as plt
import time
import ctypes


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


# cx is horizontal and cz is vertical
def itsASlime(cx, cz, worldseed):
    return not nextInt(10, (javaInt64(
        worldseed + javaInt32(cx * cx * 4987142) + javaInt32(cx * 5947611) + javaInt64(cz * cz * 4392871) + javaInt32(
            cz * 389711))) ^ 987234911)


def initialize(r, s, w, offset):
    a = np.zeros((s, s), dtype=bool)
    for i in range(s):
        for j in range(s):
            a[i][j] = itsASlime(-r + j, i + offset, w)
    return a


# this method will not be used as implemented as we initialize an area for each workers in the same time then go right
# but if we instead initialize an area then cut it in "cores" pieces then do each part with the workers and when done
# move down then this method can be useful
def goDown(a, nbr, s, x, z, w):
    a = a[nbr:]
    b = np.zeros((nbr, s), dtype=bool)
    for i in range(nbr):
        for j in range(s):
            b[i][j] = itsASlime(x + j, z + s + i, w)
    return np.concatenate((a, b))


"""Shift the current array by one column, removing the first one and appending the last,
it takes in arguments the array (a), the number oh shifts to do (usually one), the size 
of the area (aka the array lenght), the x and z coordinate to which it should start, the
worldseed (w)"""


def goRight(a, nbr, s, x, z, w):
    for i in range(s):
        for j in range(nbr):
            a[i] = np.concatenate((a[i][1:], [itsASlime(x + s + j, z + i, w)]))
    return a


def checkMask(mask, layer):
    return np.array_equal(mask, layer)


"""We first initialize a grid of size*size, we do it on the upper left corner of the radius
area (it ranges from -radius,-radius to +radius,+radius, ofc we shift that initialization
by how many rows we already did, then we extend with goRight the strip and search through it
for a pattern that match our mask.
eg:
initialization:

-5 -4 -3 -2 -1 0 +1 +2 +3 +4 +5

 0  1  0  1                     -5 + offset
 1  1  1  0                     -4 + offset
 0  0  0  1                     -3 + offset
 0  1  0  1                     -2 + offset
 
 one hop:
 
-5 -4 -3 -2 -1 0 +1 +2 +3 +4 +5

    1  0  1  0                  -5 + offset
    1  1  0  1                  -4 + offset
    0  0  1  1                  -3 + offset
    1  0  1  1                  -2 + offset
    
and so on, moving by strip, so moving after initialization 2*radius-size+1 so here with:
radius=5 and size=4 its 7 goRight called
 
"""


def workers(mask, index, offset, seed, size, radius, cores, result):
    block = initialize(radius, size, seed, offset * cores + index)
    if checkMask(mask, block):
        result.put((0, offset * cores + index))
    for i in range(-radius, radius - 1):
        block = goRight(block, 1, size, i, offset * cores + index, seed)
        if checkMask(mask, block):
            result.put((i+1, offset * cores + index))



# size is a square (default is 16x16), radius is a positive number referring to how large one quadrant (square) should be,
# then we extend at the 3 others, so basically its a square from -radius,-radius to +radius,+radius
def main(radius, seed, size, mask):
    assert size, radius > 0
    result = []
    processPool = []
    result_queue = Queue()

    # we share everything among the workers, technically its not needed but as we will instantiate quite a lot of them,
    # better be safe

    # cores,seed, size, radius  = Value('d',cpu_count()), Value('d', seed), Value('d', size), Value('d', radius)
    cores = cpu_count()
    for offset in range(-floor(radius / cores), ceil(radius / cores)):
        for i in range(cpu_count()):
            p = Process(target=workers, args=(mask, i, offset, seed, size, radius, cores, result_queue))
            p.daemon = True
            p.start()
            processPool.append(p)

        # waiting for the childs processes to catch up
        for el in processPool:
            p.join()
        result_queue.put("DONE")
        while True:
            temp=result_queue.get()

            if temp=="DONE":
                break
            result.append(temp)
    print(result)


def showAGrid(a):
    fig, ax = plt.subplots()
    im = ax.imshow(a)
    ax.set_xticks(np.arange(len(a) + 1) - 0.5, minor=1)
    ax.set_yticks(np.arange(len(a[0]) + 1) - 0.5, minor=1)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    t=time.time()
    freeze_support()
    size = 12
    seed = 2
    radius = 5000
    mask=np.zeros((size,size),dtype=bool)
    main(radius, seed, size, mask)
    print(time.time()-t)
    """a = initialize(radius, radius * 2 + size - 1, seed, -radius)
    showAGrid(a)"""
    print("The results are in chunks compared to 0 0, also you need to read it as chunkX,chunkZ")

"""


def oldgoright(a, nbr, s, x, z, w):
    i = 0
    for el in np.nditer(a, op_flags=['readwrite'], flags=['external_loop'], order='C'):
        b = np.zeros(nbr, dtype=bool)
        for j in range(nbr):
            b[j] = itsASlime(x + s + j, z + i, w)
            print(x + s + j, z + i)
        print("d")
        el[...] = np.concatenate((el[nbr:], b))

        i += 1

    return a


def test2():
    def f(common, value):
        if value == 1:
            common[0] = True
        print(common[0], value)

    # size in in chunks,radius too
    def main(radius, seed, size):
        processPool = []
        core = cpu_count()
        assert size > 0
        arr = np.concatenate(initialize(radius, size, seed), axis=0)
        print(arr)
        shared_arr = Array(ctypes.c_bool, arr)
        print(shared_arr)

        for i in range(4):
            p = Process(target=f, args=(shared_arr, i))
            p.start()
            processPool.append(p)
        for el in processPool:
            p.join()

    main(10, 1, 10)


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
    c = goRight(b, 2, 10, 0, 4, 1)
    plt.imshow(c, interpolation='nearest')
    plt.show()


"""
