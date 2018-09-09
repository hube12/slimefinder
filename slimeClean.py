from multiprocessing import Process, cpu_count, freeze_support, Queue
import numpy as np
from math import ceil, floor
import time


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

def javaInt64(val):
    return ((val + (1 << 63)) % (1 << 64)) - (1 << 63)

def itsASlime(cx, cz, worldseed):
    return not nextInt(10, (javaInt64(
            worldseed + cx * cx * 4987142 + cx * 5947611 + cz * cz * 4392871 +
            cz * 389711)) ^ 987234911)

def initialize(r, s, w, offset):
    a = np.zeros((s, s), dtype=bool)
    for i in range(s):
        for j in range(s):
            a[i][j] = itsASlime(-r + j, i + offset, w)
    return a

def goDown(a, nbr, s, x, z, w):
    a = a[nbr:]
    b = np.zeros((nbr, s), dtype=bool)
    for i in range(nbr):
        for j in range(s):
            b[i][j] = itsASlime(x + j, z + s + i, w)
    return np.concatenate((a, b))

def goRight(a, nbr, s, x, z, w):
    for i in range(s):
        for j in range(nbr):
            a[i] = np.concatenate((a[i][1:], [itsASlime(x + s + j, z + i, w)]))
    return a

def checkMask(mask, layer):
    return np.array_equal(mask, layer)


def workers(mask, index, offset, seed, size, radius, cores, result):
    block = initialize(radius, size, seed, offset * cores + index)
    if checkMask(mask, block):
        result.put((0, offset * cores + index))
    for i in range(-radius, radius - 1):
        block = goRight(block, 1, size, i, offset * cores + index, seed)
        if checkMask(mask, block):
            result.put((i + 1, offset * cores + index))

def main(radius, seed, size, mask):
    assert size, radius > 0
    result = []
    processPool = []
    result_queue = Queue()
    cores = cpu_count()
    t = time.time()
    for offset in range(-floor(radius / cores), ceil(radius / cores)):
        for i in range(cpu_count()):
            p = Process(target=workers, args=(mask, i, offset, seed, size, radius, cores, result_queue))
            p.daemon = True
            p.start()
            processPool.append(p)
        for el in processPool:
            p.join()
        result_queue.put("DONE")
        while True:
            temp = result_queue.get()
            if temp == "DONE":
                break
            result.append(temp)
        if not offset % cores:
            print("{} %".format(round(offset / (2 * radius / cores) * 100 + 50, 2)))
            print(time.time()-t)
            t = time.time()
    print(result)

def start():
    t = time.time()
    freeze_support()
    size = 16
    seed = 2
    radius = 20000
    mask = np.zeros((size, size), dtype=bool)
    main(radius, seed, size, mask)
    print(time.time() - t)
    print("The results are in chunks compared to 0 0, also you need to read it as chunkX,chunkZ")


if __name__ == '__main__':
    freeze_support()
    start()
