""" first look at mutliprocessing """

import multiprocessing as mp
import random
from time import time, sleep


def count(output):
    for num in range(4):
        sleep(1)
        print(num)
    output.put(num)


def timer(func):
    start = time()
    func()
    end = time()
    print(start-end)


def timeit(method):
    def timed(*args, **kw):
        ts = time()
        result = method(*args, **kw)
        te = time()
        print(te - ts)
        return result
    return timed


@timeit
def parallel():
    output_queue = mp.Queue()

    processes = [mp.Process(target=count, args=(output_queue, ))
                 for _ in range(4)]
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    #  exit processes
    [p.join() for p in processes]

    # #  get results
    return [output_queue.get() for p in processes]


@timeit
def serial():
    for _ in range(4):
        _ = count()


if __name__ == '__main__':
    random.seed(42)
    results = parallel()
