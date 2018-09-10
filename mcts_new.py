import multiprocessing as mp
from energypy.common.memories import calculate_returns

from collections import namedtuple, defaultdict
from time import sleep

import numpy as np

import energypy


def count(rand, output):
    for num in range(rand):
        sleep(1)
        print(num)
    output.put(num)


def single_rollout(output, seed):
    """ using uniform random policy means we don't need any stats """

    print('rollout started')
    done = False
    env = energypy.make_env('flex')
    env.seed(seed)
    actions = env.action_space.discretize(10)

    s = env.reset()

    while not done:
        action = env.action_space.sample_discrete()
        s, r, done, i = env.step(action)

    mc_returns = calculate_returns(
        i['reward'], discount=1.0
    )

    out = {
        'return': mc_returns[0],
        'action': i['action'][0]
    }
    # out = np.random.uniform()
    output.put(out)

    print(out)
    print('rollout finished')

def run_parllel():
    output = mp.Queue()

    # processes = [mp.Process(
    #     target=count, args=(np.random.randint(low=1, high=8), output))
    #              for _ in range(3)]


    processes = [mp.Process(
        target=single_rollout, args=(output, seed))
                 for seed in [40, 50, 5]]

    [p.start() for p in processes]
    [p.join() for p in processes]

    results = [output.get() for p in processes]

    return results

if __name__ == '__main__':
    results = run_parllel()
    print(results)
