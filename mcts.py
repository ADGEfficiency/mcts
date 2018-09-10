""" could use a energy_py memory strucutre (maybe?) """

import multiprocessing as mp
from energypy.common.memories import calculate_returns

from collections import namedtuple, defaultdict

import numpy as np

import energypy


def single_rollout(output):
    """ using uniform random policy means we don't need any stats """

    print('rollout started')
    done = False
    actions = env.action_space.discretize(10)

    s = env.reset()

    while not done:
        action = actions[np.random.randint(
            low=0, actions.shape[0])]
        print(action)
        s, r, done, i = env.step(action)

    rewards = i['reward']
    mc_returns = calculate_returns(rewards, 1.0)
    i['return'] = mc_returns
    output.put(i)
    print('rollout finished')


def array_to_tuple(arr):
    return tuple(map(tuple, arr))


def backprop(info, stats):
    states = info['state']
    actions = info['action']
    returns = info['return']

    first_action = actions[0]
    stats[first_action].append(float(returns[0]))

    return stats


def process_stats(stats):
    for action, returns in stats.items():
        print('{} {}'.format(action, np.mean(returns)))


if __name__ == '__main__':

    env = energypy.make_env('flex')

    stats = defaultdict(list)

    batches = 2
    batch_size = 4

    output_queue = mp.Queue()

    processes = [mp.Process(target=single_rollout, args=(output_queue, ))
                 for _ in range(batch_size)]

    [p.start() for p in processes]
    out = [output_queue.get() for p in processes]
    [p.join() for p in processes]

    for info in out:
        print(stats)
        stats = backprop(info, stats)
