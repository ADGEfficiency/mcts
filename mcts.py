"""

could use a energy_py memory strucutre (maybe?)
"""
from energypy.common.memories import calculate_returns

from collections import namedtuple, defaultdict

import numpy as np

import energypy


def single_rollout(length=None):
    """ using uniform random policy means we don't need any stats """
    done = False
    actions = env.action_space.discretize(10)

    s = env.reset()

    max_steps = 10

    steps = 0
    while not done:
        action = actions[np.random.randint(actions.shape[0])]
        s, r, done, i = env.step(action)

        if length and steps == max_steps:
            done = True

        steps += 1

    rewards = i['reward']
    mc_returns = calculate_returns(rewards, 1.0)
    i['return'] = mc_returns

    return i

def array_to_tuple(arr):
    return tuple(map(tuple, arr))

def backprop(info, stats):
    states = info['state']
    actions = info['action']
    returns = info['return']

    # first_action = array_to_tuple(actions[0])
    first_action = actions[0]
    stats[first_action].append(float(returns[0]))

    return stats

def process_stats(stats):
    for action, returns in stats.items():
        print('{} {}'.format(action, np.mean(returns)))


if __name__ == '__main__':

    env = energypy.make_env('flex')

    stats = defaultdict(list)

    rollouts = 500
    for rollout in range(rollouts):

        info = single_rollout()

        stats = backprop(info, stats)

        if rollout % 10 == 0:
            process_stats(stats)




