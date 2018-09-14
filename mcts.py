import multiprocessing as mp
from energypy.common.memories import calculate_returns

from collections import namedtuple, defaultdict
from time import sleep

import numpy as np

import energypy

NJOBS=4


def count(rand, output):
    for num in range(rand):
        sleep(1)
    output.put(num)


def single_rollout(output, env, seed):
    """ using uniform random policy means we don't need any stats """

    done = False
    actions, rewards = [], []
    while not done:
        action = env.action_space.sample_discrete()
        s, r, done, info = env.step(action)

        actions.append(action)
        rewards.append(r)

    mc_returns = calculate_returns(
        rewards, discount=1.0
    )

    out = {
        'return': mc_returns[0],
        'action': actions[0][0][0]
    }
    output.put(out)


def run_parllel(env):
    output = mp.Queue()

    seeds = [np.random.randint(0, 100) for _ in range(NJOBS)]

    processes = [mp.Process(
        target=single_rollout, args=(output, env, seed))
                 for seed in seeds]

    [p.start() for p in processes]
    [p.join() for p in processes]

    results = [output.get() for p in processes]

    return results

def backprop(infos, stats):

    for info in infos:
        action = info['action']
        mc_return = info['return']

        stats[action].append(float(mc_return))

    return stats

def rollouts(stats, env):
    infos = run_parllel(env)
    stats = backprop(infos, stats)
    return stats

def summarize(stats):
    summary = {}
    stat = namedtuple('stat', ['mean', 'std'])

    for action, returns in stats.items():
        summary[action] = stat(np.mean(returns), np.std(returns))

    return summary

def policy(summary):

    actions, returns = [], []

    for action, stat in summary.items():
        actions.append(action)
        returns.append(stat.mean)

    idx = np.argmax(returns)
    best = actions[idx]

    return np.array(best).reshape(1,1)

def get_action(env):

    stats = defaultdict(list)

    for _ in range(50):
        stats = rollouts(stats, deepcopy(env))

    summary = summarize(stats)

    return policy(summary)

if __name__ == '__main__':
    from copy import deepcopy

    done = False
    env = energypy.make_env('cartpole-v0')
    actions = env.action_space.discretize(20)
    s = env.reset()

    len = 0
    while not done:
        action = get_action(deepcopy(env))
        s, r, done, info = env.step(action)
        print(len)
        len += 1

    print(len(info['reward']))
