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

def run_parllel():
    output = mp.Queue()

    # processes = [mp.Process(
    #     target=count, args=(np.random.randint(low=1, high=8), output))
    #              for _ in range(3)]

    seeds = [np.random.randint(0, 100) for _ in range(8)]

    processes = [mp.Process(
        target=single_rollout, args=(output, seed))
                 for seed in seeds]

    [p.start() for p in processes]
    [p.join() for p in processes]

    results = [output.get() for p in processes]

    return results

if __name__ == '__main__':

    def backprop(infos, stats):

        for info in infos:
            action = info['action']
            mc_return = info['return']

            stats[action].append(float(mc_return))

        return stats

    def rollouts(stats):
        infos = run_parllel()
        stats = backprop(infos, stats)
        return stats

    def summarize(stats):
        summary = {}
        stat = namedtuple('stat', ['mean', 'std'])

        for action, returns in stats.items():
            summary[action] = stat(np.mean(returns), np.std(returns))

        print(summary)
        return summary

    def policy(stats):

        actions, returns = [], []

        for action, mc_return in stats.items():
            actions.append(action)
            returns.append(mc_return)

        return actions[np.argmax(returns)]

    stats = defaultdict(list)
    for _ in range(100):
        stats = rollouts(stats)
        summarize(stats)

        action = policy(stats)
        print(action)
