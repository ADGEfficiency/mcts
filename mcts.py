from collections import defaultdict, namedtuple
from copy import deepcopy
import multiprocessing as mp
from time import sleep

import numpy as np

import energypy
from energypy.common.memories import calculate_returns


class Agent():

    def __init__(
            self,
            env,
            njobs,
            num_rollouts,
            max_len,
    ):
        self.env = env
        self.njobs = njobs
        self.num_rollouts = num_rollouts
        self.max_len = max_len

    def single_rollout(self, output, env, seed):
        """ using uniform random policy means we don't need any stats """

        max_len = 36

        done = False
        actions, rewards = [], []
        step = 0
        while not done:
            action = env.action_space.sample_discrete()
            s, r, done, info = env.step(action)

            actions.append(action)
            rewards.append(r)

            if step >= max_len:
                done = True
            else:
                step += 1

        mc_returns = calculate_returns(
            rewards, discount=1.0
        )

        out = {
            'return': mc_returns[0],
            'action': actions[0][0][0]
        }
        output.put(out)


    def run_parllel(self, env):
        output = mp.Queue()

        seeds = [np.random.randint(0, 100) for _ in range(self.njobs)]

        processes = [mp.Process(
            target=self.single_rollout, args=(output, env, seed))
                     for seed in seeds]

        [p.start() for p in processes]
        [p.join() for p in processes]

        results = [output.get() for p in processes]

        return results

    def backprop(self, infos, stats):

        for info in infos:
            action = info['action']
            mc_return = info['return']

            stats[action].append(float(mc_return))

        return stats

    def rollouts(self, stats, env):
        infos = self.run_parllel(env)
        stats = self.backprop(infos, stats)
        return stats

    def summarize(self, stats):
        summary = {}
        stat = namedtuple('stat', ['mean', 'std'])

        for action, returns in stats.items():
            summary[action] = stat(np.mean(returns), np.std(returns))

        return summary

    def policy(self, summary):

        actions, returns = [], []

        for action, stat in summary.items():
            actions.append(action)
            returns.append(stat.mean)

        print(actions, returns)
        idx = np.argmax(returns)
        best = actions[idx]

        return np.array(best).reshape(1,1)

    def get_action(self, env):

        stats = defaultdict(list)

        for _ in range(self.num_rollouts):
            stats = self.rollouts(stats, deepcopy(env))

        summary = self.summarize(stats)

        return self.policy(summary)

if __name__ == '__main__':

    import sacred
    ex = sacred.Experiment('expt')

    @ex.config
    def config():
        njobs = 6
        num_rollouts = 1000
        max_len = 3 * 12

    @ex.capture
    def experiment(njobs, num_rollouts, max_len):
        done = False
        env = energypy.make_env(
            'battery', 
            episode_length=20, 
            episode_sample='fixed'
        )
        agent = Agent(env, njobs, num_rollouts, max_len)

        _  = env.action_space.discretize(20)
        s = env.reset()

        len = 0
        while not done:
            action = agent.get_action(deepcopy(env))
            s, r, done, info = env.step(action)

            print(len, action, np.sum(info['reward']), info['reward'][-5:])
            len += 1

        print(np.sum(info['reward']))
        print(len(info['reward']))

    @ex.automain
    def my_main():
        experiment()
