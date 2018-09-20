import copy

import energy_py

#  check seeding & check rollouts are the same for the same actions


env = energy_py.make_env('flex')
env.reset()

master_env = copy.deepcopy(env)

s, r, d, i = env.step(1)
out = env.step(0)

check = master_env.step(1)

