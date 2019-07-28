from klotski_env import KlotskiEnv
import copy
import time
import pickle

env = KlotskiEnv()
env.reset()
queue = [(env, 0)]
visited_states = set()
state_depth = {}

tic = time.time()
cnt = 0
while len(queue):
    cnt += 1
    if cnt % 100 == 0:
        print(len(visited_states))
        print("Time taken {}".format(time.time() - tic))

    current_env, level = queue.pop(0)

    if current_env.get_simple_state() in visited_states:
        continue

    state_depth[current_env.get_simple_state()] = level
    visited_states.add(current_env.get_simple_state())

    if current_env.is_over:
        continue

    for action in current_env.get_valid_actions():
        new_env = copy.deepcopy(current_env)
        new_env.step(action)

        if new_env.get_simple_state() not in visited_states:
            queue.append((new_env, level+1))

with open('state_depth.pickle', 'wb') as handle:
    pickle.dump(state_depth, handle, protocol=pickle.HIGHEST_PROTOCOL)
