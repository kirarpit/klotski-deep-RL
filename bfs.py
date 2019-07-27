from klotski_env import KlotskiEnv
import copy
import time
import ray
import sys
import os

head_ip = None
if len(sys.argv) > 1:
    head_node = sys.argv[1]
    head_ip = os.popen("host " + head_node + " | awk '{print $4}'").read()

if not ray.is_initialized():
    if head_ip is not None:
        ray.init(redis_address=head_ip + ":6379")
    else:
        ray.init()

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

    if current_env.get_state_id() in visited_states:
        continue

    state_depth[current_env.get_state_id()] = level
    visited_states.add(current_env.get_state_id())

    if current_env.is_over:
        continue

    for action in current_env.get_valid_actions():
        new_env = copy.deepcopy(current_env)
        new_env.step(action)

        if new_env.get_state_id() not in visited_states:
            queue.append((new_env, level+1))
