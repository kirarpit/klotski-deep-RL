import ray
from ray import tune
from ray.tune.registry import register_env
import sys
from env import KlotskiEnv
import os.path
import pickle
import numpy as np

head_ip = None
if len(sys.argv) > 1:
    head_node = sys.argv[1]
    head_ip = os.popen("host " + head_node + " | awk '{print $4}'").read()

if not ray.is_initialized():
    if head_ip is not None:
        ray.init(redis_address=head_ip + ":6379")
    else:
        ray.init()

register_env("klotski", lambda env_config: KlotskiEnv(env_config))

# Load state_depth dictionary
state_depth = None
if os.path.exists('state_depth.pickle'):
    with open('state_depth.pickle', 'rb') as handle:
        state_depth = pickle.load(handle)


def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["depth"] = []


def on_episode_step(info):
    episode = info["episode"]
    info = episode.last_info_for()
    if info is not None and "simple_state" in info:
        episode.user_data["depth"].append(state_depth[info["simple_state"]])


def on_episode_end(info):
    episode = info["episode"]
    episode.custom_metrics["max_depth"] = np.max(episode.user_data["depth"])
    episode.custom_metrics["visited_states_cnt"] = episode.last_info_for()["visited_states_cnt"]


checkpoint = None
if len(sys.argv) > 2:
    checkpoint = sys.argv[2]

tune.run(
    "PPO",
    name="klotski",
    restore=checkpoint,
    config={
        "env": "klotski",
        "num_workers": 1,
        "sample_batch_size": 64,
        "train_batch_size": 200,
        "sgd_minibatch_size": 32,
        "num_sgd_iter": 10,
        "batch_mode": "complete_episodes",
        "lambda": 0.98,
        "env_config": {
            "max_steps": 500,
            "novelty_scheme": "frequency",
            "rewards": {
                "invalid_move": 0,
                "max_steps": -1,
                "novel_state": 1,
                "per_step_penalty": 0,
            }
        },
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        "callbacks": {
            "on_episode_start": tune.function(on_episode_start),
            "on_episode_step": tune.function(on_episode_step),
            "on_episode_end": tune.function(on_episode_end),
        },
    },
    stop={
        "time_total_s": 60 * 60 * 2
    },
    checkpoint_at_end=True,
    checkpoint_freq=10,
    queue_trials=True,
)
