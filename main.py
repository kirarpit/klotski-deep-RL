import ray
from ray import tune
from ray.tune.registry import register_env
import sys
import os
from klotski_env import KlotskiEnv
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

register_env("klotski", lambda env_config: KlotskiEnv())


def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["depth"] = []


def on_episode_step(info):
    episode = info["episode"]
    info = episode.last_info_for()
    if info is not None and "depth" in info:
        episode.user_data["depth"].append(info["depth"])


def on_episode_end(info):
    episode = info["episode"]
    episode.custom_metrics["max_depth"] = np.max(episode.user_data["depth"])


tune.run(
    "PPO",
    name="klotski",
    config={
        "env": "klotski",
        "num_workers": 1,
        "sample_batch_size": 64,
        "train_batch_size": 4096,
        "sgd_minibatch_size": 512,
        "num_sgd_iter": 10,
        "observation_filter": "MeanStdFilter",
        "batch_mode": "complete_episodes",
        "env_config": {},
        "model": {
            "fcnet_hiddens": [64, 64],
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
