from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import pickle
import numpy as np

import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.sample_batch import DEFAULT_POLICY_ID
from ray.tune.util import merge_dicts
from ray.tune.registry import register_env
from ray.rllib.evaluation.policy_graph import clip_action
from env import KlotskiEnv

EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""
register_env("klotski", lambda env_config: KlotskiEnv(env_config))


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
                    "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        help="The algorithm or model to train. This may refer to the name "
             "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
             "user-defined trainable function or class registered in the "
             "tune registry.")
    required_named.add_argument(
        "--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument("--steps", default=1e10, help="Number of steps to roll out.")
    parser.add_argument("--episodes", default=100, help="Number of episodes to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument("--sample-action", default=False, action="store_const", const=True,
                        help="Choose best action or sample action")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
             "Surpresses loading of configuration from checkpoint.")
    return parser


def get_config(args):
    config = {}
    # Load configuration from file
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory.")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
    config["num_gpus"] = 0
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            raise ValueError("the following arguments are required: --env")
        args.env = config.get("env")
    return config


def get_agent(args, config):
    cls = get_agent_class(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    return agent


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def rollout(agent, env_name, num_steps, num_episodes, out=None, no_render=True, sample_action=False):
    def policy_agent_mapping(x):
        return DEFAULT_POLICY_ID

    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env
        multiagent = False

        if agent.local_evaluator.multiagent:
            policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]

        policy_map = agent.local_evaluator.policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: m.action_space.sample()
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    rollout_data = []
    total_timesteps = 0
    episodes = 0

    # one rollout
    while (total_timesteps < (num_steps or total_timesteps + 1)) and episodes < num_episodes:
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()
        agent_states = DefaultMapping(lambda x: state_init[mapping_cache[x]])
        prev_actions = DefaultMapping(lambda x: action_init[mapping_cache[x]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        total_reward = 0.0
        steps_this_episode = 0

        # one episode
        while not done and total_timesteps < (num_steps or total_timesteps + 1):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        if not sample_action:
                            policy = agent.get_policy(policy_id)
                            preprocessed = agent.local_evaluator.preprocessors[policy_id].transform(a_obs)
                            filtered_obs = agent.local_evaluator.filters[policy_id](preprocessed, update=False)
                            a_action = policy.sess.run(policy.logits,
                                                       feed_dict={
                                                           policy.observations: np.expand_dims(filtered_obs, 0)
                                                       })[0]
                            a_action = clip_action(a_action, env.action_space)
                            a_action = np.argmax(a_action)
                        else:
                            a_action = agent.compute_action(
                                a_obs,
                                prev_action=prev_actions[agent_id],
                                prev_reward=prev_rewards[agent_id],
                                policy_id=policy_id)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, _ = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                total_reward += list(reward.values())[0]
            else:
                total_reward += reward
            if not no_render:
                env.render()
            total_timesteps += 1
            steps_this_episode += 1
            obs = next_obs

        if out is not None:
            rollout_data.append([total_reward, steps_this_episode])
        print("Episode reward", total_reward)
        episodes += 1

    if out is not None:
        pickle.dump(rollout_data, open(out, "wb"))

    return rollout_data


if __name__ == "__main__":
    ray.init(num_cpus=2, num_gpus=0, object_store_memory=int(5e+9), redis_max_memory=int(2e+9))
    parser = create_parser()
    args = parser.parse_args()
    config = get_config(args)
    agent = get_agent(args, config)
    rollout(agent,
            args.env,
            int(args.steps),
            int(args.episodes),
            out=args.out,
            no_render=args.no_render,
            sample_action=args.sample_action)
