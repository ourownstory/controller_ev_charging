import argparse
import gym
import sys
sys.path.append('../gym_ev_charging')
import gym_ev_charging

from config_controller import get_config as get_controller_config
from gym_ev_charging.config_gym import get_config as get_gym_config
import controller
import utils_controller as utils

parser = argparse.ArgumentParser()
parser.add_argument('--controller', required=True, type=str)
parser.add_argument('--env_config', required=True, type=str)


def get_configs():
    args = parser.parse_args()
    # read with helper fun
    config = get_controller_config(args.controller, args.env_config)
    env_config = get_gym_config(args.env_config)
    # sync episode lengths
    config.max_ep_len = env_config.EPS_LEN
    config.max_ep_len_eval = env_config.EVAL_EPS_LEN
    # overwrite gamma for experiments
    config.gamma = env_config.gamma
    # save to JSON
    out_path = config.output_path
    utils.save_object(config, out_path, name="config_controller")
    utils.save_object(env_config, out_path, name="config_env")
    return config, env_config


def run_experiment():
    # get config
    config, env_config = get_configs()
    # create env
    env = gym.make(env_config.ENV_NAME)
    env.build(env_config)
    # create model
    model = getattr(controller, config.controller_name)(env, config)
    # train
    model.run_training()
    # eval
    model.run_evaluation()


if __name__ == '__main__':
    run_experiment()
    print("EXPERIMENT DONE.")
