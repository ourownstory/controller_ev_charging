import argparse
import gym
# import sys
# sys.path.append('/home/ourownstory/github/gym_ev_charging')
# import gym_ev_charging
import sys
sys.path.append('../gym_ev_charging')
from config_controller import get_config as get_controller_config
from gym_ev_charging.config_gym import get_config as get_gym_config
import controller


def run_experiment():
    args = parser.parse_args()
    config = get_controller_config(args.controller, args.env_config)
    env_config = get_gym_config(args.env_config)
    env = gym.make(env_config.env_name)
    env.build(env_config)

    model = getattr(controller, config.controller_name)(env, config)
    model.run_training()
    model.run_evaluation(num_episodes=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', required=True, type=str)
    parser.add_argument('--env_config', required=True, type=str)
    run_experiment()
