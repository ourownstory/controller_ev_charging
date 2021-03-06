from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import gym
import os

from policy_gradient.utils_pg import get_logger, Progbar
import utils_controller as utils


class MetaController(ABC):
    """
    Abstract Class for implementing a Controller
    Abstract Class for implementing a Q Network
    """

    def __init__(self, env, config, logger=None):
        """
        Initialize Q Network and env

        Args:
            config: class with hyperparameters
            logger: logger instance from logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)

        # env
        self.env = env
        # discrete vs continuous action space
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

        # some things to change during training/eval
        self.mode = None
        self.total_train_steps = 0

        # build model
        self.build()

    @abstractmethod
    def build(self):
        """
        Build modelg
        """
        raise NotImplementedError

    @property
    def policy(self):
        """
        model.policy(state) = action
        """
        return lambda state: self.get_action(state)

    def save(self):
        """
        Save model parameters

        Args:
            model_path: (string) directory
        """
        pass

    @abstractmethod
    def get_action(self, state):
        """
        Returns action with some exploration strategy

        Args:
            state: observation from gym
        """
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    def init_averages(self):
        """
        Defines extra attributes for tensorboard.
        """
        self.avg_reward = min(list(self.env.reward_range))
        self.max_reward = min(list(self.env.reward_range))
        self.std_reward = 0.
        self.eval_reward = min(list(self.env.reward_range))

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.
        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def sample_gameplay(self, env, max_ep_len, num_episodes=None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"

        """
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        while (num_episodes or t < self.config.batch_size):
            # print("episode", episode, "of", num_episodes, "t", t)
            state = env.reset()
            states, actions, rewards, infos = [], [], [], []
            episode_reward = 0
            for step in range(max_ep_len):
                t += 1
                states.append(state)

                if not self.env.config.do_not_featurize:
                    action = self.get_action(state=states[-1][None])
                else:
                    action = self.get_action(state=states[-1])

                state, reward, done, info = env.step(action)

                actions.append(action)
                rewards.append(reward)
                infos.append(info)

                episode_reward = episode_reward + reward
                if (done or step == max_ep_len - 1):
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break
            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "infos": np.array(infos)}
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break
        return paths, episode_rewards

    def record(self):
        """
        Recreate an env and record a gameplay for one episode
        """
        self.logger.info("Recording.")
        evaluating = self.mode != 'train'

        env = gym.make(self.env.config.ENV_NAME)
        env.build(self.env.config)
        env.evaluation_mode = evaluating

        # play according to policy
        paths, rewards = self.sample_gameplay(
            env=env,
            max_ep_len=self.config.max_ep_len if evaluating else self.config.max_ep_len_eval,
            num_episodes=self.config.eval_episodes if evaluating else self.config.record_episodes
        )

        # plot
        utils.plot_episodes(
            paths=paths[:self.config.plots_per_record],
            train_step=self.total_train_steps,
            env=env,
            contr_name=self.config.name,
            out_dir=self.config.plot_output,
            num=self.config.plots_per_record,
        )
        # more plots
        utils.price_energy_histogram(paths, self.config.plot_output, contr_name=self.config.name, mode=self.mode)
        # stats
        utils.compute_stats(rewards, paths, self.config, self.logger, env, save=True)

    def run_training(self):
        """
        Apply procedures of training for any model
        """
        self.mode = 'train'
        # initialize
        # self.initialize() # handled by controllers using tf in build()
        # record one game at the beginning
        if self.config.record:
            self.record()
        # model
        self.train()
        # record one game at the end
        if self.config.record:
            self.record()

    def run_evaluation(self):
        """
        Evaluates the return for num_episodes episodes.
        """
        self.logger.info("- Starting Evaluation.")
        self.mode = 'eval'
        self.record()
