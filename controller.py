from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np
import gym
import os
from copy import deepcopy

from policy_gradient.utils_pg import get_logger, Progbar
import utils_controller as utils

class Controller(ABC):
    """
    Abstract Class for implementing a Controller
    """
    def __init__(self, env, config, logger=None):
        self.env = env
        self.config = config
        self.mode = None
        
        #used for plotting frequency
        self.total_episode_counter = 0

        # must be defined by subclasses:
        self.sampled_action = None
        # self.determ_action = None

        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyperparameters
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)

        # discrete vs continuous action space
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]

        # placeholder for sampling paths
        self.observation_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.observation_dim), name="obs")

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def add_action_op(self):
        # needs to add self.sampled_action (and maybe self.determ_action)
        pass

    # @abstractmethod
    # def build(self):
    #     pass

    def initialize(self):
        """
        Assumes the graph has been constructed (have called self.build())
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()
        # tensorboard stuff
        self.add_summary()
        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def init_averages(self):
        """
        Defines extra attributes for tensorboard.

        You don't have to change or use anything here.
        """
        self.avg_reward = 0.
        self.max_reward = 0.
        self.std_reward = 0.
        self.eval_reward = 0.

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.

        You don't have to change or use anything here.

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        """
        Add summary to tensorboard

        You don't have to change or use anything here.
        """

        fd = {
            self.avg_reward_placeholder: self.avg_reward,
            self.max_reward_placeholder: self.max_reward,
            self.std_reward_placeholder: self.std_reward,
            self.eval_reward_placeholder: self.eval_reward,
        }
        summary = self.sess.run(self.merged, feed_dict=fd)
        # tensorboard stuff
        self.file_writer.add_summary(summary, t)

    def add_summary(self):
        """
        Tensorboard stuff.
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg_Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max_Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std_Reward", self.std_reward_placeholder)
        tf.summary.scalar("Eval_Reward", self.eval_reward_placeholder)
        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, self.sess.graph)

    def sample_action_discrete(self, action_logits):
        return tf.squeeze(
            tf.multinomial(
                logits=action_logits,
                num_samples=1,
            ), axis=-1)

    def sample_action_continuous(self, action_means, std):
        return action_means + std * tf.random_normal(
            shape=(tf.shape(self.observation_placeholder)[0], self.action_dim),
            mean=0.0,
            stddev=1.0,
            dtype=tf.float32,
        )

    def sample_path(self, env, max_ep_len, num_episodes=None):
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

        You do not have to implement anything in this function, but you will need to
        understand what it returns, and it is worthwhile to look over the code
        just so you understand how we are taking actions in the environment
        and generating batches to train on.
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
                states.append(state)
                
                feed = {self.observation_placeholder: states[-1][None]}
                action = self.sess.run(self.sampled_action, feed_dict=feed)[0]
                state, reward, done, info = env.step(action)
               
                actions.append(action)
                rewards.append(reward)
                infos.append(info)

                episode_reward = episode_reward + reward
                if self.config.show_plots:
                    utils.update_plot_data(plot_data, info)
                t += 1
                if (done or step == max_ep_len - 1):
                    # if self.config.gamma < 1:
                    #     episode_sum_scaling = (1.0 - (self.config.gamma ** max_ep_len)) / (1.0 - self.config.gamma)
                    # else:
                    #     episode_sum_scaling = max_ep_len
                    episode_rewards.append(episode_reward)
                    # print("break 1 at step", step, "max_ep_len", max_ep_len, "t", t, "done", done)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    # print("break 2 at step, t", step, t)
                    break
            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "infos": np.array(infos)}
            paths.append(path)
            episode += 1
            self.total_episode_counter += 1
            if num_episodes and episode >= num_episodes:
                break
        return paths, episode_rewards

    def record(self):
        """
        Recreate an env and record a video for one episode
        """
        # TODO: save plot
        new_env_config = self.env.config
        new_env = gym.make(new_env_config.ENV_NAME)
        new_env.build(new_env_config)

        paths, total_rewards = self.sample_path(
            new_env,
            max_ep_len=self.config.max_ep_len,
        )
        infos = paths[0]['infos']
        plot_data = {"times": [], "actions": [[] for _ in range(new_env.num_stations)],
                     "per_chars": [[] for _ in range(new_env.num_stations)],
                     "des_chars": [[] for _ in range(new_env.num_stations)],
                     "is_cars": [[] for _ in range(new_env.num_stations)]}
        for info in infos: 
            utils.update_plot_data(plot_data, info)
        utils.plot_episode(plot_data, self.total_episode_counter, new_env, self.config.plot_output)
 
        # env = gym.make(self.config.ENV_NAME)
        # env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        # self.evaluate(env, 1)

    def run_training(self):
        """
        Apply procedures of training for a PG.
        """
        self.mode = 'train'
        # initialize
        self.initialize()
        # record one game at the beginning
        if self.config.record:
            self.record()
        # model
        self.train()
        # record one game at the end
        if self.config.record:
            self.record()

    def run_evaluation(self, env=None, num_episodes=1):
        """
        Evaluates the return for num_episodes episodes.
        Not used right now, all evaluation statistics are computed during training
        episodes.
        """
        self.logger.info("- Starting Evalutaion.")
        self.mode = 'eval'
        if env == None: env = self.env
        env.reset()
        env.evaluation_mode = True  # TODO: make this sample from test dataset, not train
        paths, rewards = self.sample_path(
            env,
            max_ep_len=self.config.max_ep_len_eval,
            num_episodes=num_episodes
        )
        # scale to be comparable to training rewards:
        rewards = np.array(rewards) * self.config.max_ep_len / self.config.max_ep_len_eval
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Evaluation reward: {:9.1f} +/- {:.2f}".format(avg_reward, sigma_reward)
        self.logger.info(msg)
        return avg_reward


# not elegant, but makes easy to init subclasses from controller namespace
from policy_gradient.pg import PG
from baselines.baseline import BaselineZero, BaselineOne
from baselines.random import Random
