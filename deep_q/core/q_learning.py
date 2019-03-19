import os
import numpy as np
import gym
import sys
from collections import deque

from deep_q.utils.general import get_logger, Progbar, export_plot
from deep_q.utils.replay_buffer import ReplayBuffer
from deep_q.q_schedule import LinearExploration, LinearSchedule
from meta_controller import MetaController

##TODO just have this here until we have an abstract controller
import sys
sys.path.append('../..')
import utils_controller as utils


class QN(MetaController):
    """
    Abstract Class for implementing a Q Network
    """

    def get_best_action(self, state):
        """
        Returns best action according to the network

        Args:
            state: observation from gym
        Returns:
            tuple: action, q values
        """
        raise NotImplementedError

    def get_action(self, state):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        if np.random.random() < self.config.soft_epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)[0]

    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        super().init_averages()
        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0
        
    def update_averages(self, rewards, scores_eval, max_q_values=None, q_values=None):
        """
        Update the averages

        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        if max_q_values is None or q_values is None:
            raise NotImplementedError("Looking for baseclass function?")

        super().update_averages(rewards, scores_eval)

        self.max_q      = np.mean(max_q_values)
        self.avg_q      = np.mean(q_values)
        self.std_q      = np.sqrt(np.var(q_values) / len(q_values))

    def train(self):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """
        # exploration strategy
        exp_schedule = LinearExploration(self.env, self.config.eps_begin, self.config.eps_end, self.config.eps_nsteps)
        # learning rate schedule
        lr_schedule = LinearSchedule(self.config.lr_begin, self.config.lr_end, self.config.lr_nsteps)

        # initialize replay buffer and variables
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0  # time control of nb of steps
        scores_eval = []  # list of scores computed at iteration time
        scores_eval += [self._evaluate()]

        prog = Progbar(target=self.config.nsteps_train)

        # interact with environment
        while t < self.config.nsteps_train:
            self.total_train_steps += 1
            total_reward = 0
            state = self.env.reset()
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                # if self.config.render_train: self.env.render()
                # replay memory stuff
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                # chose action according to current Q and exploration
                best_action, q_values = self.get_best_action(q_input)
                action = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)

                # perform action in env
                new_state, reward, done, info = self.env.step(action)

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)

                # logging stuff
                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                        (t % self.config.learning_freq == 0)):
                    self.update_averages(
                        rewards=rewards,
                        max_q_values=max_q_values,
                        q_values=q_values,
                        scores_eval=scores_eval
                    )
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg_R", self.avg_reward),
                                                  ("Max_R", np.max(rewards)), ("eps", exp_schedule.epsilon),
                                                  ("Grads", grad_eval), ("Max_Q", self.max_q),
                                                  ("lr", lr_schedule.epsilon)])

                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(
                        t, self.config.learning_start))
                    sys.stdout.flush()

                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                scores_eval += [self._evaluate()]

            if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
                last_record = 0
                self.record()

        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self._evaluate()]
        export_plot(scores_eval, "Scores", self.config.plot_output)

    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if (t > self.config.learning_start and t % self.config.learning_freq == 0):
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()
            
        # occasionaly save the weights
        if (t % self.config.saving_freq == 0):
            self.save()

        return loss_eval, grad_eval

    def sample_gameplay(self, env=None, max_ep_len=None, num_episodes=None, verbose=True):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        if num_episodes is None:
            self.logger.info("Evaluating...")

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env

        # replay memory to play
        replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        episode_rewards = []
        paths = []
        # infos = []

        for i in range(num_episodes):
            episode_reward = 0
            state = env.reset()
            t = 0
            states, actions, rewards, infos = [], [], [], []
            while max_ep_len is None or t < max_ep_len:
                t += 1
                states.append(state)
                # store last state in buffer
                idx     = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                action = self.get_action(q_input)

                # perform action in env
                new_state, reward, done, info = env.step(action)

                actions.append(action)
                rewards.append(reward)
                infos.append(info)
                # count reward and store info
                episode_reward += reward

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)
                state = new_state

                if done:
                    break
            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "infos": np.array(infos)}
            paths.append(path)

            # updates to perform at the end of an episode
            episode_rewards.append(episode_reward)
        if verbose:
            avg_reward = np.mean(episode_rewards)
            sigma_reward = np.sqrt(np.var(episode_rewards) / len(episode_rewards))
            if num_episodes > 1:
                msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)
                self.logger.info(msg)
        return paths, episode_rewards

    def _evaluate(self, env=None, max_ep_len=None, num_episodes=None, verbose=True):
        """
        Evaluation with same procedure as the training

        Return: average episode_rewards
        """
        _, episode_rewards = self.sample_gameplay(
            env, max_ep_len, num_episodes, verbose
        )
        return np.mean(episode_rewards)
