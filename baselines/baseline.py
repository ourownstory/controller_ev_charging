from controller import Controller
import numpy as np
import tensorflow as tf


class BaselineZero(Controller):
    # def __init__(self, MAX_POWER, MIN_POWER, NUM_POWER_STEPS, num_stations):
    #     Controller.__init__(self, MAX_POWER, MIN_POWER, NUM_POWER_STEPS, num_stations)
    def __init__(self, env, config):
        """
        Initialize a Baseline Controller
        """
        super().__init__(env, config)
        self.add_action_op()

    def train(self):
        # note: action is deterministic and not trained.
        print("skipping training for deterministic baseline 0")
        pass

    def add_action_op(self):
        assert self.discrete
        # note: action is deterministic and not trained.
        self.sampled_action = tf.constant(np.zeros(self.config.batch_size))
        # self.determ_action = tf.constant(np.zeros(self.config.batch_size))


class BaselineOne(Controller):
    # def __init__(self, MAX_POWER, MIN_POWER, NUM_POWER_STEPS, num_stations):
    #     Controller.__init__(self, MAX_POWER, MIN_POWER, NUM_POWER_STEPS, num_stations)
    def __init__(self, env, config):
        """
        Initialize a Baseline Controller
        """
        super().__init__(env, config)
        self.add_action_op()

    def train(self):
        # note: action is deterministic and not trained.
        print("skipping training for deterministic baseline 1")
        pass

    def add_action_op(self):
        assert self.discrete
        # note: action is deterministic and not trained.
        self.sampled_action = tf.constant((self.action_dim - 1) + np.zeros(self.config.batch_size))
        # self.determ_action = tf.constant((self.action_dim - 1) + np.zeros(self.config.batch_size))
