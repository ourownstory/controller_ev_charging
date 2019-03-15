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
        pass

    def add_action_op(self):
        assert self.discrete
        # not implementing self.sampled_action because only used for training
        self.determ_action = tf.constant(np.zeros(self.config.batch_size))


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
        pass

    def add_action_op(self):
        assert self.discrete
        # not implementing self.sampled_action because only used for training
        self.determ_action = tf.constant((self.action_dim - 1) + np.zeros(self.config.batch_size))
