from controller import Controller
import numpy as np
import tensorflow as tf


class Random(Controller):
    # def __init__(self, max_power, min_power, num_power_steps, num_stations):
    #     Controller.__init__(self, max_power, min_power, num_power_steps, num_stations)
    def __init__(self, env, config):
        """
        Initialize a Baseline Controller
        """
        super().__init__(env, config)
        self.add_action_op()

    def train(self):
        pass

    def add_action_op(self):
        # not implementing self.sampled_action because only used for training
        self.determ_action = tf.random_uniform(
            shape=(self.config.batch_size,),
            minval=0,
            maxval=self.action_dim,
            dtype=tf.int32,
        )
