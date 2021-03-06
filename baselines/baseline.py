from meta_controller import MetaController
import numpy as np
# import tensorflow as tf


class BaselineZero(MetaController):
    def build(self):
        pass

    def train(self):
        # note: action is deterministic and not trained.
        msg = "skipping training for deterministic baseline 0"
        self.logger.info(msg)

    def add_action_op(self):
        raise NotImplementedError("use get_action instead")
        # assert self.discrete
        # # note: action is deterministic and not trained.
        # self.sampled_action = tf.constant(np.zeros(self.config.batch_size))
        # # self.determ_action = tf.constant(np.zeros(self.config.batch_size))

    def get_action(self, state):
        # return np.zeros(self.config.batch_size)
        return 0


class BaselineOne(MetaController):
    def build(self):
        pass

    def train(self):
        # note: action is deterministic and not trained.
        msg = "skipping training for deterministic baseline 1"
        self.logger.info(msg)

    def add_action_op(self):
        raise NotImplementedError("use get_action instead")
        # assert self.discrete
        # # note: action is deterministic and not trained.
        # self.sampled_action = tf.constant((self.action_dim - 1) + np.zeros(self.config.batch_size))
        # # self.determ_action = tf.constant((self.action_dim - 1) + np.zeros(self.config.batch_size))

    def get_action(self, state):
        assert self.discrete
        # return np.zeros(self.config.batch_size) + (self.action_dim - 1)
        return self.action_dim - 1

class BaselineFeasible(MetaController):
    def build(self):
        self.rev_action_map = {tuple(v): k for k, v in self.env.action_map.items()}

    def train(self):
        msg = "skipping training for deterministic baseline feasible"
        self.logger.info(msg)

    def get_action(self, state):
        a = np.zeros(self.env.num_stations)
        for i, station in enumerate(state['stations']):
            if station['is_car'] and station['per_char'] < 1.0:
                a[i] = self.env.config.MAX_POWER

        return self.rev_action_map[tuple(a)]
