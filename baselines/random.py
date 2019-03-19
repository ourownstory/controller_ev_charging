from meta_controller import MetaController
import numpy as np
# import tensorflow as tf


class Random(MetaController):
    def build(self):
        pass

    def train(self):
        # note: action is deterministic and not trained.
        msg = "skipping training for purely random baseline"
        self.logger.info(msg)

    def add_action_op(self):
        raise NotImplementedError("use get_action instead")
        # note: action is deterministic and not trained.
        # self.sampled_action = tf.random_uniform(
        #     shape=(self.config.batch_size,),
        #     minval=0,
        #     maxval=self.action_dim,
        #     dtype=tf.int32,
        # )

    def get_action(self, state):
        if self.discrete:
            # action = np.random.randint(
            #     low=0,
            #     high=self.action_dim,
            #     size=self.config.batch_size
            # )
            action = np.random.randint(
                low=0,
                high=self.action_dim,
                size=1
            )[0]
        else:
            raise NotImplementedError
        return action
