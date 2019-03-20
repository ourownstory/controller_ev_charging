import tensorflow as tf

from deep_q.q_linear import LinearQN

from policy_gradient.pg import build_mlp


class DeepQN(LinearQN):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        with tf.variable_scope(scope, reuse):
            out = build_mlp(
                mlp_input=state,
                output_size=num_actions,
                scope=scope,
                layers=self.config.layer_sizes,
            )
            # # h = tf.layers.dense(inputs=tf.layers.flatten(state), units=512, activation=tf.nn.relu)
            # h = tf.layers.dense(inputs=state, units=512, activation=tf.nn.relu)
            # h = tf.layers.dense(inputs=h, units=512, activation=tf.nn.relu)
            # h = tf.layers.dense(inputs=h, units=256, activation=tf.nn.relu)
            # out = tf.layers.dense(inputs=h, units=num_actions)
        return out
