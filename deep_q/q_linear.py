import tensorflow as tf
from deep_q.core.deep_q_learning import DQN



class LinearQN(DQN):
    """
    Implement Fully Connected with Tensorflow
    """
    def add_placeholders_op(self):
        """
        Adds placeholders to the graph

        These placeholders are used as inputs to the rest of the model and will be fed
        data during training.
        """
        # state_shape = list(self.env.observation_space.shape)
        self.s = tf.placeholder("float32", shape=(None, self.observation_dim))
        self.a = tf.placeholder("int32", shape=(None))
        self.r = tf.placeholder("float32", shape=(None))
        self.sp = tf.placeholder("float32", shape=(None, self.observation_dim))
        self.done_mask = tf.placeholder("bool", shape=(None))
        self.lr = tf.placeholder("float32", shape=())

    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels x config.state_history)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        num_actions = self.action_dim

        with tf.variable_scope(scope, reuse):
            # flat = tf.layers.flatten(state)
            out = tf.layers.dense(
                inputs=tf.layers.flatten(state),
                units=num_actions
            )
        return out

    def add_update_target_op(self, q_scope, target_q_scope):
        """
        update_target_op will be called periodically 
        to copy Q network weights to target Q network

        In DQN, we maintain two identical Q networks with
        2 different sets of weights. In tensorflow, we distinguish them
        with two different scopes.

        Periodically, we need to update all the weights of the Q network 
        and assign them with the values from the regular network. 
        Args:
            q_scope: (string) name of the scope of variables for q
            target_q_scope: (string) name of the scope of variables
                        for the target network
        """
        q_collection = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=q_scope)
        target_q_collection = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=target_q_scope)
        assign_list = [tf.assign(ref=target_var, value=var) for var, target_var in zip(q_collection, target_q_collection)]
        self.update_target_op = tf.group(*assign_list)

    def add_loss_op(self, q, target_q):
        """
        Sets the loss of a batch, self.loss is a scalar

        Args:
            q: (tf tensor) shape = (batch_size, num_actions)
            target_q: (tf tensor) shape = (batch_size, num_actions)
        """
        num_actions = self.env.action_space.n

        q_t_max = tf.reduce_max(target_q, axis=-1)
        not_done = (1 - tf.cast(self.done_mask, dtype="float32"))
        Q_samp = self.r + not_done * q_t_max * self.config.gamma
        Q = tf.reduce_sum(tf.one_hot(indices=self.a, depth=num_actions)*q, axis =-1)

        loss = tf.squared_difference(Q_samp, Q)
        self.loss = tf.reduce_mean(loss)

    def add_optimizer_op(self, scope):
        """
        Set self.train_op and self.grad_norm
        Args:
            scope: (string) scope name, that specifies if target network or not
        """
        optimizer = tf.train.AdamOptimizer(self.lr)
        var_list = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss, var_list))
        if self.config.grad_clip:
            gradients = [tf.clip_by_norm(g, self.config.clip_val) for g in gradients]
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))
        self.grad_norm = tf.global_norm(t_list=gradients)
