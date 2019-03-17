import numpy as np
import tensorflow as tf

from policy_gradient.utils_pg import get_logger, Progbar, export_plot, build_mlp
from controller import Controller


class PG(Controller):
  """
  Class for implementing a Policy Gradient Based Algorithm
  """
  def __init__(self, env, config):
    """
    Initialize Policy Gradient Class

    Args:
            env: an OpenAI Gym environment
            config: class with hyperparameters
            logger: logger instance from the logging module

    """
    super().__init__(env, config)
    self.lr = self.config.learning_rate
    self.build()

  def build(self):
    """
    Build the model by adding all necessary variables.

    we are just calling all the operations you already defined above to build the tensorflow graph.
    """

    # add placeholders
    self.add_placeholders_op()
    # create policy net
    self.add_action_op()
    # add square loss
    self.add_loss_op()
    # add optmizer for the main networks
    self.add_optimizer_op()

    # add baseline
    if self.config.use_baseline:
      self.add_baseline_op()


  def add_placeholders_op(self):
    """
    Add placeholders for observation, action, and advantage:
        self.observation_placeholder, type: tf.float32
        self.action_placeholder, type: depends on the self.discrete
        self.advantage_placeholder, type: tf.float32

    In the case of continuous action space, an action will be specified by
    'self.action_dim' float32 numbers (i.e. a vector with size 'self.action_dim')
    """
    # self.observation_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.observation_dim), name="obs")
    if self.discrete:
      self.action_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,), name="action")
    else:
      self.action_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.action_dim), name="action")

    # Define a placeholder for advantages
    self.advantage_placeholder = tf.placeholder(dtype=tf.float32, shape=(None,), name="adv")

  def add_action_op(self, scope = "policy_network"):
    """
    Build the policy network, construct the tensorflow operation to sample
    actions from the policy network outputs, and compute the log probabilities
    of the actions taken (for computing the loss later). These operations are
    stored in self.sampled_action and self.logprob. Must handle both settings
    of self.discrete.

    Args:
            scope: the scope of the neural network

    TODO:
    Discrete case:
        action_logits: the logits for each action
            HINT: use build_mlp, check self.config for layer_size and
            n_layers
        self.sampled_action: sample from these logits
            HINT: use tf.multinomial + tf.squeeze
        self.logprob: compute the log probabilities of the taken actions
            HINT: 1. tf.nn.sparse_softmax_cross_entropy_with_logits computes
                     the *negative* log probabilities of labels, given logits.
                  2. taken actions are different than sampled actions!

    Continuous case:
        To build a policy in a continuous action space domain, we will have the
        model output the means of each action dimension, and then sample from
        a multivariate normal distribution with these means and trainable standard
        deviation.

        That is, the action a_t ~ N( mu(o_t), sigma)
        where mu(o_t) is the network that outputs the means for each action
        dimension, and sigma is a trainable variable for the standard deviations.
        N here is a multivariate gaussian distribution with the given parameters.

        action_means: the predicted means for each action dimension.
            HINT: use build_mlp, check self.config for layer_size and
            n_layers
        log_std: a trainable variable for the log standard deviations.
            HINT: think about why we use log std as the trainable variable instead of std
            HINT: use tf.get_variables
        self.sampled_actions: sample from the gaussian distribution as described above
            HINT: use tf.random_normal
            HINT: use re-parametrization to obtain N(mu, sigma) from N(0, 1)
        self.lobprob: the log probabilities of the taken actions
            HINT: use tf.contrib.distributions.MultivariateNormalDiag

    """
    if self.discrete:
      action_logits = build_mlp(
          mlp_input=self.observation_placeholder,
          output_size=self.action_dim,
          scope=scope,
          n_layers=self.config.n_layers,
          size=self.config.layer_size,
          output_activation=None
      )
      self.sampled_action = self.sample_action_discrete(action_logits)
      # self.determ_action = tf.argmax(action_logits, axis=-1)

      self.logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.action_placeholder,
        logits=action_logits,
        name="XE"
      )
    else:
      action_means = build_mlp(
        mlp_input=self.observation_placeholder,
        output_size=self.action_dim,
        scope=scope,
        n_layers=self.config.n_layers,
        size=self.config.layer_size,
        output_activation=None
      )
      log_std = tf.get_variable(
        name="log_std",
        shape=(self.action_dim),
        dtype=tf.float32,
        trainable=True,
      )
      std = tf.exp(log_std)
      self.sampled_action = self.sample_action_continuous( action_means, std)
      # self.determ_action = action_means

      self.logprob = tf.contrib.distributions.MultivariateNormalDiag(
        loc=action_means,
        scale_diag=std,
      ).log_prob(
        value=self.action_placeholder,
      )

  def add_loss_op(self):
    """
    Compute the loss, averaged for a given batch.

    Recall the update for REINFORCE with advantage:
    θ = θ + α ∇_θ log π_θ(a_t|s_t) A_t
    Think about how to express this update as minimizing a
    loss (so that tensorflow will do the gradient computations
    for you).

    You only have to reference fields of 'self' that have already
    been set in the previous methods.

    """
    self.loss = -tf.reduce_mean(
      self.logprob * self.advantage_placeholder, axis=0)

  def add_optimizer_op(self):
    """
    Set 'self.train_op' using AdamOptimizer
    HINT: Use self.lr, and minimize self.loss
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    self.train_op = optimizer.minimize(loss=self.loss)

  def add_baseline_op(self, scope = "baseline"):
    """
    Build the baseline network within the scope.

    In this function we will build the baseline network.
    Use build_mlp with the same parameters as the policy network to
    get the baseline estimate. You also have to setup a target
    placeholder and an update operation so the baseline can be trained.

    Args:
        scope: the scope of the baseline network

    TODO: Set the following fields
        self.baseline
            HINT: use build_mlp, the network is the same as policy network
            check self.config for n_layers and layer_size
            HINT: tf.squeeze might be helpful
        self.baseline_target_placeholder
        self.update_baseline_op
            HINT: first construct a loss using tf.losses.mean_squared_error.
            HINT: use AdamOptimizer with self.lr

    """
    self.baseline = tf.squeeze(build_mlp(
      mlp_input=self.observation_placeholder,
      output_size=1,
      scope=scope,
      n_layers=self.config.n_layers,
      size=self.config.layer_size,
      output_activation=None
    ),
      axis=-1
    )
    self.baseline_target_placeholder = tf.placeholder(tf.float32, shape=(None), name="baseline_target")
    baseline_loss = tf.losses.mean_squared_error(labels=self.baseline_target_placeholder, predictions=self.baseline)
    baseline_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    self.update_baseline_op = baseline_optimizer.minimize(baseline_loss)

  def get_returns(self, paths):
    """
    Calculate the returns G_t for each timestep

    Args:
            paths: recorded sample paths.  See sample_path() for details.

    Return:
            returns: return G_t for each timestep

    After acting in the environment, we record the observations, actions, and
    rewards. To get the advantages that we need for the policy update, we have
    to convert the rewards into returns, G_t, which are themselves an estimate
    of Q^π (s_t, a_t):

       G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

    where T is the last timestep of the episode.

    TODO: compute and return G_t for each timestep. Use self.config.gamma.
    """

    all_returns = []
    for path in paths:
      rewards = path["reward"]
      previous_return = 0
      returns = []
      for r in rewards[::-1]:
        previous_return = r + self.config.gamma * previous_return
        returns.append(previous_return)
      returns = returns[::-1]
      all_returns.append(returns)
    returns = np.concatenate(all_returns)

    return returns

  def calculate_advantage(self, returns, observations):
    """
    Calculate the advantage

    Args:
            returns: all discounted future returns for each step
            observations: observations
    Returns:
            adv: Advantage

    Calculate the advantages, using baseline adjustment if necessary,
    and normalizing the advantages if necessary.
    If neither of these options are True, just return returns.

    TODO:
    If config.use_baseline = False and config.normalize_advantage = False,
    then the "advantage" is just going to be the returns (and not actually
    an advantage).

    if config.use_baseline, then we need to evaluate the baseline and subtract
      it from the returns to get the advantage.
      HINT: evaluate the self.baseline with self.sess.run(...)

    if config.normalize_advantage:
      after doing the above, normalize the advantages so that they have a mean of 0
      and standard deviation of 1.
    """
    adv = returns
    if self.config.use_baseline:
      baseline = self.sess.run(
        self.baseline,
        feed_dict={self.observation_placeholder: observations}
      )
      adv = adv - baseline
    if self.config.normalize_advantage:
      adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-5)
    return adv

  def update_baseline(self, returns, observations):
    """
    Update the baseline from given returns and observation.

    Args:
            returns: Returns from get_returns
            observations: observations
    TODO:
      apply the baseline update op with the observations and the returns.
      HINT: Run self.update_baseline_op with self.sess.run(...)
    """
    self.sess.run(
      self.update_baseline_op,
      feed_dict={self.observation_placeholder: observations,
                 self.baseline_target_placeholder: returns}
    )

  def train(self):
    """
    Performs training

    """
    last_eval = 0
    last_record = 0
    scores_eval = []  # list of scores computed at iteration time

    self.init_averages()

    for t in range(self.config.num_batches):

      # collect a minibatch of samples
      paths, total_rewards = self.sample_path(
        self.env,
        max_ep_len=self.config.max_ep_len,
        )
      scores_eval = scores_eval + total_rewards
      observations = np.concatenate([path["observation"] for path in paths])
      actions = np.concatenate([path["action"] for path in paths])
      rewards = np.concatenate([path["reward"] for path in paths])
      # compute Q-val estimates (discounted future returns) for each time step
      returns = self.get_returns(paths)
      advantages = self.calculate_advantage(returns, observations)

      # run training operations
      if self.config.use_baseline:
        self.update_baseline(returns, observations)
      self.sess.run(self.train_op, feed_dict={
                    self.observation_placeholder : observations,
                    self.action_placeholder : actions,
                    self.advantage_placeholder : advantages})

      # tf stuff
      if (t % self.config.summary_freq == 0):
        self.update_averages(total_rewards, scores_eval)
        self.record_summary(t)

      # compute reward statistics for this batch and log
      avg_reward = np.mean(total_rewards)
      sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
      msg = "Average reward: {:9.1f} +/- {:.2f}".format(avg_reward, sigma_reward)
      self.logger.info(msg)

      if  self.config.record and (last_record > self.config.record_freq):
        self.logger.info("Recording...")
        last_record =0
        self.record()

    self.logger.info("- Training done.")
    export_plot(scores_eval, "Score", self.config.name, self.config.plot_output)

