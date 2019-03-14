import tensorflow as tf


class config_default:
    def __init__(self, use_baseline, config_env_name):
        self.record = False
        baseline_str = 'baseline' if use_baseline else 'no_baseline'

        # output config
        self.output_path = "results/{}-{}/".format(config_env_name, baseline_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 0
        self.summary_freq = 1

        # model and training config
        self.num_batches = 100  # number of batches trained on
        self.batch_size = 1000  # number of steps used to compute each policy update
        self.max_ep_len = 1000  # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 1.0  # the discount factor
        self.use_baseline = use_baseline
        self.normalize_advantage = True

        # parameters for the policy and baseline models
        self.n_layers = 1
        self.layer_size = 16
        self.activation = tf.nn.relu

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size


class config_test(config_default):
    def __init__(self, use_baseline, config_env_name):
        super().__init__(use_baseline, config_env_name)
        # extensions


def get_config(env_name, baseline, config_env_name):
    if env_name == 'default':
        return config_default(baseline, config_env_name)
    elif env_name == 'test':
        return config_test(baseline, config_env_name)
