import tensorflow as tf


class config_default:
    def __init__(self, config_env_name):
        self.name = None
        self.controller_name = None
        self.use_baseline = True
        self.record = False
        baseline_str = 'baseline' if self.use_baseline else 'no_baseline'

        # output config
        self.output_path = "results/{}-{}/".format(config_env_name, baseline_str)
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.plot_output = self.output_path + "scores.png"
        self.record_path = self.output_path
        self.record_freq = 0
        self.summary_freq = 1
        self.show_plots = True
        self.plot_freq = 10

        # model and training config
        self.num_batches = 20  # number of batches trained on
        self.batch_size = 1000  # number of steps used to compute each policy update
        self.max_ep_len = 1000  # maximum episode length
        self.learning_rate = 3e-2
        self.gamma = 1.0  # the discount factor
        self.normalize_advantage = True

        # parameters for the policy and baseline models
        self.n_layers = 1
        self.layer_size = 16
        self.activation = tf.nn.relu

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size


class config_pg(config_default):
    def __init__(self, config_env_name):
        super().__init__(config_env_name)
        self.name = 'PG-default'
        self.controller_name = "PG"


class config_baseline0(config_default):
    def __init__(self, config_env_name):
        super().__init__(config_env_name)
        self.name = 'BaselineZero'
        self.controller_name = "BaselineZero"


class config_baseline1(config_default):
    def __init__(self, config_env_name):
        super().__init__(config_env_name)
        self.name = 'BaselineOne'
        self.controller_name = "BaselineOne"


class config_random(config_default):
    def __init__(self, config_env_name):
        super().__init__(config_env_name)
        self.name = 'Random-default'
        self.controller_name = "Random"


def get_config(env_name, config_env_name):
    if env_name == 'PG':
        return config_pg(config_env_name)
    if env_name == 'BaselineZero':
        return config_baseline0(config_env_name)
    if env_name == 'BaselineOne':
        return config_baseline1(config_env_name)
    if env_name == 'Random':
        return config_random(config_env_name)
