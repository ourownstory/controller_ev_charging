import tensorflow as tf
from abc import ABC, abstractmethod


def get_config(name, config_env_name):
    if name == 'PG':
        return config_pg(name, config_env_name)
    if name == 'BaselineZero':
        return config_baseline0(name, config_env_name)
    if name == 'BaselineOne':
        return config_baseline1(name, config_env_name)
    if name == 'Random':
        return config_random(name, config_env_name)
    if name == 'LinearQN':
        return config_linear_qn(name, config_env_name)
    if name == 'NatureQN':
        return config_nature_qn(name, config_env_name)


class Config(ABC):
    def __init__(self, name, config_env_name):
        self.name = name
        self.config_env_name = config_env_name

        # output config
        self.output_path = "results/{}-{}/".format(name, config_env_name)
        self.model_output = self.output_path + "model.weights/"
        self.log_path = self.output_path + "log.txt"
        self.plot_output = self.output_path

        # general experiment settings
        self.record = True
        # self.record_path = self.output_path
        # self.record_freq = None
        self.summary_freq = 1
        self.show_plots = True

        # for evaluation
        self.eval_episodes = 1

        self.build()

    @abstractmethod
    def build(self):
        raise NotImplementedError

class config_baseline0(Config):
    def build(self):
        self.controller_name = "BaselineZero"


class config_baseline1(Config):
    def build(self):
        self.controller_name = "BaselineOne"


class config_random(Config):
    def build(self):
        self.controller_name = "Random"


class config_pg(Config):
    def build(self):
        self.controller_name = "PG"

        self.use_baseline = True
        self.normalize_advantage = True

        # model and training config

        self.num_batches = 100  # number of batches trained on
        self.batch_size = 4 * 24 * 70  # number of steps used to compute each policy update

        self.learning_rate = 5e-2
        self.gamma = 0.95  # the discount factor

        # parameters for the policy and baseline models
        self.n_layers = 2
        self.layer_size = 128
        self.activation = tf.nn.relu

        ## These are copied from the env_variables
        # self.max_ep_len = -1  # maximum episode length
        # self.max_ep_len_eval = -1  # maximum episode length
        ## since we start new episodes for each batch
        # assert self.max_ep_len <= self.batch_size
        # if self.max_ep_len < 0:
        #     self.max_ep_len = self.batch_size

        # overwrite from general config:
        self.record_freq = self.num_batches // 10


class config_qn(Config):
    def build(self):
        # env config
        self.render_train     = False
        self.render_test      = False
        # self.overwrite_render = True
        # self.high             = 255.

        # model and training config
        self.num_episodes_test = 50
        self.grad_clip         = True
        self.clip_val          = 10
        self.saving_freq       = 5000
        self.log_freq          = 50
        self.eval_freq         = 5000
        self.soft_epsilon      = 0

        # hyper params
        self.nsteps_train       = 20000
        self.batch_size         = 128
        self.buffer_size        = 5000
        self.target_update_freq = 500
        self.gamma              = 0.95
        self.learning_freq      = 1
        self.state_history      = 1
        self.lr_begin           = 0.01
        self.lr_end             = 0.001
        self.lr_nsteps          = self.nsteps_train/2
        self.eps_begin          = 1
        self.eps_end            = 0.1
        self.eps_nsteps         = self.nsteps_train/4
        self.learning_start     = 1000

        # overwrite from general config:
        self.record_freq = self.nsteps_train // 10


class config_linear_qn(config_qn):
    def build(self):
        super().build()
        self.controller_name = "LinearQN"


class config_nature_qn(config_qn):
    def build(self):
        super().build()
        self.controller_name = "NatureQN"

