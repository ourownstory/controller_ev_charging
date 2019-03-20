from abc import ABC, abstractmethod
import os, datetime


def factory(classname):
    cls = globals()[classname]
    return cls


def get_config(name, config_env_name):
    return factory(name)(name, config_env_name)


class Config(ABC):
    def __init__(self, name, config_env_name):
        self.name = name
        self.config_env_name = config_env_name

        # output config
        self.output_path = "results/{}-{}/".format(name, config_env_name)
        self.output_path = self.output_path + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "/"
        os.makedirs(self.output_path)

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
        self.eval_episodes = 1000  # how many episodes to sample from eval set.
        self.record_episodes = 100  # to compute stats when record is triggered
        self.plots_per_record = 10  # how many plots to save per recording

        self.build()

    @abstractmethod
    def build(self):
        raise NotImplementedError

class BaselineZero(Config):
    def build(self):
        self.controller_name = "BaselineZero"


class BaselineOne(Config):
    def build(self):
        self.controller_name = "BaselineOne"


class BaselineFeasible(Config):
    def build(self):
        self.controller_name = "BaselineFeasible"


class Random(Config):
    def build(self):
        self.controller_name = "Random"


class PG(Config):
    def build(self):
        self.controller_name = "PG"

        self.use_baseline = True
        self.normalize_advantage = True

        # model and training config
        self.num_batches = 100  # number of batches trained on
        self.batch_size = 4 * 24 * 100  # number of steps used to compute each policy update

        self.learning_rate = 0.03
        # self.gamma = 0.8  # the discount factor
        self.gamma = 0.95  # the discount factor
        # self.gamma = 1  # the discount factor

        # parameters for the policy and baseline models
        self.layer_sizes = (512, 512, 256)

        # overwrite from general config:
        self.record = True
        self.record_freq = self.num_batches // 10


class PG_small(PG):
    def build(self):
        super().build()
        self.layer_sizes = (128, 128, 128)


class PG_nano(PG):
    def build(self):
        super().build()
        self.layer_sizes = (64, 32, 16)


class PG_nano_long(PG):
    def build(self):
        super().build()
        self.layer_sizes = (64, 32, 16)

        # model and training config
        self.num_batches = 10000  # number of batches trained on
        self.batch_size = 4 * 24  # number of steps used to compute each policy update

        self.learning_rate = 0.001
        self.record_freq = self.num_batches // 10


class PG_linear(PG):
    def build(self):
        super().build()
        self.layer_sizes = []


class ConfigQN(Config):
    def build(self):
        # env config
        self.render_train     = False
        self.render_test      = False

        # model and training config
        self.num_episodes_test = 50
        self.grad_clip         = True
        self.clip_val          = 10
        self.log_freq          = 50
        self.soft_epsilon      = 0.05

        # hyper params
        self.nsteps_train       = 5e5
        self.batch_size         = 32
        self.buffer_size        = 200000
        self.target_update_freq = 10000
        # self.gamma              = 0.8 # the discount factor
        self.gamma              = 0.95 # the discount factor
        # self.gamma              = 1
        self.learning_freq      = 1
        self.lr_begin           = 0.003
        self.lr_end             = 0.0003
        self.lr_nsteps          = self.nsteps_train/2
        self.eps_begin          = 1
        self.eps_end            = 0.03
        self.eps_nsteps         = self.nsteps_train/4
        self.learning_start     = 50000

        # model and training config
        self.saving_freq       = self.nsteps_train // 2
        self.log_freq          = 50
        self.eval_freq         = self.nsteps_train // 10

        # overwrite from general config:
        self.record_freq = self.nsteps_train // 5


class LinearQN(ConfigQN):
    def build(self):
        super().build()
        self.controller_name = "LinearQN"


class DeepQN(ConfigQN):
    def build(self):
        super().build()
        self.controller_name = "DeepQN"
        self.layer_sizes = (512, 512, 256)


class DeepQN_small(DeepQN):
    def build(self):
        super().build()
        self.layer_sizes = (128, 128, 128)


class DeepQN_nano(DeepQN):
    def build(self):
        super().build()
        self.layer_sizes = (64, 32, 16)


class QLearningMLP(Config):
    def build(self):
        self.controller_name = "QLearningMLP"
        self.hidden_layer_sizes = (512, 512, 256)
        self.lr = 0.001
        self.gamma = 0.9
        self.epsilon = 1

        self.num_batches = 1000  # number of batches trained on
        self.batch_size = 4 * 24 * 6  # number of steps used to compute each policy update
        self.record_freq = self.num_batches // 10

class SarsaMLP(Config):
    def build(self):
        self.controller_name = "SarsaMLP"
        self.hidden_layer_sizes = (512, 512, 256)
        self.lr = 0.001
        self.gamma = 0.9
        self.epsilon = 1

        self.num_batches = 1000  # number of batches trained on
        self.batch_size = 4 * 24 * 6  # number of steps used to compute each policy update
        self.record_freq = self.num_batches // 10
