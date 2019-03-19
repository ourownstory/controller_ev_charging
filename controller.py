# not elegant, but makes easy to init subclasses from controller namespace
# from meta_controller import MetaController

from policy_gradient.pg import PG
from baselines.baseline import BaselineZero, BaselineOne, BaselineFeasible
from baselines.random import Random
from deep_q.q_linear import LinearQN
from deep_q.q_nature import NatureQN
from deep_q.q_mlp import QLearningMLP
from deep_q.sarsa_mlp import SarsaMLP
