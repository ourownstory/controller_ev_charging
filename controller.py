# not elegant, but makes easy to init subclasses from controller namespace
# from meta_controller import MetaController

from policy_gradient.pg import PG
from baselines.baseline import BaselineZero, BaselineOne
from baselines.random import Random
from deep_q.q_linear import LinearQN
from deep_q.q_nature import NatureQN

