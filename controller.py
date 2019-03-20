#  makes easy to init subclasses from controller namespace
from policy_gradient.pg import PG
from baselines.baseline import BaselineZero, BaselineOne, BaselineFeasible
from baselines.random import Random
from deep_q.q_linear import LinearQN
from deep_q.q_deep import DeepQN
from deep_q.q_mlp import QLearningMLP
from deep_q.sarsa_mlp import SarsaMLP
from deep_q.sarsa_mlp_double import SarsaMLPDouble
from deep_q.q_mlp_double import QLearningMLPDouble
