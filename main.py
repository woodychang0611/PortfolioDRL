import utils
import TD3
import OurDDPG
import DDPG
import numpy as np
from Model import Market,Market_Env
import torch

market =Market(data_src = r'.\1. Monthly Fund Data\Monthly_Fund_Return.csv')
start_time =np.datetime64("2008-01-01")

env =Market_Env('Feature.csv','Products.csv')

# Set seeds
rand_seed =0
env.seed(rand_seed)
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)

state_dim = env.state_dim
action_dim = env.action_dim
max_action = env.max_action