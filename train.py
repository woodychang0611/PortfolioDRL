from TD3 import utils
from TD3 import TD3
from TD3 import OurDDPG
from TD3 import DDPG
import numpy as np
import random
from Model import Market,Market_Env
import torch



def eval_policy(policy, eval_episodes=10):
    eval_env = Market_Env(feature_src,fund_map_src)
    avg_reward = 0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

fund_return_src = r'.\data\Monthly_Fund_Return.csv'
feature_src = r'.\data\Feature.csv'
fund_map_src =r'.\data\FUND_MAP_SELECTED.csv'

market =Market(data_src =fund_return_src)
start_time =np.datetime64("2008-01-01")

env =Market_Env(feature_src,fund_map_src)

# Set seeds
rand_seed =random.randint(0x00000000,0xFFFFFFFF)
env.seed(rand_seed)
torch.manual_seed(rand_seed)
np.random.seed(rand_seed)

state_dim = env.state_dim
action_dim = env.action_dim
max_action = env.max_action
kwargs = {
    "state_dim": state_dim,
    "action_dim": action_dim,
    "max_action": max_action,
    "discount":0.99,
    "tau": 0.005,
}

replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
policy_noise =0.2
noise_clip = 0.5
policy_freq =2

policy_name = "TD3"

if (policy_name=="TD3"):
    kwargs["policy_noise"] = policy_noise * max_action
    kwargs["noise_clip"] = noise_clip * max_action
    kwargs["policy_freq"] = policy_freq
    policy = TD3.TD3(**kwargs)
elif(policy=="DDPG"):
    policy = DDPG.DDPG(**kwargs)

eval_policy(policy)