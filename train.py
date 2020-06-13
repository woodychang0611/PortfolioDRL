from TD3 import utils
from TD3 import TD3
from TD3 import OurDDPG
from TD3 import DDPG
import numpy as np
import random
from Model import Market_Env
import torch
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="TD3")   
args = parser.parse_args()
fund_return_src = r'./data/Monthly_Fund_Return_Selected.csv'
feature_src = r'./data/FEATURE.csv'
fund_map_src =r'./data/FUND_MAP_SELECTED.csv'
env = Market_Env(feature_src,fund_map_src,fund_return_src)



def eval_policy(policy, eval_episodes=10):
    #eval_env = Market_Env(feature_src,fund_map_src,fund_return_src)
    avg_reward  = 0
    for _ in range(eval_episodes):
        state, done = env.reset(validation=True), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward

parser.add_argument("--env", default="HalfCheetah-v2") 


env =Market_Env(feature_src,fund_map_src,fund_return_src)

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
max_timesteps = 1e5
expl_noise =0.1
policy_name = args.policy
batch_size = 256
eval_freq=1500

if (policy_name=="TD3"):
    kwargs["policy_noise"] = policy_noise * max_action
    kwargs["noise_clip"] = noise_clip * max_action
    kwargs["policy_freq"] = policy_freq
    policy = TD3.TD3(**kwargs)
elif(policy_name=="DDPG"):
    policy = OurDDPG.DDPG(**kwargs)
   
evaluations = [eval_policy(policy)]

state, done = env.reset(), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0
episode_rewards=[]
for t in range(int(max_timesteps)):
    episode_timesteps += 1

    # Select action randomly or according to policy
    action = (
        policy.select_action(np.array(state))
        + np.random.normal(0, max_action * expl_noise, size=action_dim)
    ).clip(-max_action, max_action)

    # Perform action
    next_state, reward, done = env.step(action) 
    done_bool = float(done)

    # Store data in replay buffer
    replay_buffer.add(state, action, next_state, reward, done_bool)

    state = next_state
    episode_reward += reward

    # Train agent after collecting sufficient data
    policy.train(replay_buffer, batch_size)

    if done: 
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        episode_rewards.append(episode_reward)
        # Reset environment
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1 

    # Evaluate episode
    file_name = f'{policy_name}_{t+1}'
    file_name_latest = f'{policy_name}_latest'
    if (t + 1) % eval_freq == 0:
        evaluations.append(eval_policy(policy))
        env.reset()
        np.save(f"./results/{file_name}_evaluations", evaluations)
        np.save(f"./results/{file_name}_episode_rewards",episode_rewards)
        policy.save(f"./models/{file_name}")
        policy.save(f"./models/{file_name_latest}")