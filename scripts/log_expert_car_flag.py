import gym
import numpy as np
import torch
import torchkit.pytorch_utils as ptu
# import environments
import envs.pomdp
# import recurrent model-free RL (separate architecture)
from policies.models.policy_mlp import ModelFreeOffPolicy_MLP as Policy_MLP
# import the replay buffer
from utils import helpers as utl

import pickle

cuda_id = 0 # -1 if using cpu
ptu.set_gpu_mode(torch.cuda.is_available() and cuda_id >= 0, cuda_id)

env_name = "Car-Flag-F-v0"
env = gym.make(env_name)
max_trajectory_len = env._max_episode_steps
act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]
state_dim = env.state_space.shape[0]
print(env, obs_dim, act_dim, max_trajectory_len)

agent = Policy_MLP(
            obs_dim=obs_dim,
            action_dim=act_dim,
            state_dim=state_dim,
            encoder="lstm",
            algo="td3",
            action_embedding_size=8,
            state_embedding_size=32,
            reward_embedding_size=8,
            rnn_hidden_size=128,
            dqn_layers=[128, 128],
            policy_layers=[128, 128],
            lr=0.0003,
            gamma=0.9,
            tau=0.005,
        ).to(ptu.device)

agent_dir = "logs/pomdp/CarFlag/agent.pt"
agent.load_state_dict(torch.load(agent_dir))

num_rollouts = 100

@torch.no_grad()
def collect_rollouts(
    num_rollouts,
    act_dim
):
    """collect num_rollouts of trajectories in task and save into policy buffer
    :param 
    """
    total_steps = 0
    total_rewards = 0.0

    expert_trajs = []

    for idx in range(num_rollouts):
        steps = 0
        rewards = 0.0
        obs = ptu.from_numpy(env.reset())
        obs = obs.reshape(1, obs.shape[-1])
        done_rollout = False

        initial_action = np.zeros(act_dim)
        oa_dict = {"obs": [], "acs": []}

        oa_dict["obs"].append(ptu.get_numpy(obs)[0])
        oa_dict["acs"].append(initial_action)

        state = ptu.from_numpy(env.get_state())
        state = state.reshape(1, state.shape[-1])

        while not done_rollout:
            # policy takes hidden state as input for rnn, while takes obs for mlp
            action, _, _, _ = agent.act(state, deterministic=True)
            # observe reward and next obs (B=1, dim)
            next_obs, reward, done, info = utl.env_step(
                env, action.squeeze(dim=0), rendering=True
            )
            done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

            oa_dict["obs"].append(ptu.get_numpy(next_obs)[0])
            oa_dict["acs"].append(ptu.get_numpy(action)[0])

            # update statistics
            steps += 1
            rewards += reward.item()

            # set: obs <- next_obs
            next_state = ptu.from_numpy(env.get_state())
            next_state = next_state.reshape(1, next_state.shape[-1])

            state = next_state.clone()

        expert_trajs.append(oa_dict)

        total_steps += steps
        total_rewards += rewards

    return expert_trajs


expert_trajs = collect_rollouts(num_rollouts, act_dim)

with open('CarFlag_expert_paths.pickle', 'wb') as f:
    pickle.dump(expert_trajs, f)
