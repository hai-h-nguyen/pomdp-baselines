import gym
import numpy as np
import torch
import torchkit.pytorch_utils as ptu
# import environments
import envs.pomdp
# import recurrent model-free RL (separate architecture)
from policies.models.policy_mlp import ModelFreeOffPolicy_MLP as Policy_MLP
# import the replay buffer
from buffers.seq_replay_buffer_vanilla import SeqReplayBuffer
from utils import helpers as utl

cuda_id = 0 # -1 if using cpu
ptu.set_gpu_mode(torch.cuda.is_available() and cuda_id >= 0, cuda_id)

env_name = "Two-Boxes-F-v0"
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

agent_dir = "logs/pomdp/Two-Boxes-F-v0/agent.pt"
agent.load_state_dict(torch.load(agent_dir))

num_updates_per_iter = 1.0 # training frequency
sampled_seq_len = 64 # context length
buffer_size = 1e6
batch_size = 32

num_iters = 150 
num_init_rollouts_pool = 5
num_rollouts_per_iter = 1
total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
n_env_steps_total = max_trajectory_len * total_rollouts
_n_env_steps_total = 0
# print("total env episodes", total_rollouts, "total env steps", n_env_steps_total)

@torch.no_grad()
def collect_rollouts(
    num_rollouts, 
    random_actions=False, 
    deterministic=False, 
    train_mode=True
):
    """collect num_rollouts of trajectories in task and save into policy buffer
    :param 
        random_actions: whether to use policy to sample actions, or randomly sample action space
        deterministic: deterministic action selection?
        train_mode: whether to train (stored to buffer) or test
    """
    if not train_mode:
        assert random_actions == False and deterministic == True

    total_steps = 0
    total_rewards = 0.0

    for idx in range(num_rollouts):
        steps = 0
        rewards = 0.0
        obs = ptu.from_numpy(env.reset())
        obs = obs.reshape(1, obs.shape[-1])
        done_rollout = False

        if train_mode:
            # temporary storage
            obs_list, act_list, rew_list, next_obs_list, term_list = (
                [],
                [],
                [],
                [],
                [],
            )

        while not done_rollout:
            if random_actions:
                action = ptu.FloatTensor(
                    [env.action_space.sample()]
                )  # (1, A)
            else:  
                # policy takes hidden state as input for rnn, while takes obs for mlp
                action, _, _, _ = agent.act(obs, deterministic=deterministic)
            # observe reward and next obs (B=1, dim)
            next_obs, reward, done, info = utl.env_step(
                env, action.squeeze(dim=0), rendering=True
            )
            done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

            # update statistics
            steps += 1
            rewards += reward.item()

            # early stopping env: such as rmdp, pomdp, generalize tasks. term ignores timeout
            term = (
                False
                if "TimeLimit.truncated" in info
                or steps >= max_trajectory_len
                else done_rollout
            )

            if train_mode:
                # append tensors to temporary storage
                obs_list.append(obs)  # (1, dim)
                act_list.append(action)  # (1, dim)
                rew_list.append(reward)  # (1, dim)
                term_list.append(term)  # bool
                next_obs_list.append(next_obs)  # (1, dim)

            # set: obs <- next_obs
            obs = next_obs.clone()

        if train_mode:
            # add collected sequence to buffer
            policy_storage.add_episode(
                observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),  # (L, dim)
                actions=ptu.get_numpy(torch.cat(act_list, dim=0)),  # (L, dim)
                rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),  # (L, dim)
                terminals=np.array(term_list).reshape(-1, 1),  # (L, 1)
                next_observations=ptu.get_numpy(
                    torch.cat(next_obs_list, dim=0)
                ),  # (L, dim)
            )
        print("Mode:", "Train" if train_mode else "Test", 
             "env_steps", steps, 
             "total rewards", rewards)
        total_steps += steps
        total_rewards += rewards

    if train_mode:
        return total_steps
    else:
        return total_rewards / num_rollouts

policy_storage = SeqReplayBuffer(
                max_replay_buffer_size=int(buffer_size),
                observation_dim=obs_dim,
                action_dim=act_dim,
                sampled_seq_len=sampled_seq_len,
                sample_weight_baseline=0.0,
            )

env_steps = collect_rollouts(num_rollouts=num_init_rollouts_pool, 
                             random_actions=False,
                             train_mode=False,
                             deterministic=True
                            )
_n_env_steps_total += env_steps

# evaluation parameters
last_eval_num_iters = 0
log_interval = 5
eval_num_rollouts = 10
learning_curve = {
    'x': [],
    'y': [],
}

while _n_env_steps_total < n_env_steps_total:

    env_steps = collect_rollouts(num_rollouts=num_rollouts_per_iter, 
                                train_mode=True
                                )
    _n_env_steps_total += env_steps

    current_num_iters = _n_env_steps_total // (
                            num_rollouts_per_iter * max_trajectory_len)
    if (current_num_iters != last_eval_num_iters
        and current_num_iters % log_interval == 0):
        last_eval_num_iters = current_num_iters
        average_returns = collect_rollouts(
                                num_rollouts=eval_num_rollouts, 
                                train_mode=False, 
                                random_actions=False, 
                                deterministic=True
                            )
        learning_curve['x'].append(_n_env_steps_total)
        learning_curve['y'].append(average_returns)
        # print(_n_env_steps_total, average_returns)