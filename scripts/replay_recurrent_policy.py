import gym
import numpy as np
import torch
import torchkit.pytorch_utils as ptu
import envs.pomdp
from policies.models.policy_rnn import ModelFreeOffPolicy_Separate_RNN as Policy_RNN
from utils import helpers as utl
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--algo', type=str)
parser.add_argument('--env', type=str)
parser.add_argument('--agent-dir', type=str)

args = parser.parse_args()

cuda_id = -1
ptu.set_gpu_mode(torch.cuda.is_available() and cuda_id >= 0, cuda_id)

env = gym.make(args.env)
max_trajectory_len = env._max_episode_steps
act_dim = env.action_space.n
obs_dim = env.observation_space.shape[0]
state_dim = env.state_space.shape[0]
print(env, obs_dim, act_dim, max_trajectory_len)

agent = Policy_RNN(
            obs_dim=obs_dim,
            action_dim=act_dim,
            state_dim=state_dim,
            encoder="lstm",
            algo=args.algo,
            action_embedding_size=8,
            state_embedding_size=32,
            rnn_hidden_size=128,
            dqn_layers=[128, 128],
            policy_layers=[128, 128],
            lr=0.0003,
            gamma=0.9,
            tau=0.005,
            target_entropy=0.5  # This is not important during evaluation
        ).to(ptu.device)

agent_dir = args.agent_dir
agent.load_state_dict(torch.load(agent_dir))

num_iters = 150 
num_init_rollouts_pool = 5
num_rollouts_per_iter = 1
total_rollouts = num_init_rollouts_pool + num_iters * num_rollouts_per_iter
n_env_steps_total = max_trajectory_len * total_rollouts
_n_env_steps_total = 0

@torch.no_grad()
def collect_rollouts(
    num_rollouts,
    deterministic=False,
):
    """collect num_rollouts of trajectories in task and save into policy buffer
    :param 
        random_actions: whether to use policy to sample actions, or randomly sample action space
        deterministic: deterministic action selection?
        train_mode: whether to train (stored to buffer) or test
    """
    total_steps = 0
    total_rewards = 0.0

    for idx in range(num_rollouts):
        steps = 0
        rewards = 0.0
        obs = ptu.from_numpy(env.reset())
        obs = obs.reshape(1, obs.shape[-1])
        done_rollout = False

        # get hidden state at timestep=0, None for mlp
        action, internal_state = agent.get_initial_info()

        while not done_rollout:
            # policy takes hidden state as input for rnn, while takes obs for mlp
            (action, _, _, _), internal_state = agent.act(
                prev_internal_state=internal_state,
                prev_action=action,
                obs=obs,
                deterministic=deterministic,
            )
            # observe reward and next obs (B=1, dim)
            next_obs, reward, done, info = utl.env_step(
                env, action.squeeze(dim=0), rendering=True
            )
            done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

            # update statistics
            steps += 1
            rewards += reward.item()

            # set: obs <- next_obs
            obs = next_obs.clone()

        print("Mode:", "Test",
             "env_steps", steps,
             "total rewards", rewards)
        total_steps += steps
        total_rewards += rewards

    return total_rewards / num_rollouts


env_steps = collect_rollouts(num_rollouts=num_init_rollouts_pool, deterministic=True)
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

    env_steps = collect_rollouts(num_rollouts=num_rollouts_per_iter)
    _n_env_steps_total += env_steps

    current_num_iters = _n_env_steps_total // (
                            num_rollouts_per_iter * max_trajectory_len)
    if (current_num_iters != last_eval_num_iters
            and current_num_iters % log_interval == 0):
        last_eval_num_iters = current_num_iters
        average_returns = collect_rollouts(
                                num_rollouts=eval_num_rollouts, 
                                deterministic=True
                            )
        learning_curve['x'].append(_n_env_steps_total)
        learning_curve['y'].append(average_returns)
        # print(_n_env_steps_total, average_returns)