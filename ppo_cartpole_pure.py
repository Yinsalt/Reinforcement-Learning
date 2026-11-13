import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

N, T = 16, 128
epochs, num_minibatches = 4, 4
gamma, gae_lambda = 0.99, 0.95
clip_coef, vf_coef, ent_coef = 0.2, 0.5, 0.01
iterations = 150

envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(N)],
	autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
obs_dim = envs.single_observation_space.shape[0]
act_dim = envs.single_action_space.n

actor = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, act_dim))
critic = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
params = list(actor.parameters()) + list(critic.parameters())
optimizer = torch.optim.Adam(params, lr=2.5e-4)

obs = torch.zeros(T, N, obs_dim)
actions = torch.zeros(T, N, dtype=torch.long)
log_probs = torch.zeros(T, N)
values = torch.zeros(T, N)
rewards = torch.zeros(T, N)
dones = torch.zeros(T, N)

next_obs, _ = envs.reset(seed=42)
next_obs = torch.tensor(next_obs, dtype=torch.float32)

for iteration in range(iterations):

	for t in range(T):
		with torch.no_grad():
			dist = Categorical(logits=actor(next_obs))
			action = dist.sample()
			obs[t] = next_obs
			actions[t] = action
			log_probs[t] = dist.log_prob(action)
			values[t] = critic(next_obs).squeeze(-1)
		next_obs, reward, terminated, truncated, _ = envs.step(action.numpy())
		next_obs = torch.tensor(next_obs, dtype=torch.float32)
		rewards[t] = torch.tensor(reward, dtype=torch.float32)
		dones[t] = torch.tensor(terminated | truncated, dtype=torch.float32)

	with torch.no_grad():
		next_value = critic(next_obs).squeeze(-1)
	advantages = torch.zeros(T, N)
	lastgaelam = 0
	for t in reversed(range(T)):
		nextvalues = next_value if t == T - 1 else values[t + 1]
		nonterminal = 1.0 - dones[t]
		delta = rewards[t] + gamma * nextvalues * nonterminal - values[t]
		lastgaelam = delta + gamma * gae_lambda * nonterminal * lastgaelam
		advantages[t] = lastgaelam
	returns = advantages + values

	b_obs, b_actions, b_log_probs = obs.reshape(T * N, -1), actions.reshape(-1), log_probs.reshape(-1)
	b_adv, b_returns = advantages.reshape(-1), returns.reshape(-1)
	mb_size = T * N // num_minibatches
	for epoch in range(epochs):
		idx = torch.randperm(T * N)
		for start in range(0, T * N, mb_size):
			mb = idx[start:start + mb_size]
			dist = Categorical(logits=actor(b_obs[mb]))
			ratio = (dist.log_prob(b_actions[mb]) - b_log_probs[mb]).exp()
			adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
			pg_loss = -torch.min(adv * ratio, adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
			v_loss = 0.5 * ((critic(b_obs[mb]).squeeze(-1) - b_returns[mb]) ** 2).mean()
			loss = pg_loss + vf_coef * v_loss - ent_coef * dist.entropy().mean()
			optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(params, 0.5)
			optimizer.step()

	print(iteration, round(T * N / max(dones.sum().item(), 1)))
