import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt


num_envs = 16
num_steps = 128
batch_size = num_envs * num_steps
num_minibatches = 4

update_epochs = 4
gamma = 0.99
gae_lambda = 0.95
clip_coef = 0.2
vf_coef = 0.5
ent_coef = 0.01
learning_rate = 2.5e-4
max_grad_norm = 0.5
total_timesteps = 2000000
seed = 42

DEVICE = torch.device("cpu")


def make_env(name, idx, seed=seed):
	def thunk():
		env = gym.make(name)
		env = gym.wrappers.RecordEpisodeStatistics(env)
		env.action_space.seed(seed + idx)
		return env
	return thunk


def make_envs(name, num_envs=num_envs, seed=seed):
	return gym.vector.SyncVectorEnv([make_env(name, i, seed) for i in range(num_envs)],
									autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def layer_init(layer, std=np.sqrt(2), bias=0.0):
	nn.init.orthogonal_(layer.weight, std)
	nn.init.constant_(layer.bias, bias)
	return layer


class Actor(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super().__init__()
		self.net = nn.Sequential(
			layer_init(nn.Linear(input_dim, hidden_dim)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_dim, hidden_dim)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_dim, output_dim), std=0.01)
		)

	def forward(self, x):
		return self.net(x)


class Critic(nn.Module):
	def __init__(self, input_dim, hidden_dim):
		super().__init__()
		self.net = nn.Sequential(
			layer_init(nn.Linear(input_dim, hidden_dim)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_dim, hidden_dim)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_dim, 1), std=1.0)
		)

	def forward(self, x):
		return self.net(x)


class RolloutBuffer:
	def __init__(self, num_steps, num_envs, obs_dim, device):
		self.T, self.N, self.device = num_steps, num_envs, device
		self.obs = torch.zeros((num_steps, num_envs, obs_dim), device=device)
		self.actions = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
		self.log_probs = torch.zeros((num_steps, num_envs), device=device)
		self.values = torch.zeros((num_steps, num_envs), device=device)
		self.rewards = torch.zeros((num_steps, num_envs), device=device)
		self.dones = torch.zeros((num_steps, num_envs), device=device)
		self.step = 0

	def add(self, obs, action, log_prob, value, reward, done):
		t = self.step
		self.obs[t] = obs
		self.actions[t] = action
		self.log_probs[t] = log_prob
		self.values[t] = value
		self.rewards[t] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
		self.dones[t] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
		self.step += 1

	def compute_gae(self, next_value, gamma, gae_lambda):
		self.advantages = torch.zeros_like(self.rewards)
		lastgaelam = 0
		for t in reversed(range(self.T)):
			nextvalues = next_value if t == self.T - 1 else self.values[t + 1]
			nonterminal = 1.0 - self.dones[t]
			delta = self.rewards[t] + gamma * nextvalues * nonterminal - self.values[t]
			lastgaelam = delta + gamma * gae_lambda * nonterminal * lastgaelam
			self.advantages[t] = lastgaelam
		self.returns = self.advantages + self.values

	def minibatches(self, num_minibatches):
		b_obs = self.obs.reshape(self.T * self.N, -1)
		b_actions = self.actions.reshape(-1)
		b_log_probs = self.log_probs.reshape(-1)
		b_adv = self.advantages.reshape(-1)
		b_returns = self.returns.reshape(-1)
		batch_size = self.T * self.N
		mb_size = batch_size // num_minibatches
		idx = torch.randperm(batch_size, device=self.device)
		for start in range(0, batch_size, mb_size):
			mb = idx[start:start + mb_size]
			yield b_obs[mb], b_actions[mb], b_log_probs[mb], b_adv[mb], b_returns[mb]

	def reset(self):
		self.step = 0


class Agent:
	def __init__(self, env_name, hidden_dim=64):
		self.env_name = env_name
		self.envs = make_envs(env_name)
		self.obs_dim = self.envs.single_observation_space.shape[0]
		self.act_dim = self.envs.single_action_space.n
		self.device = DEVICE
		self.actor = Actor(self.obs_dim, hidden_dim, self.act_dim).to(self.device)
		self.critic = Critic(self.obs_dim, hidden_dim).to(self.device)
		self.params = list(self.actor.parameters()) + list(self.critic.parameters())
		self.optimizer = optim.Adam(self.params, lr=learning_rate, eps=1e-5)
		self.memory = RolloutBuffer(num_steps, num_envs, self.obs_dim, self.device)
		self.global_step = 0
		self.episodes = []
		self.history = {"step": [], "approx_kl": [], "clipfrac": [], "entropy": [], "explained_var": []}

	def to_tensor(self, obs):
		return torch.tensor(obs, dtype=torch.float32, device=self.device)

	def get_action_and_value(self, obs, action=None):
		logits = self.actor(obs)
		dist = Categorical(logits=logits)
		if action is None:
			action = dist.sample()
		log_prob = dist.log_prob(action)
		entropy = dist.entropy()
		value = self.critic(obs).squeeze(-1)
		return action, log_prob, entropy, value

	def collect(self, next_obs):
		self.memory.reset()
		for t in range(num_steps):
			obs_t = self.to_tensor(next_obs)
			with torch.no_grad():
				action, log_prob, _, value = self.get_action_and_value(obs_t)
			next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
			self.memory.add(obs_t, action, log_prob, value, reward, terminated | truncated)
			self.global_step += num_envs
			final = infos.get("final_info", {})
			if "episode" in final:
				for i in np.where(final["_episode"])[0]:
					self.episodes.append((self.global_step, float(final["episode"]["r"][i])))
		with torch.no_grad():
			next_value = self.critic(self.to_tensor(next_obs)).squeeze(-1)
		self.memory.compute_gae(next_value, gamma, gae_lambda)
		return next_obs

	def learn(self):
		kls, clipfracs, entropies = [], [], []
		ratio_check = None
		for epoch in range(update_epochs):
			for mb_obs, mb_actions, mb_old_logp, mb_adv, mb_returns in self.memory.minibatches(num_minibatches):
				_, newlogprob, entropy, newvalue = self.get_action_and_value(mb_obs, mb_actions)
				logratio = newlogprob - mb_old_logp
				ratio = logratio.exp()

				with torch.no_grad():
					if ratio_check is None:
						ratio_check = (ratio - 1).abs().max().item()
					kls.append(((ratio - 1) - logratio).mean().item())
					clipfracs.append(((ratio - 1).abs() > clip_coef).float().mean().item())

				mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
				pg_loss = torch.max(-mb_adv * ratio,
									-mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)).mean()
				v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
				ent = entropy.mean()
				loss = pg_loss - ent_coef * ent + vf_coef * v_loss

				self.optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.params, max_grad_norm)
				self.optimizer.step()
				entropies.append(ent.item())

		y = self.memory.returns.flatten()
		var_y = y.var()
		explained_var = (1 - (y - self.memory.values.flatten()).var() / var_y).item() if var_y > 0 else float("nan")
		return {"approx_kl": np.mean(kls), "clipfrac": np.mean(clipfracs), "entropy": np.mean(entropies),
				"explained_var": explained_var, "ratio_check": ratio_check}

	def train(self, total_timesteps=total_timesteps):
		num_iterations = total_timesteps // batch_size
		next_obs, _ = self.envs.reset(seed=seed)
		for iteration in range(1, num_iterations + 1):
			frac = 1.0 - (iteration - 1) / num_iterations
			self.optimizer.param_groups[0]["lr"] = frac * learning_rate
			next_obs = self.collect(next_obs)
			stats = self.learn()

			self.history["step"].append(self.global_step)
			for k in ["approx_kl", "clipfrac", "entropy", "explained_var"]:
				self.history[k].append(stats[k])

			if iteration % 10 == 0:
				recent = [r for _, r in self.episodes[-20:]]
				avg = np.mean(recent) if recent else float("nan")
				print(f"it {iteration:>4}/{num_iterations}  step {self.global_step:>7}  return(20) {avg:7.1f}  "
					f"kl {stats['approx_kl']:.4f}  clip {stats['clipfrac']:.3f}  ent {stats['entropy']:.3f}  "
					f"ev {stats['explained_var']:.2f}  ratio0 {stats['ratio_check']:.1e}")
		self.envs.close()

	def evaluate(self, n_episodes=10, render=False):
		env = gym.make(self.env_name, render_mode="human" if render else None)
		returns = []
		for ep in range(n_episodes):
			obs, _ = env.reset(seed=1000 + ep)
			done, ret = False, 0.0
			while not done:
				with torch.no_grad():
					action = self.actor(self.to_tensor(obs).unsqueeze(0)).argmax(dim=-1).item()
				obs, reward, terminated, truncated, _ = env.step(action)
				ret += reward
				done = terminated or truncated
			returns.append(ret)
		env.close()
		return np.mean(returns), np.std(returns)

	def save(self, path):
		torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

	def load(self, path):
		ckpt = torch.load(path, map_location=self.device)
		self.actor.load_state_dict(ckpt["actor"])
		self.critic.load_state_dict(ckpt["critic"])


def plot(agent, filename):
	fig, axes = plt.subplots(1, 4, figsize=(18, 4))
	if agent.episodes:
		steps, rets = zip(*agent.episodes)
		smooth = [np.mean(rets[max(0, i - 20):i + 1]) for i in range(len(rets))]
		axes[0].plot(steps, rets, alpha=0.2)
		axes[0].plot(steps, smooth)
	axes[0].set_title("Episoden-Return")
	for ax, key in zip(axes[1:], ["approx_kl", "clipfrac", "entropy"]):
		ax.plot(agent.history["step"], agent.history[key])
		ax.set_title(key)
	for ax in axes:
		ax.set_xlabel("Umgebungsschritte")
	fig.tight_layout()
	fig.savefig(filename)
	plt.show()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="CartPole-v1")
	parser.add_argument("--timesteps", type=int, default=total_timesteps)
	parser.add_argument("--render", type=int, default=3)
	args = parser.parse_args()

	torch.manual_seed(seed)
	np.random.seed(seed)

	agent = Agent(args.env)
	agent.train(args.timesteps)
	agent.save(f"ppo_{args.env}.pt")
	mean, std = agent.evaluate(n_episodes=10)
	print(f"Auswertung über 10 Episoden: {mean:.1f} ± {std:.1f}")
	plot(agent, f"ppo_{args.env}.png")
	if args.render > 0:
		agent.evaluate(n_episodes=args.render, render=True)
