import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym


num_envs = 16
num_timesteps = 256
num_total = num_envs*num_timesteps

update_epochs = 4
num_minibatches = 8
total_timesteps = 1000000

gamma = 0.99
gae_lambda = 0.95

clip_coef = 0.2
vf_coef = 0.5
ent_coef = 0.01
lr = 2.5e-4
seed = 42

device = "cpu"


def layer_init(layer,std=np.sqrt(2),constant=0):
	nn.init.orthogonal_(layer.weight,std)
	nn.init.constant_(layer.bias,constant)
	return layer


class Actor(nn.Module):
	def __init__(self,input_dim,hidden_dim,output_dim):
		super().__init__()
		self.net = nn.Sequential(
			layer_init(nn.Linear(input_dim,hidden_dim)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_dim,hidden_dim)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_dim,output_dim),std=0.01)
		)

	def forward(self,x):
		return self.net(x)


class Critic(nn.Module):
	def __init__(self, input_dim, hidden_dim):
		super().__init__()
		self.net = nn.Sequential(
			layer_init(nn.Linear(input_dim,hidden_dim)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_dim,hidden_dim)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_dim,1),std=1.0)
		)

	def forward(self,x):
		return self.net(x)



class Buffer():
	def __init__(self,num_timesteps, num_envs, obs_dim,device=device):
		self.num_timesteps, self.num_envs = num_timesteps, num_envs
		self.obs_dim = obs_dim
		self.device=device

		self.obs = torch.zeros((self.num_timesteps, self.num_envs, self.obs_dim),dtype=torch.float32).to(self.device)
		self.actions = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.long).to(self.device)
		self.log_probs = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.float32).to(self.device)
		self.advantages = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.float32).to(self.device)
		self.values = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.float32).to(self.device)
		self.returns = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.float32).to(self.device)
		self.dones = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.float32).to(self.device)
		self.rewards = torch.zeros((self.num_timesteps, self.num_envs),dtype=torch.float32).to(self.device)

		self.step = 0

	def reset(self):
		self.step = 0

	def append(self, obs, actions, log_probs, values, rewards, dones):
		t = self.step
		self.obs[t] = obs
		self.actions[t] = actions
		self.log_probs[t] = log_probs
		self.values[t] = values
		self.rewards[t] = torch.as_tensor(rewards,dtype=torch.float32,device=self.device)
		self.dones[t] = torch.as_tensor(dones,dtype=torch.float32,device=self.device)
		self.step+=1


	def compute_gae(self,next_value,gamma,gae_lambda):
		last_gae = 0
		for t in reversed(range(self.num_timesteps)):
			next_values = next_value if t==(self.num_timesteps-1) else self.values[t+1]
			nonterminal = 1.0-self.dones[t]
			delta = self.rewards[t] + gamma * nonterminal * next_values - self.values[t]
			last_gae = delta + gamma*gae_lambda*nonterminal*last_gae
			self.advantages[t] = last_gae
		self.returns = self.advantages + self.values

	def minibatches(self, num_minibatches):
		b_obs = self.obs.reshape(self.num_envs*self.num_timesteps, -1)
		b_actions = self.actions.reshape(-1)
		b_returns = self.returns.reshape(-1)
		b_log_probs = self.log_probs.reshape(-1)
		b_advantages = self.advantages.reshape(-1)

		batch_size = self.num_timesteps*self.num_envs
		minibatch_size = batch_size//num_minibatches
		mixed = torch.randperm(batch_size,device=self.device)
		for idx in range(0,batch_size,minibatch_size):
			minibatch = mixed[idx:idx+minibatch_size]
			yield b_obs[minibatch],b_actions[minibatch],b_log_probs[minibatch],b_advantages[minibatch],b_returns[minibatch]


def to_tensor(obs):
	return torch.tensor(obs,dtype=torch.float32,device=device)


def get_av(obs,actor,critic,action=None):
	logits = actor(obs)
	dist = Categorical(logits=logits)
	if action is None:
		action = dist.sample()
	log_prob = dist.log_prob(action)
	entropy = dist.entropy()
	value = critic(obs).squeeze(-1)
	return action, log_prob, entropy, value

if __name__ == "__main__":
	envs = gym.vector.SyncVectorEnv([lambda: gym.make("Acrobot-v1") for _ in range(num_envs)],autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
	obs_dim=envs.single_observation_space.shape[0]
	act_dim=envs.single_action_space.n

	actor = Actor(obs_dim, 128,act_dim)
	critic = Critic(obs_dim,128)
	params = list(actor.parameters())+list(critic.parameters())
	optimizer = optim.Adam(params,lr=lr)
	buffer = Buffer(num_timesteps, num_envs, obs_dim)

	global_step = 0
	num_iterations=total_timesteps//num_total

	next_obs, _ = envs.reset(seed=seed)
	for iteration in range(1,num_iterations+1):
		frac = 1.0 - (iteration-1)/num_iterations
		optimizer.param_groups[0]["lr"]=frac*lr

		buffer.reset()
		for t in range(num_timesteps):
			obs_t = to_tensor(next_obs)
			with torch.no_grad():
				action, log_prob,_,value = get_av(obs_t,actor,critic)
			next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
			if truncated.any(): print(infos.keys(), infos["final_obs"])
			timeout = truncated & ~terminated
			if timeout.any():
				idx = np.where(timeout)[0] 
				final_obs = np.stack([infos["final_obs"][i] for i in idx]) 
				with torch.no_grad():
					final_value = critic(to_tensor(final_obs)).squeeze(-1) 
				reward[idx] += gamma * final_value.cpu().numpy() 
			buffer.append(obs_t,action,log_prob,value,reward,terminated|truncated)
			global_step+=num_envs

		with torch.no_grad():
			next_value = critic(to_tensor(next_obs)).squeeze(-1)
		buffer.compute_gae(next_value,gamma,gae_lambda)

		for epoch in range(update_epochs):
			for mb_obs, mb_actions, mb_old_logp, mb_advantages, mb_returns in buffer.minibatches(num_minibatches):
				_, newlogprob,entropy,newvalue = get_av(mb_obs,actor,critic,mb_actions)
				logratio = newlogprob - mb_old_logp
				ratio = logratio.exp()

				mb_advantages = (mb_advantages - mb_advantages.mean())/(mb_advantages.std()+1e-8)
				pg_loss = torch.max(-mb_advantages*ratio, -mb_advantages*torch.clamp(ratio,1-clip_coef,1+clip_coef)).mean()
				v_loss = 0.5 * ((newvalue-mb_returns)**2).mean()
				ent=entropy.mean()
				loss=pg_loss - ent_coef*ent + vf_coef*v_loss
				optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(params,0.5)
				optimizer.step()

		print(iteration, global_step, -round(num_total/max(buffer.dones.sum().item(),1)))  

	envs.close()
	env = gym.make("Acrobot-v1", render_mode="human")
	for ep in range(5):
		obs, _ = env.reset()
		done, ret = False, 0.0
		while not done:
			with torch.no_grad():
				action = actor(to_tensor(obs).unsqueeze(0)).argmax(dim=-1).item()  
			obs, reward, terminated, truncated, _ = env.step(action)
			ret += reward
			done = terminated or truncated
		print("Episode", ep, ret)
	env.close()
