import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gymnasium as gym


num_envs = 16
num_steps = 1024
batch_size = num_envs*num_steps
num_minibatches = 64

update_epochs = 4
gamma = 0.999
gae_lambda = 0.98
clip_coef = 0.2
vf_coef = 0.5
ent_coef = 0.01
learning_rate = 3e-4
max_grad_norm = 0.5
total_timesteps = 3000000
hidden_dim=128
seed = 42

DEVICE = torch.device("cpu")

###########################################################################

class Buffer():
	def __init__(self,n_steps,n_envs,obs_size,act_size,device):
		self.n_steps = n_steps
		self.n_envs = n_envs
		self.obs_size = obs_size
		self.act_size = act_size
		self.device = device

		self.step = 0 

		self.running_mean = torch.zeros(obs_size, device=device) 
		self.running_var = torch.ones(obs_size, device=device) 
		self.count = 1e-4 
		self.training = True 

		self.states = torch.zeros((n_steps,n_envs,obs_size),dtype=torch.float32).to(device) 
		self.actions = torch.zeros((n_steps,n_envs,act_size),dtype=torch.float32).to(device) 
		self.rewards = torch.zeros((n_steps,n_envs),dtype=torch.float32).to(device)
		self.values = torch.zeros((n_steps,n_envs),dtype=torch.float32).to(device)
		self.dones = torch.zeros((n_steps,n_envs),dtype=torch.float32).to(device)

		self.advantages = torch.zeros((n_steps,n_envs),dtype=torch.float32).to(device)
		self.returns = torch.zeros((n_steps,n_envs),dtype=torch.float32).to(device)
		self.log_probs = torch.zeros((n_steps,n_envs),dtype=torch.float32).to(device)

	def set_train(self):
		self.training=True

	def set_eval(self):
		self.training=False

	def reset(self):
		self.step = 0

	def normalize(self, states): 
		if self.training: 
			batch_mean = states.mean(dim=0) 
			batch_var = states.var(dim=0, unbiased=False) 
			batch_count = states.shape[0]  
			delta = batch_mean - self.running_mean 
			total = self.count + batch_count
			m2 = self.running_var*self.count + batch_var*batch_count + delta**2*self.count*batch_count/total 
			self.running_mean = self.running_mean + delta*batch_count/total 
			self.running_var = m2/total
			self.count = total
		return torch.clamp((states - self.running_mean)/torch.sqrt(self.running_var + 1e-8), -10, 10)

	def add(self, states, actions, rewards, dones, values, log_probs):
		t = self.step
		self.states[t] = states
		self.actions[t] = actions 
		self.rewards[t] = torch.as_tensor(rewards,dtype=torch.float32,device=self.device)
		self.dones[t] = torch.as_tensor(dones,dtype=torch.float32,device=self.device)
		self.log_probs[t] = log_probs
		self.values[t] = values

		self.step+=1

	def calculate_advantage(self,next_value,gamma=0.999, gae_lambda=0.98):
		last_gaelam = 0 
		for t in reversed(range(self.n_steps)):
			next_values = next_value if t==(self.n_steps-1) else self.values[t+1] 
			nonterminal = 1.0-self.dones[t] 
			delta = self.rewards[t] + gamma*nonterminal*next_values - self.values[t]
			last_gaelam = delta + gamma*gae_lambda*nonterminal*last_gaelam
			self.advantages[t] = last_gaelam
		self.returns = self.advantages + self.values 

	def minibatches(self, batch_size): 
		total = self.n_steps * self.n_envs
		mixed_idx = torch.randperm(total, device=self.device) 

		states = self.states.reshape(total, -1) 
		actions = self.actions.reshape(total, -1)
		log_probs = self.log_probs.reshape(-1)
		advantages = self.advantages.reshape(-1)
		returns = self.returns.reshape(-1)

		for start in range(0, total, batch_size):
			mb = mixed_idx[start:start+batch_size] 
			yield states[mb], actions[mb], log_probs[mb], advantages[mb], returns[mb]

###########################################################################

def layer_init(layer,std=np.sqrt(2),bias=0.0):
	nn.init.orthogonal_(layer.weight,std)
	nn.init.constant_(layer.bias,bias)
	return layer


class Actor(nn.Module):
	def __init__(self,input_size,hidden_size,output_size):
		super().__init__()
		self.net = nn.Sequential(
			layer_init(nn.Linear(input_size,hidden_size)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_size,hidden_size)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_size,output_size),std=0.1) 
		)
		self.log_std = nn.Parameter(torch.zeros(output_size)) 

	def forward(self,x):
		return self.net(x), self.log_std.exp()


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

###########################################################################

def make_env(name,idx,env_kwargs,seed=seed):
	def thunk():
		env = gym.make(name,**env_kwargs)
		env = gym.wrappers.RecordEpisodeStatistics(env) 
		env.action_space.seed(seed+idx)
		return env
	return thunk

def make_envs(name,env_kwargs,num_envs=num_envs,seed=seed):
	return gym.vector.SyncVectorEnv([make_env(name,i,env_kwargs,seed) for i in range(num_envs)],
				autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)  

###########################################################################

class Agent():
	def __init__(self,env_name, hidden_dim=hidden_dim,**env_kwargs):
		self.env_name = env_name
		self.env_kwargs = env_kwargs
		self.envs = make_envs(env_name,env_kwargs)
		self.obs_dim = self.envs.single_observation_space.shape[0]
		self.act_dim = self.envs.single_action_space.shape[0]
		self.act_low = self.envs.single_action_space.low 
		self.act_high = self.envs.single_action_space.high
		self.device = DEVICE
		self.actor = Actor(self.obs_dim,hidden_dim,self.act_dim).to(self.device)
		self.critic = Critic(self.obs_dim,hidden_dim).to(self.device)
		self.params = list(self.actor.parameters())+list(self.critic.parameters())
		self.optimizer = optim.Adam(self.params, lr=learning_rate, eps=1e-5)
		self.buffer = Buffer(num_steps, num_envs, self.obs_dim,self.act_dim,self.device)
		self.global_step = 0
		self.episodes = [] 

	def to_tensor(self,obs):
		return torch.tensor(obs,dtype=torch.float32,device=self.device)

	def get_action_and_value(self,obs,action=None): 
		mean, std = self.actor(obs) 
		dist = Normal(mean, std) 
		if action is None:
			action = dist.sample()
		log_prob = dist.log_prob(action).sum(-1)
		entropy = dist.entropy().sum(-1)
		value = self.critic(obs).squeeze(-1) 
		return action, log_prob, entropy, value 

	def collect(self,next_obs):
		self.buffer.reset()
		self.buffer.set_train()
		for t in range(num_steps):
			obs_t = self.buffer.normalize(self.to_tensor(next_obs))  
			with torch.no_grad():
				action, log_prob, _, value = self.get_action_and_value(obs_t) 
			env_action = np.clip(action.cpu().numpy(), self.act_low, self.act_high) 
			next_obs, reward, terminated, truncated, infos = self.envs.step(env_action) 
			timeout = truncated & ~terminated 
			if timeout.any():
				idx = np.where(timeout)[0]
				final_obs = np.stack([infos["final_obs"][i] for i in idx])
				self.buffer.set_eval() 
				with torch.no_grad():
					final_value = self.critic(self.buffer.normalize(self.to_tensor(final_obs))).squeeze(-1)
				self.buffer.set_train()
				reward[idx] += gamma * final_value.cpu().numpy()  
			self.buffer.add(states=obs_t,actions=action,rewards=reward,dones=terminated|truncated,values=value,log_probs=log_prob)
			self.global_step+=num_envs
			final = infos.get("final_info",{})
			if "episode" in final:
				for i in np.where(final["_episode"])[0]:
					self.episodes.append((self.global_step,float(final["episode"]["r"][i]))) 
		self.buffer.set_eval() 
		with torch.no_grad():
			next_value = self.critic(self.buffer.normalize(self.to_tensor(next_obs))).squeeze(-1)
		self.buffer.set_train()
		self.buffer.calculate_advantage(next_value,gamma,gae_lambda)
		return next_obs

	def learn(self):
		kls, clipfracs = [],[]
		ratio_check = None
		for epoch in range(update_epochs): 
			for mb_states, mb_actions, mb_old_logp, mb_adv, mb_returns in self.buffer.minibatches(batch_size//num_minibatches):
				_, newlogprob, entropy, newvalue = self.get_action_and_value(mb_states, mb_actions)
				logratio = newlogprob - mb_old_logp
				ratio = logratio.exp()

				with torch.no_grad():
					if ratio_check is None:
						ratio_check = (ratio-1).abs().max().item() 
					kls.append(((ratio-1)-logratio).mean().item())  
					clipfracs.append(((ratio-1).abs()>clip_coef).float().mean().item()) 
				mb_adv = (mb_adv-mb_adv.mean())/(mb_adv.std()+1e-8)
				pg_loss = torch.max(-mb_adv*ratio, -mb_adv*torch.clamp(ratio,1-clip_coef,1+clip_coef)).mean() 
				v_loss = 0.5 * ((newvalue-mb_returns)**2).mean()
				loss = pg_loss - ent_coef*entropy.mean() + vf_coef*v_loss
				self.optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.params,max_grad_norm)
				self.optimizer.step()
		return {"approx_kl": np.mean(kls), "clipfrac": np.mean(clipfracs), "ratio_check": ratio_check}

	def train(self,total_timesteps=total_timesteps):
		num_iterations = total_timesteps//batch_size
		next_obs, _ = self.envs.reset(seed=seed)
		for iteration in range(1,num_iterations +1):
			frac = 1.0 - ( iteration -1)/num_iterations
			self.optimizer.param_groups[0]["lr"] = frac * learning_rate
			next_obs = self.collect(next_obs)
			stats = self.learn()
			recent = [r for _,r in self.episodes[-20:]]
			avg = np.mean(recent) if recent else float("nan")
			std = self.actor.log_std.exp().detach().cpu().numpy().round(2)
			print(f"it {iteration:>3}/{num_iterations}  step {self.global_step:>7}  return(20) {avg:7.1f}  std {std}  kl {stats['approx_kl']:.4f}  clip {stats['clipfrac']:.3f}  ratio0 {stats['ratio_check']:.1e}")
		self.envs.close()

	def evaluate(self,n_episodes=5,render=True):
		self.buffer.set_eval()  
		env = gym.make(self.env_name, render_mode="human" if render else None, **self.env_kwargs)
		for ep in range(n_episodes):
			obs, _ = env.reset(seed=1000+ep)
			done, ret = False, 0.0
			while not done:
				with torch.no_grad():
					obs_t = self.buffer.normalize(self.to_tensor(obs).unsqueeze(0)) 
					mean, _ = self.actor(obs_t)
				action = np.clip(mean.squeeze(0).cpu().numpy(), self.act_low, self.act_high)
				obs, reward, terminated, truncated, _ = env.step(action)
				ret += reward
				done = terminated or truncated
			print(f"Episode {ep}: {ret:.1f}")
		env.close()


if __name__ == "__main__":
	agent = Agent("BipedalWalker-v3")
	agent.train()
	agent.evaluate()
