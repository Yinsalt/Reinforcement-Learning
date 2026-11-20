import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np
import gymnasium as gym

import time
from collections import deque


###########################################################################


BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_STEPS = 500_000
HIDDEN_SIZE = 256
LR_A = 3e-4
LR_C = 3e-4
UPDATE_EVERY = 50
RENDER_EPISODES = 5
ENV_ID = "Humanoid-v5"
ENV_KWARGS = dict(
	include_cinert_in_observation=False,
	include_cvel_in_observation=False,
	include_qfrc_actuator_in_observation=False,
	include_cfrc_ext_in_observation=False,
)

###########################################################################


def initialize_layer(layer, std=np.sqrt(2),bias=0):
	nn.init.orthogonal_(layer.weight, gain=std)
	nn.init.constant_(layer.bias,bias)
	return layer


class ResBlock(nn.Module):
	def __init__(self, layer_size, factor=4):
		super().__init__()
		self.layer_size = layer_size
		self.LN = nn.LayerNorm(self.layer_size)
		self.hidden = max(1,int(layer_size*factor))
		self.net = nn.Sequential(
			initialize_layer(nn.Linear(layer_size,self.hidden)),
			nn.ReLU(),
			initialize_layer(nn.Linear(self.hidden,layer_size))
		)
		
	def forward(self,x):
		return x + self.net(self.LN(x))




class Actor(nn.Module):
	def __init__(self, state_size, hidden_size, action_size,device,lr,wd=0.0,max_action=1.0):
		super().__init__()
		self.max_action = max_action
		self.base_net = nn.Sequential(
			initialize_layer(nn.Linear(state_size,hidden_size)),
			nn.ReLU(),
			ResBlock(hidden_size), nn.LayerNorm(hidden_size),
			initialize_layer(nn.Linear(hidden_size,2*hidden_size)),
			nn.ReLU(),
			ResBlock(2*hidden_size), nn.LayerNorm(2*hidden_size),
			initialize_layer(nn.Linear(2*hidden_size,hidden_size)),
			nn.ReLU()
		)
		self.mu = nn.Sequential(
			ResBlock(hidden_size), nn.LayerNorm(hidden_size),
			initialize_layer(nn.Linear(hidden_size,action_size),std=0.01)
		)
		self.log_std = nn.Sequential(
			ResBlock(hidden_size), nn.LayerNorm(hidden_size),
			initialize_layer(nn.Linear(hidden_size,action_size),std=0.01)
		)
		self.to(device)
		self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

	def forward(self,x):
		h = self.base_net(x)
		mu = self.mu(h)
		log_std = self.log_std(h).clamp(LOG_STD_MIN,LOG_STD_MAX)
		return mu, log_std
		
	def sample(self,x):
		mu, log_std = self(x)
		dist = Normal(mu,log_std.exp())
		u = dist.rsample()
		a = torch.tanh(u)
		log_prob = dist.log_prob(u) - 2*(np.log(2) - u - F.softplus(-2*u))
		log_prob = log_prob.sum(dim=-1,keepdim=True)
		return self.max_action*a, log_prob, self.max_action*torch.tanh(mu)
		

class Critic(nn.Module):
	def __init__(self,state_size,action_size,hidden_size,device,lr,wd=0.0):
		super().__init__()
		self.net = nn.Sequential(
			initialize_layer(nn.Linear(state_size+action_size,hidden_size)),
			nn.ReLU(),
			ResBlock(hidden_size), nn.LayerNorm(hidden_size),
			initialize_layer(nn.Linear(hidden_size,2*hidden_size)),
			nn.ReLU(),
			ResBlock(2*hidden_size), nn.LayerNorm(2*hidden_size),
			initialize_layer(nn.Linear(2*hidden_size,hidden_size)),
			nn.ReLU(),
			ResBlock(hidden_size), nn.LayerNorm(hidden_size),
			initialize_layer(nn.Linear(hidden_size,1),std=1.0)
		)
		self.to(device)
		self.optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)

	def forward(self,state,action):
		return self.net(torch.cat([state,action],dim=-1))




###########################################################################



class ReplayBuffer():

	def __init__(self, obs_size, action_size, max_size,device):
		
		self.step = 0
		self.max_size = max_size
		self.obs_size = obs_size
		self.action_size = action_size
		
		self.states = torch.zeros((max_size,obs_size),dtype=torch.float32)
		self.actions = torch.zeros((max_size,action_size),dtype=torch.float32)
		self.rewards = torch.zeros((max_size,1),dtype=torch.float32)
		self.terminated = torch.zeros((max_size,1),dtype=torch.float32)
		self.states_ = torch.zeros((max_size,obs_size),dtype=torch.float32)
		
		self.device = device
		
	def save(self,state,action,reward,state_,terminated):
		t = self.step%self.max_size
		
		self.states[t]=torch.as_tensor(state,dtype=torch.float32)
		self.actions[t]=torch.as_tensor(action,dtype=torch.float32)
		self.rewards[t]=float(reward)
		self.states_[t]=torch.as_tensor(state_,dtype=torch.float32)
		self.terminated[t]=float(terminated)
		self.step +=1
		
	def __len__(self):
		return min(self.max_size,self.step)
		
	def sample(self,batchsize=64):
		idx = torch.randint(0,len(self),(batchsize,))
		return (self.states[idx].to(self.device),
				self.actions[idx].to(self.device),
				self.rewards[idx].to(self.device),
				self.states_[idx].to(self.device),
				self.terminated[idx].to(self.device))


###########################################################################

class Agent():
	def __init__(self, env, num_steps, hidden_size, lr_a, lr_c, lr_alpha=3e-4, wd_c=0.0, gamma=GAMMA, tau=TAU, batch_size = BATCH_SIZE, device = DEVICE):
		self.env = env
		self.num_steps = num_steps
		self.hidden_size = hidden_size
		self.batch_size = batch_size
		self.device = device
		self.gamma = gamma
		self.tau = tau
		self.warmup_steps = 10_000
		self.training = True
		
		self.max_action = float(env.action_space.high[0])
		self.act_size = env.action_space.shape[0]
		self.obs_size = env.observation_space.shape[0]
		
		self.actor = Actor(self.obs_size, self.hidden_size, self.act_size, device, lr_a, 0.0, self.max_action)
		self.critic_1 = Critic(self.obs_size, self.act_size, self.hidden_size, device, lr_c, wd_c)
		self.critic_2 = Critic(self.obs_size, self.act_size, self.hidden_size, device, lr_c, wd_c)
		self.tgt_critic_1 = Critic(self.obs_size, self.act_size, self.hidden_size, device, lr_c, wd_c)
		self.tgt_critic_2 = Critic(self.obs_size, self.act_size, self.hidden_size, device, lr_c, wd_c)
		
		self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
		self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
		self.target_entropy = -float(self.act_size)
		
		self.buffer = ReplayBuffer(self.obs_size, self.act_size, num_steps, device)
		
		self.initialize()
		
	@property
	def alpha(self):
		return self.log_alpha.exp().detach()
		
	def initialize(self):
		for net, tgt in ((self.critic_1,self.tgt_critic_1),(self.critic_2,self.tgt_critic_2)):
			tgt.load_state_dict(net.state_dict())
			tgt.requires_grad_(False)
			tgt.eval()
			
	def set_train(self):
		self.training = True
		
	def set_eval(self):
		self.training = False
		
	def soft_update(self):
		with torch.no_grad():
			for net, tgt in ((self.critic_1,self.tgt_critic_1),(self.critic_2,self.tgt_critic_2)):
				for tp, p in zip(tgt.parameters(), net.parameters()):
					tp.lerp_(p, self.tau)
					
	def get_action(self,obs):
		if self.training and self.buffer.step < self.warmup_steps:
			return self.env.action_space.sample()
		obs = torch.as_tensor(obs,dtype=torch.float32,device=self.device).unsqueeze(0)
		with torch.no_grad():
			a, _, a_det = self.actor.sample(obs)
		return (a if self.training else a_det).cpu().numpy()[0]
		
	
	def learn(self):
		if len(self.buffer) < self.batch_size:
			return None
		
		s,a,r,s_,term = self.buffer.sample(self.batch_size)
		
		with torch.no_grad():
			a_, log_prob_, _ = self.actor.sample(s_)
			q_ = torch.min(self.tgt_critic_1(s_,a_),self.tgt_critic_2(s_,a_))
			pred = r + self.gamma*(1-term)*(q_ - self.alpha*log_prob_)
		
		loss_c1 = F.mse_loss(self.critic_1(s,a),pred)
		loss_c2 = F.mse_loss(self.critic_2(s,a),pred)
		
		self.critic_1.optimizer.zero_grad()
		self.critic_2.optimizer.zero_grad()
		loss_c1.backward()
		loss_c2.backward()
		self.critic_1.optimizer.step()
		self.critic_2.optimizer.step()
		
		self.critic_1.requires_grad_(False)
		self.critic_2.requires_grad_(False)
		
		a_pi, log_prob, _ = self.actor.sample(s)
		q_pi = torch.min(self.critic_1(s,a_pi),self.critic_2(s,a_pi))
		loss_a = (self.alpha*log_prob - q_pi).mean()
		
		self.actor.optimizer.zero_grad()
		loss_a.backward()
		self.actor.optimizer.step()
		
		self.critic_1.requires_grad_(True)
		self.critic_2.requires_grad_(True)
		
		loss_alpha = -(self.log_alpha.exp()*(log_prob.detach() + self.target_entropy)).mean()
		
		self.alpha_optimizer.zero_grad()
		loss_alpha.backward()
		self.alpha_optimizer.step()
		
		self.soft_update()
		
		return loss_a.item(), loss_c1.item(), loss_c2.item(), self.alpha.item(), -log_prob.mean().item()
		
		
		


###########################################################################

if __name__ == "__main__":
	env = gym.make(ENV_ID,**ENV_KWARGS)
	agent = Agent(env,TOTAL_STEPS,HIDDEN_SIZE,LR_A,LR_C)
	print(f"obs {agent.obs_size} | act {agent.act_size} | max_action {agent.max_action} | target_entropy {agent.target_entropy}")
	
	scores = deque(maxlen=10)
	lengths = deque(maxlen=10)
	speeds = deque(maxlen=10)
	stats = deque(maxlen=1000)
	episode = 0
	ep_return, ep_len, ep_vel = 0.0, 0, 0.0
	t0 = time.time()
	solve_score = env.spec.reward_threshold
	
	obs, _ = env.reset()
	
	for step in range(TOTAL_STEPS):
		action = agent.get_action(obs)
		obs_, reward, terminated, truncated, info = env.step(action)
		agent.buffer.save(obs,action,reward,obs_,terminated)
		ep_return += reward
		ep_len += 1
		ep_vel += info.get("x_velocity",0.0)
		obs = obs_
		
		if step >= agent.warmup_steps and step % UPDATE_EVERY == 0:
			for _ in range(UPDATE_EVERY):
				out = agent.learn()
				if out is not None:
					stats.append(out)
		
		if terminated or truncated:
			episode += 1
			scores.append(ep_return)
			lengths.append(ep_len)
			speeds.append(ep_vel/ep_len)
			sps = (step+1)/(time.time()-t0)
			if stats:
				la, lc1, lc2, al, ent = np.mean(stats,axis=0)
			else:
				la = lc1 = lc2 = al = ent = float("nan")
			print(f"Ep {episode:5d} | Step {step+1:7d} | Ret {ep_return:7.1f} | Avg10 {np.mean(scores):7.1f} | Len {np.mean(lengths):6.1f} | vx {np.mean(speeds):5.2f} | A {la:7.2f} | C {(lc1+lc2)/2:7.3f} | α {al:6.3f} | H {ent:6.2f} | {sps:4.0f} sps")
			
			if solve_score is not None and len(scores) == scores.maxlen and np.mean(scores) > solve_score:
				print(f"Solved in {episode} episodes / {step+1} steps")
				break
			
			ep_return, ep_len, ep_vel = 0.0, 0, 0.0
			obs, _ = env.reset()
	
	env.close()
	torch.save(agent.actor.state_dict(),"sac_humanoid_actor.pt")
	
	env = gym.make(ENV_ID,render_mode="human",**ENV_KWARGS)
	agent.env = env
	agent.set_eval()
	for ep in range(RENDER_EPISODES):
		obs, _ = env.reset()
		done, total, length = False, 0.0, 0
		while not done:
			action = agent.get_action(obs)
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated or truncated
			total += reward
			length += 1
		print(f"Render-Episode {ep+1}: {total:.1f} ({length} Steps)")
	env.close()
