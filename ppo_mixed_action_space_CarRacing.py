import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import torchvision as tv
from torchvision.transforms import v2
from collections import deque
import gymnasium as gym

#############################################################################################################################

N_STACK = K = 4
H,W = 64,64
num_envs = 32
num_steps = 256
batch_size = num_envs*num_steps
num_minibatches = 16

update_epochs = 4
gamma = 0.999
gae_lambda = 0.95
clip_coef = 0.2
vf_coef = 0.5
ent_coef_c = 0.001
ent_coef_d = 0.01 
learning_rate = 3e-4
max_grad_norm = 0.5
total_timesteps = 2000000
hidden_dim=128
seed = 42

DEVICE = torch.device("cuda")

#############################################################################################################################

class RunningBuffer():
	def __init__(self, n_envs, n_stack=4, stepsize=1, H=H, W=W):
		self.stepsize = stepsize
		L = (n_stack - 1) * stepsize + 1
		self.buffer = torch.zeros((n_envs, L, H, W), dtype=torch.uint8)
		self.tf = v2.Compose([v2.Grayscale(),
				v2.Resize((H,W),antialias=True)
				])

	def transform(self, frames):  # np.uint8 (n_envs, 96, 96, 3)
		return self.tf(torch.from_numpy(frames).permute(0,3,1,2))  # (n_envs, 1, H, W)

	def append(self, frames):
		self.buffer = torch.cat([self.transform(frames), self.buffer[:, :-1]], dim=1)

	def initialize(self, frames, env_ids=None):
		f = self.transform(frames)
		if env_ids is None:
			self.buffer[:] = f 
		else:
			env_ids = torch.as_tensor(env_ids)
			self.buffer[env_ids] = f[env_ids]

	def get_frames(self):
		return self.buffer[:, ::self.stepsize].clone()
	

class MemoryBuffer():
	def __init__(self, num_steps, num_envs, n_stack, H, W, act_size_continuous, device, gamma, gae_lambda):
		self.obs_buf = torch.zeros((num_steps,num_envs,n_stack,H,W),dtype=torch.uint8,device=device)
		self.act_buf_c = torch.zeros((num_steps,num_envs,act_size_continuous),dtype=torch.float32,device=device)
		self.act_buf_d = torch.zeros((num_steps,num_envs),dtype=torch.long,device=device)
		self.logp_buf = torch.zeros((num_steps,num_envs),dtype=torch.float32,device=device)
		self.rew_buf = torch.zeros((num_steps,num_envs),dtype=torch.float32,device=device)
		self.done_buf = torch.zeros((num_steps,num_envs),dtype=torch.float32,device=device)
		self.val_buf = torch.zeros((num_steps,num_envs),dtype=torch.float32,device=device)
		self.step = 0

		self.device=device
		self.num_steps = num_steps
		self.num_envs = num_envs
		self.n_stack = n_stack
		self.H = H
		self.W = W
		self.act_size_continuous = act_size_continuous


		self.gamma = gamma
		self.gae_lambda = gae_lambda
		
		self.running_return = torch.zeros(num_envs,dtype=torch.float32)
		
		
		self.advantages = torch.zeros((num_steps,num_envs),dtype=torch.float32).to(device)
		self.returns = torch.zeros((num_steps,num_envs),dtype=torch.float32).to(device)
		

	def add(self, obs, act_c, act_d, logp, rew, done, val):
		t = self.step
		self.obs_buf[t] = obs
		self.act_buf_c[t] = act_c
		self.act_buf_d[t] = act_d
		self.logp_buf[t] = logp
		self.rew_buf[t] = torch.as_tensor(rew,dtype=torch.float32)
		self.done_buf[t] = torch.as_tensor(done,dtype=torch.float32)
		self.val_buf[t] = val.squeeze(-1)
		self.step = (t + 1) % self.obs_buf.shape[0]


	def calculate_advantage(self,next_value):
		next_value = next_value.squeeze(-1)
		last_gaelam = 0
		for t in reversed(range(self.num_steps)):
			next_values = next_value if t==(self.num_steps-1) else self.val_buf[t+1]
			nonterminal = 1.0-self.done_buf[t]
			delta = self.rew_buf[t] + self.gamma*nonterminal*next_values - self.val_buf[t]
			last_gaelam = delta + self.gamma*self.gae_lambda*nonterminal*last_gaelam
			self.advantages[t]=last_gaelam
		self.returns = self.advantages + self.val_buf
	
	def minibatches(self,batch_size):
		total = self.num_steps * self.num_envs
		mixed_idx = torch.randperm(total,device=self.device)
		states = self.obs_buf.reshape(total,self.n_stack,self.H,self.W)
		actions_c = self.act_buf_c.reshape(total,-1)
		actions_d = self.act_buf_d.reshape(-1)
		log_probs = self.logp_buf.reshape(-1)
		advantages = self.advantages.reshape(-1)
		returns = self.returns.reshape(-1)
		for start in range(0, total, batch_size):
			mb = mixed_idx[start:start+batch_size]
			yield states[mb], actions_c[mb], actions_d[mb], log_probs[mb], advantages[mb], returns[mb]
		
		
#############################################################################################################################	


def layer_init(layer,std=np.sqrt(2),bias=0.0):
	nn.init.orthogonal_(layer.weight,gain=std)
	nn.init.constant_(layer.bias,bias)
	return layer



class BaseNet(nn.Module):
	def __init__(self, hidden_size):
		super().__init__()
		self.base_net = nn.Sequential(
			layer_init(nn.Conv2d(4,16,3,1,1),std=1.0),
			nn.SELU(),
			layer_init(nn.Conv2d(16,16,3,1,1),std=1.0),
			nn.MaxPool2d(2),#32x32
			nn.SELU(),
			layer_init(nn.Conv2d(16,32,3,1,1),std=1.0),
			nn.MaxPool2d(2),#16x16
			nn.SELU(),
			nn.Flatten(),
			layer_init(nn.Linear(16*16*32,2*hidden_size),std=1.0),
			nn.SELU()
		)
	def forward(self,x):
		return self.base_net(x.float()/255)  # uint8 -> [0,1]
		
class Actor(nn.Module):
	def __init__(self,hidden_size):
		super().__init__()
		self.net = nn.Sequential(
			layer_init(nn.Linear(2*hidden_size,hidden_size)),
			nn.Tanh(),
			layer_init(nn.Linear(hidden_size,hidden_size)),
			nn.Tanh()
		)
		self.steer_mu = layer_init(nn.Linear(hidden_size,1),std=0.01)
		self.pedal_logits = layer_init(nn.Linear(hidden_size,3),std=0.01)
		self.log_std = nn.Parameter(torch.full((1,),-0.5))

	def forward(self,hidden_state):
		h = self.net(hidden_state)
		return Normal(self.steer_mu(h),self.log_std.exp()), Categorical(logits=self.pedal_logits(h))
		
class Critic(nn.Module):
	def __init__(self,hidden_size):
		super().__init__()
		self.net = nn.Sequential(
			layer_init(nn.Linear(2*hidden_size,hidden_size),std=1.0),
			nn.SELU(),
			layer_init(nn.Linear(hidden_size,hidden_size),std=1.0),
			nn.SELU(),
			layer_init(nn.Linear(hidden_size,1),std=1.0)		
		)
		
	def forward(self,hidden_state):
		return self.net(hidden_state)

#############################################################################################################################


class ActionRepeat(gym.Wrapper):
	def __init__(self,env,k=K,patience=30):
		super().__init__(env)
		self.k = k
		self.patience = patience
		self.idle = 0

	def reset(self,**kwargs):
		self.idle = 0
		return self.env.reset(**kwargs)

	def step(self,action):
		total = 0.0
		for _ in range(self.k):
			obs, r, term, trunc, info = self.env.step(action)
			total += r
			if term or trunc:
				break
		self.idle = 0 if total > 0 else self.idle + 1
		if self.idle >= self.patience:
			trunc = True
		return obs, total, term, trunc, info

def make_env(name,idx,env_kwargs,seed=seed):
	def thunk():
		return ActionRepeat(gym.make(name,**env_kwargs))
	return thunk

def make_envs(name,env_kwargs,num_envs=num_envs,seed=seed):
	return gym.vector.AsyncVectorEnv([make_env(name,i,env_kwargs,seed) for i in range(num_envs)],
				autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
#############################################################################################################################


class Agent():
	def __init__(self,env_name,hidden_dim=hidden_dim,gamma=gamma,gae_lambda=gae_lambda,**env_kwargs):
		self.env_name = env_name
		self.env_kwargs = env_kwargs
		self.envs = make_envs(env_name,env_kwargs)
		self.device = DEVICE

		self.n_stack = N_STACK 
		self.act_size_c = 1 

		self.base = BaseNet(hidden_dim).to(self.device)
		self.actor = Actor(hidden_dim).to(self.device)
		self.critic = Critic(hidden_dim).to(self.device)
		self.params = list(self.base.parameters())+list(self.actor.parameters())+list(self.critic.parameters())
		self.optimizer = optim.Adam(self.params,lr=learning_rate,eps=1e-5)

		self.frames = RunningBuffer(num_envs,n_stack=self.n_stack,stepsize=1)
		self.buffer = MemoryBuffer(num_steps,num_envs,self.n_stack,H,W,self.act_size_c,self.device,gamma=gamma,gae_lambda=gae_lambda)


		self.global_step = 0
		self.episodes = []


	def get_action_and_value(self,obs,action_c=None,action_d=None):
		hidden_state = self.base(obs)
		dist_c, dist_d = self.actor(hidden_state)
		if action_c is None:
			action_c = dist_c.sample()
			action_d = dist_d.sample()
		log_prob = dist_c.log_prob(action_c).sum(-1) + dist_d.log_prob(action_d) 
		entropy_c = dist_c.entropy().sum(-1)
		entropy_d = dist_d.entropy()
		value = self.critic(hidden_state).squeeze(-1)
		return action_c, action_d, log_prob, entropy_c, entropy_d, value

	def to_env_action(self,action_c,action_d):
		steer = action_c.squeeze(-1).clamp(-1.0,1.0)
		gas = (action_d==1).float()*0.3
		brake = (action_d==2).float()*0.8
		return torch.stack([steer,gas,brake],dim=-1).cpu().numpy()

	def collect(self):
		for t in range(num_steps):
			obs_t = self.frames.get_frames().to(self.device)
			with torch.no_grad():
				action_c, action_d, log_prob, _, _, value = self.get_action_and_value(obs_t)
			env_action = self.to_env_action(action_c,action_d)
			next_obs, reward, terminated, truncated, infos = self.envs.step(env_action)
			done = terminated | truncated
			self.buffer.add(obs_t,action_c,action_d,log_prob,reward/10.0,done,value)
			self.frames.append(next_obs)
			self.global_step += num_envs
			self.ep_ret += reward
			for i in np.where(done)[0]:
				self.episodes.append((self.global_step,self.ep_ret[i]))
				self.ep_ret[i] = 0.0
			if done.any():
				self.frames.initialize(next_obs,env_ids=done) 
		with torch.no_grad():
			_, _, _, _, _, next_value = self.get_action_and_value(self.frames.get_frames().to(self.device))
		self.buffer.calculate_advantage(next_value)


	def save(self,path):
		torch.save({"base": self.base.state_dict(), "actor": self.actor.state_dict(), "critic": self.critic.state_dict()},path)

	def load(self,path):
		ckpt = torch.load(path,map_location=self.device)
		self.base.load_state_dict(ckpt["base"])
		self.actor.load_state_dict(ckpt["actor"])
		self.critic.load_state_dict(ckpt["critic"])


	def learn(self):
		kls, clipfracs, ents_c, ents_d = [],[],[],[]
		ratio_check = None
		for epoch in range(update_epochs):
			for mb_states, mb_ac, mb_ad, mb_old_logp, mb_adv, mb_returns in self.buffer.minibatches(batch_size//num_minibatches):
				_, _, newlogprob, ent_c, ent_d, newvalue = self.get_action_and_value(mb_states,mb_ac,mb_ad)
				logratio = newlogprob - mb_old_logp
				ratio = logratio.exp()

				with torch.no_grad():
					if ratio_check is None:
						ratio_check = (ratio-1).abs().max().item()
					kls.append(((ratio-1)-logratio).mean().item())
					clipfracs.append(((ratio-1).abs()>clip_coef).float().mean().item())
					ents_c.append(ent_c.mean().item())
					ents_d.append(ent_d.mean().item())
				mb_adv = (mb_adv-mb_adv.mean())/(mb_adv.std()+1e-8)
				pg_loss = torch.max(-mb_adv*ratio, -mb_adv*torch.clamp(ratio,1-clip_coef,1+clip_coef)).mean()
				v_loss = 0.5 * ((newvalue-mb_returns)**2).mean()
				loss = pg_loss + vf_coef*v_loss - ent_coef_c*ent_c.mean() - ent_coef_d*ent_d.mean()
				self.optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.params,max_grad_norm)
				self.optimizer.step()
		return {"approx_kl": np.mean(kls), "clipfrac": np.mean(clipfracs), "ratio_check": ratio_check,
				"ent_c": np.mean(ents_c), "ent_d": np.mean(ents_d)}

	def train(self,total_timesteps=total_timesteps):
		num_iterations = total_timesteps//batch_size
		obs, _ = self.envs.reset(seed=seed)
		self.frames.initialize(obs)
		self.ep_ret = np.zeros(num_envs)
		best = -float("inf")
		for iteration in range(1,num_iterations+1):
			frac = 1.0 - (iteration-1)/num_iterations
			self.optimizer.param_groups[0]["lr"] = frac * learning_rate
			self.collect()
			stats = self.learn()
			recent = [r for _,r in self.episodes[-20:]]
			avg = np.mean(recent) if recent else float("nan")
			std = self.actor.log_std.exp().item()
			saved = ""
			if len(recent) == 20 and avg > best:
				best = avg
				self.save("carracing_best.pt")
				saved = "  *saved*"
			print(f"it {iteration:>3}/{num_iterations}  step {self.global_step:>7}  return(20) {avg:7.1f}  std {std:.2f}  ent_d {stats['ent_d']:.3f}  kl {stats['approx_kl']:.4f}  clip {stats['clipfrac']:.3f}  ratio0 {stats['ratio_check']:.1e}{saved}")
		self.save("carracing_last.pt")
		self.envs.close()

	def evaluate(self,n_episodes=5,render=True):
		env = ActionRepeat(gym.make(self.env_name, render_mode="human" if render else None, **self.env_kwargs),patience=10**9)
		frames = RunningBuffer(1,n_stack=self.n_stack,stepsize=1)
		for ep in range(n_episodes):
			obs, _ = env.reset(seed=1000+ep)
			frames.initialize(obs[None])  # (1, 96, 96, 3)
			done, ret = False, 0.0
			while not done:
				with torch.no_grad():
					hidden = self.base(frames.get_frames().to(self.device))
					dist_c, dist_d = self.actor(hidden)
					action_c = dist_c.mean  # deterministic
					action_d = dist_d.probs.argmax(-1)
				obs, reward, terminated, truncated, _ = env.step(self.to_env_action(action_c,action_d)[0])
				frames.append(obs[None])
				ret += reward
				done = terminated or truncated
			print(f"Episode {ep}: {ret:.1f}")
		env.close()


if __name__ == "__main__":
	torch.manual_seed(seed)
	np.random.seed(seed)
	agent = Agent("CarRacing-v3")
	agent.train()
	agent.load("carracing_best.pt")
	agent.evaluate()


