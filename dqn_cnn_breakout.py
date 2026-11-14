import copy
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
from torch.utils.tensorboard import SummaryWriter

gym.register_envs(ale_py)

#######################################################################################################################

ENV_ID = "ALE/Breakout-v5"
SEED = 0
NUM_ENVS = 8
TOTAL_STEPS = 5_000_000
FRAME_SKIP = 4
STACK = 4
SCREEN = 84

BUFFER_SIZE = 100_000
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
LEARNING_STARTS = 50_000
TRAIN_FREQ = 4
TARGET_SYNC = 1_000
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.01, 1_000_000
LOG_EVERY = 10_000
SAVE_EVERY = 500_000

#######################################################################################################################

class MaxAndSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        frames = []
        for _ in range(self.skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            frames.append(obs)
            if term or trunc:
                break
        obs = np.maximum(frames[-1], frames[-2]) if len(frames) > 1 else frames[-1]
        return obs, total_reward, term, trunc, info


class FireOnLifeLoss(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        self.lives = 0

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, _, _, info = self.env.step(1)
        self.lives = info["lives"]
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        if info["lives"] < self.lives and not (term or trunc):
            obs, r2, term, trunc, info = self.env.step(1)
            reward += r2
        self.lives = info["lives"]
        return obs, reward, term, trunc, info


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id, frameskip=1)
        env = MaxAndSkip(env, skip=FRAME_SKIP)
        env = FireOnLifeLoss(env)
        env = gym.wrappers.ResizeObservation(env, (SCREEN, SCREEN))
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
        env = gym.wrappers.FrameStackObservation(env, stack_size=STACK)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk

#######################################################################################################################

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng()
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, actions, rewards, dones, next_obs):
        idx = (self.ptr + np.arange(len(actions))) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.next_obs[idx] = next_obs
        self.ptr = (self.ptr + len(actions)) % self.capacity
        self.size = min(self.size + len(actions), self.capacity)

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        to_t = lambda arr: torch.from_numpy(arr[idx]).to(self.device, non_blocking=True)
        return to_t(self.obs), to_t(self.actions), to_t(self.rewards), to_t(self.dones), to_t(self.next_obs)

#######################################################################################################################

class QNet(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *obs_shape)).shape[1]
        self.head = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        return self.head(self.cnn(x.float() / 255.0))

#######################################################################################################################

def epsilon(step):
    frac = min(step / EPS_DECAY, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)

#######################################################################################################################

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, SEED + i) for i in range(NUM_ENVS)])
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    q = QNet(obs_shape, n_actions).to(device)
    q_target = copy.deepcopy(q).requires_grad_(False)
    opt = torch.optim.Adam(q.parameters(), lr=LR, eps=1.5e-4)
    buffer = ReplayBuffer(BUFFER_SIZE, obs_shape, device)
    writer = SummaryWriter(f"runs/dqn_breakout_{int(time.time())}")
    rng = np.random.default_rng(SEED)
    returns = deque(maxlen=100)

    obs, info = envs.reset(seed=SEED)
    lives = info["lives"].copy()
    autoreset = np.zeros(NUM_ENVS, dtype=bool)
    step, updates, loss = 0, 0, None
    next_log, next_save = LOG_EVERY, SAVE_EVERY
    t0 = time.time()

	## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ##

    while step < TOTAL_STEPS:
        eps = epsilon(step)
        with torch.no_grad():
            greedy = q(torch.from_numpy(obs).to(device)).argmax(1).cpu().numpy()
        explore = rng.random(NUM_ENVS) < eps
        actions = np.where(explore, rng.integers(n_actions, size=NUM_ENVS), greedy)

        next_obs, rewards, term, trunc, info = envs.step(actions)
        life_lost = info["lives"] < lives
        done_target = term | life_lost
        keep = ~autoreset
        buffer.add(obs[keep], actions[keep], np.sign(rewards[keep]), done_target[keep], next_obs[keep])

        if "episode" in info:
            for i in np.flatnonzero(info["_episode"]):
                returns.append(float(info["episode"]["r"][i]))
                writer.add_scalar("charts/episode_return", info["episode"]["r"][i], step)

        lives = info["lives"].copy()
        autoreset = term | trunc
        obs = next_obs
        step += NUM_ENVS

        if buffer.size >= LEARNING_STARTS:
            for _ in range(max(1, NUM_ENVS // TRAIN_FREQ)):
                b_obs, b_act, b_rew, b_done, b_next = buffer.sample(BATCH_SIZE)
                with torch.no_grad():
                    next_act = q(b_next).argmax(1, keepdim=True)
                    next_q = q_target(b_next).gather(1, next_act).squeeze(1)
                    target = b_rew + GAMMA * (1.0 - b_done) * next_q
                q_sa = q(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(q_sa, target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                opt.step()
                updates += 1
                if updates % TARGET_SYNC == 0:
                    q_target.load_state_dict(q.state_dict())

        if step >= next_log:
            next_log += LOG_EVERY
            sps = int(step / (time.time() - t0))
            mean_ret = np.mean(returns) if returns else float("nan")
            writer.add_scalar("charts/epsilon", eps, step)
            writer.add_scalar("charts/sps", sps, step)
            if loss is not None:
                writer.add_scalar("losses/td_loss", loss.item(), step)
                writer.add_scalar("losses/q_mean", q_sa.mean().item(), step)
            print(f"step {step:>9} | eps {eps:.3f} | ret100 {mean_ret:6.1f} | updates {updates:>7} | sps {sps}")

        if step >= next_save:
            next_save += SAVE_EVERY
            torch.save(q.state_dict(), "dqn_breakout.pt")

    torch.save(q.state_dict(), "dqn_breakout.pt")
    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
