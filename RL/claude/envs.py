# envs.py
import ale_py
import gymnasium as gym
import pdb
import numpy as np
from collections import deque
import cv2
from gymnasium.spaces import Box

class NoopResetEnv(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset"""
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

class FireResetEnv(gym.Wrapper):
    """Take action on reset for environments that are fixed until firing"""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    """Make end-of-life == end-of-episode, but only reset on true game over"""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame"""
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, _, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    """Clip the reward to {+1, 0, -1} by its sign"""
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
    
    def reward(self, reward):
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    """Warp frames to 84x84 as done in the Nature paper and later work"""
    def __init__(self, env, width=84, height=84, grayscale=True):
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = Box(
                low=0,
                high=255,
                shape=(self.height, self.width, 1),
                dtype=np.uint8,
            )
        else:
            self.observation_space = Box(
                low=0,
                high=255,
                shape=(self.height, self.width, 3),
                dtype=np.uint8,
            )
    
    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame

class FrameStack(gym.Wrapper):
    """Stack k last frames"""
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], shp[2] * k),
            dtype=env.observation_space.dtype
        )
    
    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()
    
    def step(self, action):
        ob, reward, done, _, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info
    
    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):
    """Lazy frame concatenation for memory efficiency"""
    def __init__(self, frames):
        self._frames = frames
        self._out = None
    
    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out
    
    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out
    
    def __len__(self):
        return len(self._force())
    
    def __getitem__(self, i):
        return self._force()[i]

def make_atari_env(env_id, seed=0, frame_stack=4):
    """Create a wrapped atari environment"""
    env = gym.make(env_id)
    env.reset(seed=seed)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, frame_stack)
    return env

class VecEnv:
    """Base class for vectorized environments"""
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
    
    def reset(self):
        raise NotImplementedError
    
    def step(self, actions):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError

class DummyVecEnv(VecEnv):
    """Dummy vectorized environment for sequential execution"""
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        
        self.actions = None
    
    def reset(self):
        obs = [env.reset() for env in self.envs]
        # Convert LazyFrames to numpy arrays
        obs = [np.array(o) if hasattr(o, '_force') else o for o in obs]
        return np.stack(obs)
    
    def step(self, actions):
        self.actions = actions
        obs = []
        rewards = []
        dones = []
        infos = []
        
        for i, env in enumerate(self.envs):
            ob, reward, done, info = env.step(actions[i])
            if done:
                ob = env.reset()
            # Convert LazyFrames to numpy arrays
            if hasattr(ob, '_force'):
                ob = np.array(ob)
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return np.stack(obs), np.array(rewards), np.array(dones), infos
    
    def close(self):
        for env in self.envs:
            env.close()

def make_vec_envs(env_name, num_processes, seed, frame_stack=4):
    """Create vectorized environments"""
    envs = [lambda: make_atari_env(env_name, seed + i, frame_stack) 
            for i in range(num_processes)]
    envs = DummyVecEnv(envs)
    return envs