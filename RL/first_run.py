import gymnasium as gym
import ale_py, pdb

gym.register_envs(ale_py)

env = gym.make('ALE/SpaceInvaders-v5')
obs, info = env.reset()
for i in range(10):
    act = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(act)
    pdb.set_trace()
env.close()

