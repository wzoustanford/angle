import gymnasium as gym 

class DataFeed: 
    def __init(self):
        return 
    def next(self): 
        return 

class AtariDataFeed(DataFeed)
    def __init__(self, ale_name): 
        gym.register_envs(ale_py) 
        # e.g. ale_name = 'ALE/SpaceInvaders-v5' 
        self.env = gym.make(ale_name) 

    def next(self, act): 
        return env.step(act) 
