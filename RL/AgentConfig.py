from data.DataFeed import AtariDataFeed 
from model.policy_model import PolicyModel 
from model.value_model import ValueModel 
from model.q_learning import TDDoubleQLearning 

@dataclass 
class AgentConfig:
    def __init__(self):
        return

@dataclass 
class AtariAgentConfig:
    def __init__(self, ale_name):
        self.data_feed_name = ale_name 
        self.data_feed = AtariDataFeed(self.data_feed_name)
        self.q_learning = QLearning()
        self.policy_lr = 0.001 
        self.q_lr = 0.001 
        self.policy_update_interval = 5 
        self.meta_update_interval = -1 
        