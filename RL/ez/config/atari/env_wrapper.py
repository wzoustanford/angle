import numpy as np
from core.game import Game
from core.utils import arr_to_str


class AtariWrapper(Game):
    def __init__(self, env, discount: float, cvt_string=True, seed=None):
        """Atari Wrapper
        Parameters
        ----------
        env: Any
            another env wrapper
        discount: float
            discount of env
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
        seed: int
            random seed for the environment
        """
        super().__init__(env, env.action_space.n, discount)
        self.cvt_string = cvt_string
        self.seed = seed

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def get_max_episode_steps(self):
        return self.env.get_max_episode_steps()

    def step(self, action):
        # Handle both old gym API and new gymnasium API
        step_result = self.env.step(action)
        if len(step_result) == 5:
            # gymnasium API: (obs, reward, terminated, truncated, info)
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            # old gym API: (obs, reward, done, info)
            observation, reward, done, info = step_result
        
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        # Return the 4-value format that EfficientZero expects
        return observation, reward, done, info

    def reset(self, **kwargs):
        # If seed is stored and not already in kwargs, add it
        if self.seed is not None and 'seed' not in kwargs:
            kwargs['seed'] = self.seed
            # Only use the seed once
            self.seed = None
            
        # Handle both old gym API and new gymnasium API
        reset_result = self.env.reset(**kwargs)
        if isinstance(reset_result, tuple):
            # gymnasium API: returns (observation, info)
            observation, _ = reset_result
        else:
            # old gym API: returns observation
            observation = reset_result
            
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        # Return just the observation for compatibility with EfficientZero
        return observation

    def close(self):
        self.env.close()
