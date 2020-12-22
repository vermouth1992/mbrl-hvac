import numpy as np
from gym import ActionWrapper, Env, spaces, Wrapper, ObservationWrapper


class ClipActionWrapper(ActionWrapper):
    def __init__(self, env: Env):
        super(ClipActionWrapper, self).__init__(env=env)
        if isinstance(env.action_space, spaces.Box):
            self.lower_bound = env.action_space.low
            self.upper_bound = env.action_space.high
        else:
            self.lower_bound = None
            self.upper_bound = None

    def action(self, action):
        return np.clip(action, a_min=self.lower_bound, a_max=self.upper_bound)


class ObservationActionWrapper(Wrapper):
    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        return self.observation(observation)

    def step(self, action):
        action = self.action(action)
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        raise NotImplementedError

    def action(self, action):
        raise NotImplementedError


class ObservationDTypeWrapper(ObservationWrapper):
    def __init__(self, env, obs_dtype=np.float32):
        super(ObservationDTypeWrapper, self).__init__(env=env)
        self.obs_dtype = obs_dtype

    def observation(self, observation):
        return observation.astype(self.obs_dtype)
