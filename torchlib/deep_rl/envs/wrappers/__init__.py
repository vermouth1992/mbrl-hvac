from .model_based import model_based_wrapper_dict
from .common import ClipActionWrapper, ObservationDTypeWrapper, ObservationActionWrapper
import gym


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)


def get_model_based_wrapper(env_name):
    if env_name not in model_based_wrapper_dict:
        raise ValueError('No cost fn defined for {}'.format(env_name))
    return model_based_wrapper_dict[env_name]
