"""
Helper functions for using Reinforcement Learning with CityLearn environment.
"""

from gym.spaces import Box


def dict_to_action_space(aspace_dict):
    return Box(low=aspace_dict["low"],
               high=aspace_dict["high"],
               dtype=aspace_dict["dtype"])


def action_space_to_dict(aspace):
    """ Only for box space """
    return {"high": aspace.high,
            "low": aspace.low,
            "shape": aspace.shape,
            "dtype": str(aspace.dtype)
            }


def env_reset(env):
    """Construct environment information whilst resetting."""
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations }
    return obs_dict