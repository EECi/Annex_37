from gym.spaces import Box


def dict_to_action_space(aspace_dict):
    return Box(low=aspace_dict["low"],
               high=aspace_dict["high"],
               dtype=aspace_dict["dtype"])


class Agent:
    def __init__(self):
        self.num_buildings = None
        self.action_space = None

        # You may want to track some variables, eg. observations or actions from previous time steps. Do so here.
        # ==============================================================================================================
        self.prev_observation = None
        # ==============================================================================================================

    def register_reset(self, observation):
        """Get the first observation after env.reset, return action"""
        action_space = observation["action_space"]
        self.action_space = [dict_to_action_space(asd) for asd in action_space]
        obs = observation["observation"]
        self.num_buildings = len(obs)
        return self.compute_action(obs)

    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def compute_action(self, observation):
        """
        Below is the suggested format for this function:
        - Use the observation to compute the reward.
        - Update the agent parameters using the reward.
        - Compute the action for the observation

        Inputs:
            observation - List of observations from the env: observation[building_index, observation_index]
        Returns:
            actions - List of actions: actions[building_index]
                eg. for 5 buildings each with action 0.5 the format is as follows [[0.5], [0.5], [0.5], [0.5], [0.5]]
                Actions are between 0 and 1.
        Please make sure the action for each building is in the same order as the observations for each building.

        """
        assert self.num_buildings is not None

        # Define your reward here
        # ==============================================================================================================
        rewards = [0 for _ in range(self.num_buildings)]
        # ==============================================================================================================

        # Update your agent parameters here
        # ==============================================================================================================

        # ==============================================================================================================

        # Compute your action here
        # ==============================================================================================================
        actions = [[0] for _ in range(self.num_buildings)]
        # ==============================================================================================================
        return actions
