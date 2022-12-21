from citylearn.reward_function import RewardFunction


class UserReward(RewardFunction):
    def __init__(self, agent_count, **kwargs):
        super().__init__(agent_count, **kwargs)
