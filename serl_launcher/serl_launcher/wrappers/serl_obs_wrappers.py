import gym
from gym.spaces import flatten_space, flatten


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.env.observation_space["state"]),
                **(self.env.observation_space["images"]),
            }
        )

    def observation(self, obs):
        obs = {
            "state": flatten(self.env.observation_space["state"], obs["state"]),
            **(obs["images"]),
        }
        return obs


class ScaleObservationWrapper(gym.ObservationWrapper):
    """
    This observation wrapper scales the observations with the provided hyperparams
    (to somewhat normalize the observations space)
    """

    def __init__(
        self,
        env,
        translation_scale=100.0,
        rotation_scale=10.0,
        force_scale=1.0,
        torque_scale=10.0,
    ):
        super().__init__(env)
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.force_scale = force_scale
        self.torque_scale = torque_scale

    def scale_wrapper_get_scales(self):
        return dict(
            translation_scale=self.translation_scale,
            rotation_scale=self.rotation_scale,
            force_scale=self.force_scale,
            torque_scale=self.torque_scale,
        )

    def observation(self, obs):
        obs["state"]["tcp_pose"][:3] *= self.translation_scale
        obs["state"]["tcp_pose"][3:] *= self.rotation_scale
        obs["state"]["tcp_vel"][:3] *= self.translation_scale
        obs["state"]["tcp_vel"][3:] *= self.rotation_scale
        obs["state"]["tcp_force"] *= self.force_scale
        obs["state"]["tcp_torque"] *= self.torque_scale
        return obs
