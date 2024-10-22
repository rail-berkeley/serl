from scipy.spatial.transform import Rotation as R
import gymnasium as gym
import numpy as np
from gym import Env
from franka_env.utils.transformations import (
    construct_homogeneous_matrix,
    construct_rotation_matrix
)


class RelativeFrame(gym.Wrapper):
    """
    This wrapper transforms the observation and action to be expressed in the end-effector frame.
    Optionally, it can transform the tcp_pose into a relative frame defined as the reset pose.

    This wrapper is expected to be used on top of the base UR5 environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,)),
                "tcp_force": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "tcp_torque": spaces.Box(-np.inf, np.inf, shape=(3,)),
                "gripper_state": spaces.Box(-np.inf, np.inf, shape=(2,)),
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.rotation_matrix = np.eye((3))
        self.rotation_matrix_reset = np.eye((3))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Homogeneous transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)

        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(info["intervene_action"])

        # Update rotation matrix
        self.rotation_matrix = construct_rotation_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.rotation_matrix = construct_rotation_matrix(obs["state"]["tcp_pose"])
        self.rotation_matrix_reset = self.rotation_matrix.copy()
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = np.linalg.inv(
                construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        """
        Transform observations from spatial(base) frame into body(end-effector) frame
        using the rotation and homogeneous matrix
        """
        obs["state"]["tcp_vel"][:3] = self.rotation_matrix_reset.transpose() @ obs["state"]["tcp_vel"][:3]
        obs["state"]["tcp_vel"][3:6] = self.rotation_matrix_reset.transpose() @ obs["state"]["tcp_vel"][3:6]
        obs["state"]["tcp_force"] = self.rotation_matrix.transpose() @ obs["state"]["tcp_force"]
        obs["state"]["tcp_torque"] = self.rotation_matrix.transpose() @ obs["state"]["tcp_torque"]

        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_b_r = self.T_r_o_inv @ T_b_o

            # Reconstruct transformed tcp_pose vector
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action: np.ndarray):
        """
        Transform action from body(end-effector) frame into spatial(base) frame
        using the rotation matrix
        """
        action = np.array(action)  # in case action is a jax read-only array
        action[:3] = self.rotation_matrix_reset @ action[:3]
        action[3:6] = self.rotation_matrix_reset @ action[3:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        """
        Transform action from spatial(base) frame into body(end-effector) frame
        using the rotation matrix.
        """
        action = np.array(action)
        action[:3] = self.rotation_matrix_reset.transpose() @ action[:3]
        action[3:6] = self.rotation_matrix_reset.transpose() @ action[3:6]
        return action
