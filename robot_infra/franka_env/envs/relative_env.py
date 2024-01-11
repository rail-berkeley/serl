from scipy.spatial.transform import Rotation as R
import gym
import numpy as np
from gym import Env


class RelativeFrame(gym.Wrapper):
    """
    This wrapper makes the observation and action relative to the end-effector frame.
    This wrapper is expected to be used on top of the base Franka environment, which has the following
    observation space:
    {
        "state": spaces.Dict(
            {
                "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(7,)), # xyz + quat
                ......
            }
        ),
        ......
    }, and at least 6 DoF action space with (x, y, z, rx, ry, rz, ...)
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        self.adjoint_matrix = np.zeros((6, 6))

        self.include_relative_pose = include_relative_pose
        if self.include_relative_pose:
            # Transformation matrix from reset pose's relative frame to base frame
            self.T_r_o_inv = np.zeros((4, 4))

    def step(self, action):
        # action is assumed to be (x, y, z, rx, ry, rz, gripper)
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)

        obs, reward, done, truncated, info = self.env.step(transformed_action)

        # this is to convert the spacemouse intervention action
        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action(info["intervene_action"])

        # Update adjoint matrix
        self.adjoint_matrix = self.construct_adjoint_matrix(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Update adjoint matrix
        self.adjoint_matrix = self.construct_adjoint_matrix(obs["state"]["tcp_pose"])
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_r_o_inv = self.construct_T_matrix_inv(obs["state"]["tcp_pose"])

        # Transform observation to spatial frame
        return self.transform_observation(obs), info

    def transform_observation(self, obs):
        # Transform observations from spatial(base) frame into body(end-effector) frame using the adjoint matrix
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        R_inv = adjoint_inv[:3, :3]
        obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]
        # obs['state']['tcp_force'] = R_inv @ obs['state']['tcp_force']
        # obs['state']['tcp_torque'] = R_inv @ obs['state']['tcp_torque']

        # let the current pose and rotation in base frame be p_b_o and theta_b_o
        if self.include_relative_pose:
            T_b_o = self.construct_T_matrix(obs["state"]["tcp_pose"])
            # Transformation matrix from current pose to reset pose's relative frame
            T_b_r = self.T_r_o_inv @ T_b_o
            p_b_r = T_b_r[:3, 3]
            theta_b_r = R.from_matrix(T_b_r[:3, :3]).as_quat()
            # xyz + quat in relative frame
            obs["state"]["tcp_pose"] = np.concatenate((p_b_r, theta_b_r))

        return obs

    def transform_action(self, action):
        # Transform action from body(end-effector) frame into into spatial(base) frame using the adjoint matrix
        action = np.array(action)  # in case action is a jax read-only array
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def construct_adjoint_matrix(self, tcp_pose):
        # Construct the adjoint matrix for a spatial velocity vector
        rotation = R.from_quat(tcp_pose[3:]).as_matrix()
        translation = np.array(tcp_pose[:3])
        skew_matrix = np.array(
            [
                [0, -translation[2], translation[1]],
                [translation[2], 0, -translation[0]],
                [-translation[1], translation[0], 0],
            ]
        )
        adjoint_matrix = np.zeros((6, 6))
        adjoint_matrix[:3, :3] = rotation
        adjoint_matrix[3:, 3:] = rotation
        adjoint_matrix[:3, 3:] = skew_matrix @ rotation
        return adjoint_matrix

    def construct_T_matrix(self, tcp_pose):
        # Construct the transformation matrix from relative frame to base frame
        rotation = R.from_quat(tcp_pose[3:]).as_matrix()
        translation = np.array(tcp_pose[:3])
        T = np.zeros((4, 4))
        T[:3, :3] = rotation
        T[:3, 3] = translation
        T[3, 3] = 1
        return T

    def construct_T_matrix_inv(self, tcp_pose):
        # Construct the inverse of the transformation matrix from relative frame to base frame
        rotation = R.from_quat(tcp_pose[3:]).as_matrix()
        translation = np.array(tcp_pose[:3])
        T = np.zeros((4, 4))
        T[:3, :3] = rotation.T
        T[:3, 3] = -rotation.T @ translation
        T[3, 3] = 1
        return T
