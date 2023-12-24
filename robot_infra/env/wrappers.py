import time
from gym import Env, spaces
import gym
import numpy as np
from gym.spaces import Box
import copy
from robot_infra.spacemouse.spacemouse_teleop import SpaceMouseExpert
from scipy.spatial.transform import Rotation as R


class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def seed(self, _seed):
        return self.wrapped_env.seed(_seed)

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        if attr == 'planner':
            return self._planner
        if attr == 'set_vf':
            return self.set_vf
        return getattr(self._wrapped_env, attr)
        # try:
        #     getattr(self, attr)
        # except Exception:
        #     return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)

class GripperCloseEnv(ProxyEnv):
    def __init__(
            self,
            env,
    ):
        ProxyEnv.__init__(self, env)
        ub = self._wrapped_env.action_space
        assert ub.shape == (7,)
        self.action_space = Box(ub.low[:6], ub.high[:6])
        self.observation_space = spaces.Dict(
            {
                "state_observation": spaces.Dict(
                    {
                        "tcp_pose": spaces.Box(-np.inf, np.inf, shape=(6,)), # xyz + euler
                        "tcp_vel": spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "tcp_force": spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "image_observation": spaces.Dict(
                    {
                    "wrist_1": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                    "wrist_1_full": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                    "wrist_2": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                    "wrist_2_full": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                    }
                ),
            }
        )

    def step(self, action):
        a = np.zeros(self._wrapped_env.action_space.shape)
        a[:6] = copy.deepcopy(action)
        a[6] = 1
        return self._wrapped_env.step(a)

class SpacemouseIntervention(ProxyEnv):
    def __init__(self, env, gripper_enabled=False):
        ProxyEnv.__init__(self, env)
        self._wrapped_env = env
        self.action_space = self._wrapped_env.action_space
        self.gripper_enabled = gripper_enabled
        if self.gripper_enabled:
            assert self.action_space.shape == (7,) # maybe not so elegant
        self.observation_space = self._wrapped_env.observation_space
        self.expert = SpaceMouseExpert(
            xyz_dims=3,
            xyz_remap=[0, 1, 2],
            xyz_scale=200,
            rot_scale=200,
            all_angles=True
        )
        self.last_intervene = 0

    def expert_action(self, action):
        '''
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        '''
        controller_a, _, left, right = self.expert.get_action()
        expert_a = np.zeros((6,))
        if self.gripper_enabled:
            expert_a = np.zeros((7,))
            expert_a[-1] = np.random.uniform(-1, 0)

        expert_a[:3] = controller_a[:3] # XYZ
        expert_a[3] = controller_a[4]  # Roll
        expert_a[4] = controller_a[5] # Pitch
        expert_a[5] = -controller_a[6] # Yaw

        if self.gripper_enabled:
            if left:
                expert_a[6] = np.random.uniform(0, 1)
                self.last_intervene = time.time()

            if np.linalg.norm(expert_a[:6]) > 0.001:
                self.last_intervene = time.time()
        else:
            if np.linalg.norm(expert_a) > 0.001:
                self.last_intervene = time.time()

        if time.time() - self.last_intervene < 0.5:
            return expert_a, left, right
        return action, left, right

    def step(self, action):
        expert_action, left, right = self.expert_action(action)
        o, r, done, truncated, env_info = self._wrapped_env.step(expert_action)
        env_info['expert_action'] = expert_action
        env_info['right'] = right
        return o, r, done, truncated, env_info

class FourDoFWrapper(gym.ActionWrapper):
    def __init__(self, env: Env):
        super().__init__(env)

    def action(self, action):
        a = np.zeros(4)
        a[:3] = action[:3]
        a[-1] = action[-1]
        return a

class TwoCameraFrankaWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        ProxyEnv.__init__(self, env)
        self.env = env
        self.observation_space = spaces.Dict(
            {
                "state": spaces.flatten_space(self.env.observation_space['state_observation']),
                "wrist_1": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                "wrist_2": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
                # "side_1": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
            }
        )

    def observation(self, obs):
        ob = {
            'state': spaces.flatten(self.env.observation_space['state_observation'],
                            obs['state_observation']),
            'wrist_1': obs['image_observation']['wrist_1'][...,::-1], # flip color channel
            'wrist_2': obs['image_observation']['wrist_2'][...,::-1], # flip color channel
            # 'side_1': obs['image_observation']['side_1'][...,::-1], # flip color channel
        }
        return ob

class ResetFreeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.task_id = 0 # 0: place into silver bin, 1: place into brown bin

    def reset(self, task_id=0):
        self.task_id = task_id
        print(f'reset to task {self.task_id}')
        if self.task_id == 0:
            self.resetpos[1] = self.centerpos[1] + 0.1
        else:
            self.resetpos[1] = self.centerpos[1] - 0.1
        return self.env.reset()

class RelativeFrame(gym.Wrapper):
    """
    This wrapper makes the observation and action relative to the end-effector frame.
    """
    def __init__(self, env: Env):
        super().__init__(env)
        self.adjoint_matrix = np.zeros((6, 6))

    def step(self, action):
        # Transform action from end-effector frame to base frame
        transformed_action = self.transform_action(action)

        obs, reward, done, info = self.env.step(transformed_action)
        
        # Update adjoint matrix
        self.adjoint_matrix = self.construct_adjoint_matrix(obs['state_observation']['tcp_pose'])
        
        # Transform observation to spatial frame
        transformed_obs = self.transform_observation(obs)
        return transformed_obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        
        # Update adjoint matrix
        self.adjoint_matrix = self.construct_adjoint_matrix(obs['state_observation']['tcp_pose'])
        
        # Transform observation to spatial frame
        return self.transform_observation(obs)

    def transform_observation(self, obs):
        # Transform observations from spatial(base) frame into body(end-effector) frame using the adjoint matrix
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        R_inv = np.linalg.inv(adjoint_inv[:3, :3])
        obs['state_observation']['tcp_vel'] = adjoint_inv @ obs['state_observation']['tcp_vel']
        obs['state_observation']['tcp_force'] = R_inv @ obs['state_observation']['tcp_force']
        obs['state_observation']['tcp_torque'] = R_inv @ obs['state_observation']['tcp_torque']

        return obs
    
    def transform_action(self, action):
        # Transform action from body(end-effector) frame into into spatial(base) frame using the adjoint matrix
        action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def construct_adjoint_matrix(self, tcp_pose):
        # Construct the adjoint matrix for a spatial velocity vector
        rotation = R.from_quat(tcp_pose[3:]).as_matrix()
        translation = np.array(tcp_pose[:3])
        skew_matrix = np.array([[0, -translation[2], translation[1]],
                                [translation[2], 0, -translation[0]],
                                [-translation[1], translation[0], 0]])
        adjoint_matrix = np.zeros((6, 6))
        adjoint_matrix[:3, :3] = rotation
        adjoint_matrix[3:, 3:] = rotation
        adjoint_matrix[:3, 3:] = skew_matrix @ rotation
        return adjoint_matrix
