from typing import Callable, Union

import gymnasium as gym
import numpy as np

try:
    import roboverse
except ImportError:
    pass
import mujoco_manipulation
import tensorflow as tf

from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.dmcgym import DMCGYM
from serl_launcher.wrappers.mujoco import GCMujocoWrapper
from serl_launcher.wrappers.norm import UnnormalizeActionProprio
from serl_launcher.wrappers.roboverse import GCRoboverseWrapper
from serl_launcher.wrappers.video_recorder import VideoRecorder


def make_mujoco_gc_env(
    env_name: str,
    max_episode_steps: int,
    action_proprio_metadata: dict,
    save_video: bool,
    save_video_dir: str,
    save_video_prefix: str,
    goals: Union[np.ndarray, Callable],
    obs_horizon: int,
    act_exec_horizon: int,
):
    env = mujoco_manipulation.load(env_name)
    env = DMCGYM(env)
    env = GCMujocoWrapper(env, goals)
    env = UnnormalizeActionProprio(env, action_proprio_metadata)
    if obs_horizon is not None or act_exec_horizon is not None:
        env = ChunkingWrapper(env, obs_horizon, act_exec_horizon)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if save_video:
        env = VideoRecorder(
            env,
            save_folder=save_video_dir,
            save_prefix=save_video_prefix,
            goal_conditioned=True,
        )

    env.reset()

    return env


def wrap_roboverse_gc_env(
    env: gym.Env,
    max_episode_steps: int,
    action_proprio_metadata: dict,
    goal_sampler: Union[np.ndarray, Callable],
):
    env = GCRoboverseWrapper(env, goal_sampler)
    env = UnnormalizeActionProprio(env, action_proprio_metadata)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def make_roboverse_gc_env(
    env_name: str,
    max_episode_steps: int,
    action_proprio_metadata: dict,
    save_video: bool,
    save_video_dir: str,
    save_video_prefix: str,
    goals: Union[np.ndarray, Callable],
):
    env = roboverse.make(env_name, transpose_image=False)
    env = wrap_roboverse_gc_env(env, max_episode_steps, action_proprio_metadata, goals)

    if save_video:
        env = VideoRecorder(
            env,
            save_folder=save_video_dir,
            save_prefix=save_video_prefix,
            goal_conditioned=True,
        )

    env.reset()

    return env


PROTO_TYPE_SPEC = {
    "observations/images0": tf.uint8,
    "observations/state": tf.float32,
    "next_observations/images0": tf.uint8,
    "next_observations/state": tf.float32,
    "actions": tf.float32,
    "terminals": tf.bool,
    "truncates": tf.bool,
    "info/place_success": tf.bool,
    "info/target_object": tf.uint8,
    "info/object_positions": tf.float32,
    "info/target_position": tf.float32,
    "info/object_names": tf.string,
}


def _decode_example(example_proto):
    # decode the example proto according to PROTO_TYPE_SPEC
    features = {
        key: tf.io.FixedLenFeature([], tf.string) for key in PROTO_TYPE_SPEC.keys()
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_tensors = {
        key: tf.io.parse_tensor(parsed_features[key], dtype)
        for key, dtype in PROTO_TYPE_SPEC.items()
    }

    return {
        "observations": {
            "image": parsed_tensors["observations/images0"],
            "proprio": parsed_tensors["observations/state"],
        },
        "next_observations": {
            "image": parsed_tensors["next_observations/images0"],
            "proprio": parsed_tensors["next_observations/state"],
        },
        "actions": parsed_tensors["actions"],
        "terminals": parsed_tensors["terminals"],
        "truncates": parsed_tensors["truncates"],
        "infos": {
            "place_success": parsed_tensors["info/place_success"],
            "object_positions": parsed_tensors["info/object_positions"],
            "target_position": parsed_tensors["info/target_position"],
            "target_object": parsed_tensors["info/target_object"],
            "object_names": parsed_tensors["info/object_names"],
        },
    }


def load_tf_dataset(data_path):
    """Load a sim dataset in TFRecord format."""
    data_paths = tf.io.gfile.glob(tf.io.gfile.join(data_path, "*.tfrecord"))

    # shuffle again using the dataset API so the files are read in a
    # different order every epoch
    dataset = tf.data.Dataset.from_tensor_slices(data_paths)

    # yields raw serialized examples
    dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

    # yields trajectories
    dataset = dataset.map(_decode_example, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset
