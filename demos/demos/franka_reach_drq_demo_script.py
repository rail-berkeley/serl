#!/usr/bin/env python3
import os
import time
from datetime import datetime

import numpy as np

# Logging
from absl import app, flags, logging
from oxe_envlogger.envlogger import AutoOXEEnvLogger

# DRL
import gym
import mujoco
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

# Needed to create the franka environment
import franka_sim

# Teleoperation imports
import sys, select, termios, tty

# RLDS/TFDS
import json, glob, inspect
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets import folder_dataset
#-------------------------------------------------------------------------------------------
# Flags
#-------------------------------------------------------------------------------------------
flags.DEFINE_string("env", "PandaReachSparseCube-v0", "Name of environment.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 200, "Maximum length of trajectory.")
flags.DEFINE_boolean("debug", True, "Debug mode.")  # debug mode will disable wandb logging
#flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")
flags.DEFINE_string("output_dir", "/data/data/serl/demos/franka_reach_drq_demo_script",
                    "Directory to save the output data. This is where the RLDS logs will be saved.")                     
flags.DEFINE_integer("num_demos", 2, "Number of episodes to log.")
flags.DEFINE_boolean("enable_envlogger", True, "Enable envlogger.")
flags.DEFINE_string("teleop_mode", "keyboard", "Teleoperation mode: 'keyboard' or 'spacemouse'.")

FLAGS = flags.FLAGS

#-------------------------------------------------------------------------------------------
## Telop Config Variables
#-------------------------------------------------------------------------------------------
ACTION_MAX = 10 # Maximum action value for clipping actions

# Bind, xyz, gripper vals to keys
moveBindings = {
    'i':(1,0,0,0),
    ',':(-1,0,0,0),
    'j':(0,1,0,0),
    'l':(0,-1,0,0),
    'u':(0,0,0,1),
    'o':(0,0,0,-1),
    'm':(0,0,1,0),
    '.':(0,0,-1,0),
        }

# Extend bindings to include camera controls
camBindings = {
    'a': ("azimuth", -5),   # rotate left
    'd': ("azimuth", 5),    # rotate right
    'w': ("elevation", 2),  # tilt up
    's': ("elevation", -2), # tilt down
    'q': ("distance", -0.1),# zoom in
    'e': ("distance", 0.1), # zoom out
}

def update_camera(viewer,key):
    """
    Update the camera view based on keyboard input. Assumes higher level function has checked for the existance of key in camBindings.
    Controls:
        'a' : rotate left
        'd' : rotate right
        'w' : tilt up
        's' : tilt down
        'q' : zoom in
        'e' : zoom out
    """
    if hasattr(viewer, 'cam'):
        # Get current camera parameters
        attr, delta = camBindings[key]            
        val = getattr(viewer.cam, attr)

        setattr(viewer.cam, attr, val + delta)   

def close_logger_and_env(env):
    """
    Best-effort shutdown:
      1) close embedded envloggers/writers if present
      2) close dm_env (if you have a DeepMind-style env)
      3) close Gym/Gymnasium env
      4) close the Mujoco viewer
    Also walks through common wrapper attributes (env, _env, unwrapped).
    """
    import logging

    visited = set()

    def _safe_close(obj, label=""):
        if obj is None or id(obj) in visited:
            return
        visited.add(id(obj))

        # 1) Close common logger/writer attributes first (flush TFRecord)
        for name in ("_envlogger", "envlogger", "logger", "_logger", "writer"):
            try:
                logger_obj = getattr(obj, name, None)
                if logger_obj is not None and hasattr(logger_obj, "close"):
                    logger_obj.close()
            except Exception as e:
                logging.warning(f"close_logger_and_env: {label}.{name}.close() raised {e!r}")

        # 2) Close dm_env if present
        try:
            dm = getattr(obj, "dm_env", None)
            if dm is not None and hasattr(dm, "close"):
                dm.close()
        except Exception as e:
            logging.warning(f"close_logger_and_env: {label}.dm_env.close() raised {e!r}")

        # 3) Close Gym/Gymnasium env
        try:
            if hasattr(obj, "close"):
                obj.close()
        except Exception as e:
            logging.warning(f"close_logger_and_env: {label}.close() raised {e!r}")

        # 4) Close Mujoco viewer if accessible
        try:
            # Some stacks keep viewer at env._viewer.viewer; others just env._viewer
            viewer = None
            vwrap = getattr(obj, "_viewer", None)
            if vwrap is not None:
                viewer = getattr(vwrap, "viewer", vwrap)
            if viewer is not None and hasattr(viewer, "close"):
                viewer.close()
        except Exception as e:
            logging.warning(f"close_logger_and_env: {label} viewer close raised {e!r}")

        # Recurse into common wrapper links
        for child_name in ("env", "_env", "unwrapped", "environment", "base_env"):
            child = getattr(obj, child_name, None)
            if child is not None and child is not obj:
                _safe_close(child, f"{label}.{child_name}" if label else child_name)

    _safe_close(env, "env")

def finalize_tfds_metadata_beamless(builder_dir: str):
    """
    Beam-free finalize: count TFRecord examples per shard and write
    numShards/shardLengths into dataset_info.json so TFDS will load.
    """
    import os, json, glob
    import tensorflow as tf

    info_path = os.path.join(builder_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Missing dataset_info.json in {builder_dir}")

    with open(info_path) as f:
        info = json.load(f)

    ds_name    = info["name"]                    # e.g. "PandaReachSparseCube-v0"
    file_fmt   = info.get("fileFormat", "tfrecord")
    tmpl_str   = info["splits"][0]["filepathTemplate"]

    # Prefer strict pattern "<name>-<split>.<fmt>-<shard>"
    shard_paths = sorted(glob.glob(os.path.join(builder_dir, f"{ds_name}-*.{file_fmt}-*")))
    if not shard_paths:
        # Fallback to any tfrecord-like file
        shard_paths = sorted(glob.glob(os.path.join(builder_dir, f"*.{file_fmt}*")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No {file_fmt} shards found in {builder_dir}. "
            f"Expected like '{ds_name}-train.{file_fmt}-00000' per template '{tmpl_str}'."
        )

    # Count episodes (1 Example = 1 episode with envlogger/RLDS)
    shard_lengths = [sum(1 for _ in tf.data.TFRecordDataset(p)) for p in shard_paths]

    # Write lengths for each split using this template
    for s in info["splits"]:
        if s.get("filepathTemplate") == tmpl_str:
            s["numShards"]    = len(shard_paths)
            s["shardLengths"] = shard_lengths
    # Re-write dataset_info.json with updated shard info
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Sanity log
    import tensorflow_datasets as tfds
    b = tfds.builder_from_directory(builder_dir)
    print("[finalize] splits:", {k: v.num_examples for k, v in b.info.splits.items()})

def ensure_dir_exists():
    """
    For oxe_envlogger + RLDS compatibility, data must be written in the following format:
    /data/data/serl/demos/franka_reach_drq_demo_script/
    └── session_20250821_222412/
        └── PandaReachSparseCube-v0/
            └── 0.1.0/
                dataset_info.json
                features.json
                PandaReachSparseCube-v0-train.tfrecord-00000-of-00001

    We can have a base path with customized sessions inside. 
    Inside each session we have: env-version-files


    Returns
    -------
    out_path : str
        The path to the output directory.
    """
    # Customize the path
    root = FLAGS.output_dir 
    session = datetime.now().strftime("session_%Y%m%d_%H%M%S")
    session_root = os.path.join(root, f"{FLAGS.num_demos}_demos_{session}")
    #session_root = os.path.join(root, f"{session}_num_demos_{FLAGS.num_demos}")

    # Dataset details
    dataset_name = FLAGS.env
    version = "0.1.0"  # PArt of RLDS format. needed.

    # Create output filename with configuration details
    dataset_dir = os.path.join(session_root, dataset_name, version)
    os.makedirs(dataset_dir, exist_ok=True)
    logging.info(f"TFDS builder dir: {dataset_dir}")
    
    return dataset_dir

def getKey(settings):
    """
    Waits briefly for a keypress and returns the pressed key.

    Parameters
    ----------
    settings : list
        Original terminal settings so they can be restored after reading.

    Returns
    -------
    key : str
        The key pressed by the user, or '' (empty string) if no key was pressed.
    """
    # Put terminal into raw mode so keypress is captured instantly
    tty.setraw(sys.stdin.fileno())

    # Wait for human to input action, we will not provide a timeout so it is blocking.
    rlist, _, _ = select.select([sys.stdin], [], []) # select.select(rlist, wlist, xlist[, timeout])

    if rlist:
        # Read exactly one character if a key is pressed
        key = sys.stdin.read(1)
    else:
        key = ''

    # Restore terminal to original settings
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def get_kb_demo_action(env,speed=0.075):
    """
    Reads keyboard input and maps it to a 3D action vector for robot control or camera action. 
    TODO: currently can only read one key at a time. Needs to be extended to read multiple keys to handle both.
    Otherwise, none actions are still considered steps in the loop. 

    The function uses non-blocking keyboard input to allow interactive
    teleoperation. Keys are mapped to directions in Cartesian space:
      - 'i' : +x (forward)
      - ',' : -x (backward)
      - 'j' : +y (left)
      - 'l' : -y (right)
      - 'm' : +z (up)
      - '.' : -z (down)
      - 'k' : stop (zero vector)

    Parameters
    ----------
    speed : float, optional
        Step size for each key press (default 0.2).

    Returns
    -------
    np.ndarray
        Action vector of shape (3,), where each entry corresponds to
        [x, y, z] translation command. Example: [0.2, 0.0, 0.0].
    """
    # Save current terminal settings so we can restore later
    settings = termios.tcgetattr(sys.stdin)

    # Initialize action as a zero vector (no movement)
    action = np.zeros(4, dtype=float)

    try:
        # Capture the pressed key 
        key = getKey(settings)

        # Check keys for camera first, if so, update camera and get another key for action
        if key in camBindings:               
            if hasattr(env.unwrapped, "_viewer"):
                update_camera(env.unwrapped._viewer.viewer,key) 
                key = getKey(settings) # get another key for action

        elif key in moveBindings:
            # Lookup (x, y, z) direction and scale by speed
            dx, dy, dz, g= moveBindings[key]
            action = np.array([dx, dy, dz, g], dtype=float) * speed

        elif key == 'k':
            # 'k' means stop → zero vector
            action = np.zeros(4, dtype=float)

        elif key == '\x03':  # CTRL-C
            raise KeyboardInterrupt
        
    finally:
        # Restore terminal even if something goes wrong
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

    # Clip action values to prevent excessive commands
    action = np.clip(action, -ACTION_MAX, ACTION_MAX)

    return action

def set_front_cam_view(env):
    """
    Set the camera view to a front-facing perspective for better visualization.
    
    Args:
        env: The environment instance containing the viewer.
    
    Returns:
        viewer: The viewer with updated camera settings.
    """
    viewer = env.unwrapped._viewer.viewer  # Access the viewer from the environment
    
    if hasattr(viewer, 'cam'):
        viewer.cam.lookat[:] = [0, 0, 0.1]   # Center of robot (adjust as needed)
        viewer.cam.distance = 2.0            # Camera distance
        viewer.cam.azimuth = 155             # 0 = right, 90 = front, 180 = left
        viewer.cam.elevation = -30           # Negative = above, positive = below

        # Hide menu
        viewer._hide_overlay = True
    
    return viewer

##############################################################################
def main(unused_argv):
    logging.info(f'Creating gym environment...')

    # Render mode configuration based on debug flag
    if FLAGS.debug:
        _render_mode = 'human'  # Render mode for the environment, can be 'human' or 'rgb_array'
    else:
        _render_mode = 'rgb_array'  # Use 'rgb_array' for automated testing without GUI

    # Create the environment with the specified render mode and wrappers
    env = gym.make(FLAGS.env, render_mode=_render_mode)

    if FLAGS.env == "PandaPickCube-v0":
        env = gym.wrappers.FlattenObservation(env)

    if FLAGS.env == "PandaReachSparseCube-v0":
        env = SERLObsWrapper(
            env,
            target_hw=(128, 128),
            img_dtype=np.uint8,   # or np.float32
            normalize=False,      # True if using float32 in [0,1]
        )
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    logging.info(f'Done creating {FLAGS.env} environment.')

    # Set camera to front view if viewer is available
    if hasattr(env.unwrapped, '_viewer'):
        viewer = set_front_cam_view(env)
        if viewer:
            logging.info('Camera view set to front-facing perspective.')
        else:
            logging.warning('Failed to set camera view. Viewer not available.')

    # Wrap with oxe_envlogger to record demos
    dataset_dir = None
    session_root = None
    if FLAGS.enable_envlogger:

        dataset_dir = ensure_dir_exists()

        # Will save as many episodes as possible into files of 200MB each by default.
        env = AutoOXEEnvLogger(
            env=env,
            dataset_name=FLAGS.env,
            directory=dataset_dir,
            #split_name="train", # "train", "test", or "validation"
        )
    logging.info('Recording %r demos...', FLAGS.num_demos)

    #--- LOOP DEMOS/EPISODES ---
    # Loop through the number of demos specified by the user to record demonstrations
    try:
        for i in range(FLAGS.num_demos):

            # Log custom metadata during new episode: language embeddings randomly.
            if FLAGS.enable_envlogger:
                # The "language_embedding" is a standard field used in robotics datasets (like the OXE format that envlogger creates) to store a numerical representation of a natural language instruction for an episode.
                # How to Reconcile it with 5 Random Numbers: The five random numbers are just placeholder data. This script is a demonstration and doesn't involve a real language model.
                env.set_episode_metadata({
                    "language_embedding": np.random.random((5,)).astype(np.float32)
                })
                env.set_step_metadata({"timestamp": time.time()})

            logging.info('episode %r', i)
            
            # Start a new episode
            env.reset()
            terminated = False
            truncated = False

            step = 0

            # Termination occurs when the hand reaches the target or the maximum trajectory length is reached
            while not (terminated or truncated):          
                
                # Get action from the demo function
                action = get_kb_demo_action(env)            

                # example to log custom step metadata
                if FLAGS.enable_envlogger:
                    env.set_step_metadata({"timestamp": np.float32(time.time())})

                return_step = env.step(action)

                # NOTE: to handle gym.Env.step() return value change in gym 0.26
                if len(return_step) == 5:
                    obs, reward, terminated, truncated, info = return_step
                else:
                    obs, reward, terminated, info = return_step
                    truncated = False

                print(f" step: {step}", f"reward: {reward:.3f}\n")
                step += 1

        logging.info('Done recording %r demos.', FLAGS.num_demos)
    
    finally:
        # Finalize TFDS metadata so SERL/TFDS can load the split
        if FLAGS.enable_envlogger and dataset_dir is not None:

            # Close the environment to flush/write data. AutoOXEEnvLogger implements dm_env.close() 
            # which flushes data to disk. This is important to ensure all data is written before finalizing metadata.
            # If you skip this step, some data may not be written and the dataset may be incomplete.
            logging.info("Closing environment to flush data to disk...")

            # closes logger(s) + dm_env + gym + viewer
            # env.unwrapped.unwrapped.unwrapped.close() # close mujoco viewer
            # env.env.env.env.close()            
            close_logger_and_env(env)                      

            # Scan files and write metadata
            finalize_tfds_metadata_beamless(dataset_dir)

    # Note: async_drq_sim will read from the dataset_dir you printed above.

if __name__ == '__main__':
    app.run(main)