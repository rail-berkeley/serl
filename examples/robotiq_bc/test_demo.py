import numpy as np
import threading
import pickle as pkl
import gymnasium as gym
from pynput import keyboard

from robotiq_env.envs.wrappers import SpacemouseIntervention, Quat2EulerWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SerlObsWrapperNoImages

exit_program = threading.Event()

"""
Script to test the demos recorded --> they can be repeated easily (if the box is in the same position)
"""


def on_space(key, info_dict):
    if key == keyboard.Key.space:
        for key, item in info_dict.items():
            print(f'{key}:  {item}', end='   ')
        print()


def on_esc(key):
    if key == keyboard.Key.esc:
        exit_program.set()


if __name__ == "__main__":
    env = gym.make("robotiq-grip-v1")
    env = SpacemouseIntervention(env)
    # env = RelativeFrame(env)
    env = Quat2EulerWrapper(env)
    env = SerlObsWrapperNoImages(env)
    # env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    obs, _ = env.reset()

    info_dict = {'state': env.unwrapped.curr_pos, 'gripper_state': env.unwrapped.gripper_state,
                 'force': env.unwrapped.curr_force}
    listener_1 = keyboard.Listener(daemon=True, on_press=lambda event: on_space(event, info_dict=info_dict))
    listener_1.start()

    listener_2 = keyboard.Listener(on_press=on_esc, daemon=True)
    listener_2.start()

    file_path = "robotiq_test_20_demos_2024-03-26_12-23-50.pkl"

    with open(file_path, "rb") as f:
        transitions = pkl.load(f)

    try:
        for transition in transitions:
            if exit_program.is_set():
                raise KeyboardInterrupt  # stop program, but clean up before

            action = transition["actions"]
            next_obs, rew, done, truncated, info = env.step(action=action)

            print(next_obs - transition["next_observations"])

            if transition["dones"] or done:
                print(f"done with {transition['dones']}  {done}")
                env.reset()

    except KeyboardInterrupt:
        print(f'\nProgram was interrupted, cleaning up...')

    finally:
        env.close()
        listener_1.stop()
        listener_2.stop()
