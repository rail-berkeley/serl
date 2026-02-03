"""
Scripted Controller for Franka FR3 Robot - Demonstration Data Generation

This script implements a scripted controller for generating expert demonstration data 
for Deep Reinforcement Learning (DRL) algorithms using the Franka FR3 robot in a 
pick-and-place task. The generated data serves as bootstrapping demonstrations for 
training RL agents with your desired algorithm.

Note that different error thresholds lead to different rewards and in turn done values. 
Adjust carefully. Depending on the the controller and clipping scaling (ACTION_MAX) settings, you may get very different behaviors. 
The current program clips actions and leads to small increments that allows to more precise movements and close the error. 

The controller uses a 4-phase hierarchical approach:
1. Approach Object (move gripper above object)
2. Grasp Object (move to object and close gripper)  
3. Transport to Goal (move grasped object to target)
4. Maintain Position (hold final position)

Output: Compressed NPZ file containing action, observation, and info sequences

TODO: Convert to a class-based structure for better modularity and reusability.
"""
import os
import numpy as np
import gym
from time import sleep, perf_counter
from datetime import datetime

import franka_sim 
import franka_sim.envs.panda_reach_gym_env as panda_reach_env

# Global variables to store episode data across all iterations
observations = []   # List storing observation sequences for each episode  
actions = []        # List storing action sequences for each episode
rewards = []        # List storing reward sequences for each episode
infos = []          # List storing info dictionaries for each episode
terminateds = []    # List storing terminated flags for each episode
truncateds = []     # List storing truncated flags for each episode
dones = []          # List storing done flags (terminated or truncated) for each episode
transition_ctr = 0  # Global counter for transitions across all episodes

#-------------------------------------------------------------------------------------------
## Key Config Variables
#-------------------------------------------------------------------------------------------
# Proportional and derivative control gain for action scaling -- empirically tuned
Kp = 10.0      # Values between 20 and 24 seem to be somewhat stable for Kv = 24
Kv = 10.0 

ACTION_MAX = 10 # Maximum action value for clipping actions
ERROR_THRESHOLD = 0.008 # Note!! When this number is changed, the way rewards are computed in the PandaReachCubeEnv.step() L220 must also be changed such that done=True only at the end of a successfull run.

# Number of demonstration episodes to generate
NUM_DEMOS = 20

# Robot configuration
robot = 'franka'    # Robot type used in the environment, can be 'franka' or 'fetch'
task  = 'reach'     # Task type used in the environment, can be 'reach' or 'pick-and-place'

# Debug mode for rendering and visualization
DEBUG = False

if DEBUG:
    _render_mode = 'human'  # Render mode for the environment, can be 'human' or 'rgb_array'
else:
    _render_mode = 'rgb_array'  # Use 'rgb_array' for automated testing without GUI

# Indices for franka_sim reach environment observations
if robot == 'franka' and task == 'reach':

    opi = np.array([0, 3])  # Indices for object position in observation
    gpi = np.array([3])     # Indices for gripper position in observation
    rpi = np.array([4, 7])   # Indices for robot position in observation
    rvi = np.array([7, 10])   # Indices for robot velocity in observation

# Weld constraint flag
weld_flag = True    # Flag to activate weld constraint during pick-and-place

#-------------------------------------------------------------------------------------------
# Franka sim environments do not have weld constraints like the franka_mujoco environments.
# def activate_weld(env, constraint_name="grasp_weld"):
#     """
#     Activate a weld constraint during pick portion of a demo
#     :param env: The environment containing the model
#     :param constraint_name: The name of the weld constraint to activate
#     :return: True if the weld was successfully activated, False if the constraint was not found
#     """

#     try:
#         # Activate the weld constraint
#         env.unwrapped.model.eq(constraint_name).active = 1     
#         print("Activated weld")   
#         return True
    
#     except KeyError:
#         print(f"Warning: Constraint '{constraint_name}' not found")
#         return False

# def deactivate_weld(env, constraint_name="grasp_weld"):
#     """
#     Deactivate a weld constraint during place portion of a demo
#     :param env: The environment containing the model
#     :param constraint_name: The name of the weld constraint to deactivate
#     :return: True if the weld was successfully deactivated, False if the constraint was not
#     found
#     """ 
    
#     try:
#         # Deactivate the weld constraint
#         env.unwrapped.model.eq(constraint_name).active = 0    
#         print("Deactivated weld")    
#         return True
    
#     except KeyError:
#         print(f"Warning: Constraint '{constraint_name}' not found")
#         return False

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
        viewer.cam.distance = 3.0            # Camera distance
        viewer.cam.azimuth = 135             # 0 = right, 90 = front, 180 = left
        viewer.cam.elevation = -30           # Negative = above, positive = below

        # Hide menu
        viewer._hide_overlay = True
    
    return viewer

def store_transition_data(episode_dict, new_obs, rewards, action, info, terminated, truncated, done):
    """
    Store transition data in the episode dictionary and update global counter.
    """
    global transition_ctr
    transition_ctr += 1
    
    episode_dict["observations"].append(new_obs)
    episode_dict["rewards"].append(rewards)
    episode_dict["actions"].append(action)
    episode_dict["infos"].append(info)
    episode_dict["terminateds"].append(terminated)
    episode_dict["truncateds"].append(truncated)
    episode_dict["dones"].append(done) 

def store_episode_data(episode_data):
    """
    Store complete episode data in global lists only if we succeeded (avoid bad demos).
    """
    actions.append(episode_data["actions"])
    observations.append(episode_data["observations"])
    infos.append(episode_data["infos"])
    rewards.append(episode_data["rewards"])

    # Optionally, also store the done/terminated/truncated flags globally if needed:
    terminateds.append(episode_data["terminateds"])
    truncateds.append(episode_data["truncateds"])
    dones.append(episode_data["dones"])    

def update_state_info(episode_data, time_step, dt, error, reward):
    """
    Update and return the current state information. Always get the latest entry with [-1]
    
    Args:
        new_obs (dict): New observation dictionary containing the current state.
        time_step (int): Current time step in the episode.
        dt (float): Current time step in the episode.
        error (np.ndarray): Current error vector between object and end-effector positions. 
        reward (float): Current reward value for the action taken.

    Returns:
        object_pos (np.ndarray): Current position of the object in the environment.
        gripper_pos (float): Current position of the gripper.
        current_pos (np.ndarray): Current position of the end-effector.
        current_vel (np.ndarray): Current velocity of the end-effector.
    """
    object_pos  = episode_data["observations"][-1][ opi[0]:opi[1] ]   # Block position
    gripper_pos = episode_data["observations"][-1][ gpi[0]            ]   # Gripper position
    current_pos = episode_data["observations"][-1][ rpi[0]:rpi[1]   ]   # Panda/tcp position
    current_vel = episode_data["observations"][-1][ rvi[0]:rvi[1]   ]  # Panda/t


    # Print debug information
    print(
        f"Step: {time_step}, ErrNorm: {np.linalg.norm(error):.4f}, "
        f"bot_pos: {np.array2string(current_pos, precision=3)}, "
        f"obj_pos: {np.array2string(object_pos, precision=3)}, "
        f"fgr_pos: {np.array2string(gripper_pos, precision=2)}, "
        f"err:     {np.array2string(error, precision=3)}, "
        f"Action:  {np.array2string(episode_data['actions'][-1], precision=3)}, "
        f"dt: {dt:.4f}, reward: {reward:.3f}"
    )
    
    
    return object_pos, gripper_pos, current_pos, current_vel
    
def compute_error(object_pos, current_pos, prev_error, dt):
    """Compute the error and its derivative between the object position and the current end-effector position.
    Args:
        object_pos (np.ndarray): The position of the object in the environment.
        current_pos (np.ndarray): The current position of the end-effector.
        prev_error (np.ndarray): The previous error value for derivative calculation.
        dt (float): Time step for derivative calculation.
        
    Returns:
        error (np.ndarray): The current error vector between the object and end-effector positions.
        derror (np.ndarray): The derivative of the error vector.
    """
    error = object_pos - current_pos  # Calculate the error vector
    derror = (error - prev_error) / dt  # Calculate the derivative of the error vector 

    prev_error = error.copy()  # Update previous error for next iteration
    return error, derror

def demo(env, lastObs):
    """
    Executes a scripted reach sequence using a hierarchical approach.
    
    Implements 1-phase control strategy:
    1. Approach: Move gripper above object (3cm offset)

    Store observations, actions, and info in global lists for later replay buffer inclusion.

    Gripper:
    - The gripper in Mujoco ranges from a value of 0 to 0.4, where 0 is fully open and 0.4 is fully closed.
    
    Args:
        env: Gymnasium environment instance
        lastObs: Flattened observations set as object_pos, gripper_pos, panda/tcp_pos, panda/tcp_vel
            - observations:
                object_pos[0:3],
                gripper_pos[3]
                panda/tcp_pos[4:7],
                panda/tcp_vel[7:10]

    Returns:
    """

    ## Init goal, current_pos, and object position from last observation
    object_pos       = np.zeros(3, dtype=np.float32)
    current_pos      = np.zeros(3, dtype=np.float32)
    gripper_pos      = np.zeros(1, dtype=np.float32)
    object_rel_pos   = np.zeros(3, dtype=np.float32)


    # Initialize (single) episode data collection
    episodeObs  = []        # Observations for this episode  
    episodeAcs  = []        # Actions for this episode
    episodeRews = []        # Rewards for this episode
    episodeInfo = []        # Info for this episode
    episodeTerminated = []  # Terminated flags for this episode
    episodeTruncated = []   # Truncated flags for this episode
    episodeDones = []       # Done flags (terminated or truncated) for this episode
    
    # Dictionary to store episode data
    episode_data = {
        "observations": episodeObs,
        "actions": episodeAcs,
        "rewards": episodeRews,
        "infos": episodeInfo,
        "terminateds": episodeTerminated,
        "truncateds": episodeTruncated,
        "dones": episodeDones
    }   

    # close gripper
    fgr_pos = 0

    # Error thresholds
    error_threshold = ERROR_THRESHOLD  # Threshold for stopping condition (Xmm)

    finger_delta_fast = 0.05    # Action delta for fingers 5cm per step (will get clipped by controller)... more of a scalar. 
    finger_delta_slow = 0.005   # Franka has a range from 0 to 4cm per finger

    ## Extract data   
    object_pos = lastObs[opi[0]:opi[1]]  # block pos
    current_pos = lastObs[rpi[0]:rpi[1]] # panda/tcp_pos

    # Relative position between end-effector and object
    dt = env.unwrapped.model.opt.timestep  # Mujoco time step
    prev_error = np.zeros_like(object_pos)
    error, derror = compute_error(object_pos, current_pos, prev_error, dt)    

    time_step = 0  # Track total time_steps in episode
    episodeObs.append(lastObs) # Store initial observation

    # Initialize previous time for dt calculation
    prev_time = perf_counter()  # Start time for dt calculation
    
    # Phase 1: Reach
    # Terminate when distance to above-object position < error_threshold
    print(f"----------------------------------------------- Phase 1: Reach -----------------------------------------------")
    while np.linalg.norm(error) >= error_threshold and time_step <= env.spec.max_episode_steps:
        env.render()  # Visual feedback
        
        # Record current time and compute dt
        curr_time = perf_counter()
        dt = curr_time - prev_time
        prev_time = curr_time

        # Initialize action vector [x, y, z]
        action = np.array([0., 0., 0.])
        
        # Proportional control with gain of 6
        action[:3] = error * Kp + derror * Kv
        prev_error = error.copy()  # Update previous error for next iteration
        
        # Clip action to prevent excessive movements
        action = np.clip(action/ACTION_MAX, -0.1, 0.1)  #

        # Keep gripper closed -- no need. only 3 dimensions of control
        #action[ len(action)-1 ] = -finger_delta_fast # Maintain gripper closed.
        
        # Unpack new Gymnasium step API
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store episode data
        store_transition_data(episode_data, new_obs, reward, action, info, terminated, truncated, done)

        # Update and print state information
        object_pos,gripper_pos,cur_pos,cur_vel = update_state_info(episode_data, time_step, dt, error,reward)

        # Update error for next iteration
        error, derror = compute_error(object_pos, cur_pos, prev_error, dt)

       # Update time step
        time_step += 1

        # Sleep
        #if DEBUG:
        sleep(0.25)  # Activated when DEBUG is True for better visualization.        

    # Store complete episode data in global lists only if we succeeded (avoid bad demos)
    store_episode_data(episode_data)

    # Deactivate weld constraint after successful pick -- franka_sim env does not have weld like franka_mujoco env.
    # if weld_flag:
    #     deactivate_weld(env, constraint_name="grasp_weld")

    # Break out of the loop to start a new episode
    return True
        
    # # If we reach here, the episode was not successful
    # if weld_flag:
    #     print("Failed to transport object to goal position. Deactivating weld.")
    #     deactivate_weld(env, constraint_name="grasp_weld")
    
def main():
    """
    Orchestrates the data generation process by running multiple episodes 
    of the task.
    
    Creates environment, runs scripted episodes, and saves demonstration data
    to compressed NPZ.

    Arguments that can be configured with flags:
    - env
    - render
    - demo_ctr

    """
    # Initialize the Panda environment.
    env = gym.make("PandaReachCube-v0", render_mode=_render_mode)
    env = gym.wrappers.FlattenObservation(env) 

    # Adjust physical settings
    # env.model.opt.time_step = 0.001  # Smaller time_step for more accurate physics. Default is 0.002.
    # env.model.opt.iterations = 100  # More solver iterations for better contact resolution. Default is 50.

    # Configuration parameters
    initStateSpace = "random"       # Initial state space configuration

    # Demos configs
    num_demos = NUM_DEMOS  # Number of demonstration episodes to generate--ADJUST THIS VALUE FOR MORE OR LESS DEMOS**  

    demo_ctr = 0         # Counter for successful demonstration episodes 
    
    # Reset environment to initial state - render for the first time.
    obs, _ = env.reset() # For reach environment expect 10 observations: r_pos, r_vel, finger, object_pos.

    # Adjust camera view for better visualization
    viewer = set_front_cam_view(env)

    print("Reset!")
    
    # Generate demonstration episodes
    while len(actions) < num_demos:
        obs,_ = env.reset() # Reset environment for new episode
        
        print(f"We will run a total of: {num_demos} demos!!")
        print("Demo: #", len(actions)+1)

        # Execute pick-and-place task
        res = demo(env, obs)

        # Print success message
        if res:                        
            demo_ctr += 1
            print("Episode completed successfully!")
            print(f"Total successful demos: {demo_ctr}/{num_demos}")

    # Close the environment after all episodes are done
    env.close()
    
    ## Write data to demos folder. Assumes mounted /data folder and internal data folder.
    script_dir = '/data/data/serl/demos'    

    # Create output filename with configuration details
    fileName = "data_" + robot + "_" + task
    fileName += "_" + initStateSpace
    fileName += "_" + str(num_demos)
    
    # Add timestamp to filename for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fileName += "_" + timestamp
    fileName += ".npz"

    # Build a filename in that same directory
    out_path = os.path.join(script_dir, fileName)    

    # Ensure the directory exists
    os.makedirs(script_dir, exist_ok=True)
    
    # Save collected data to compressed numpy NPZ file
    # Set acs,obs,info as keys in dict and values as np.arrays of type objects. This allows you to handle different lengths.
    np.savez_compressed(
        out_path,
        acs=np.array(actions, dtype=object),
        obs=np.array(observations, dtype=object),
        rewards=np.array(rewards, dtype=object),
        info=np.array(infos, dtype=object),
        terminateds=np.array(terminateds, dtype=object),
        truncateds=np.array(truncateds, dtype=object),
        dones=np.array(dones, dtype=object),
        transition_ctr=transition_ctr,
        num_demos=num_demos
    )
    
    print(f"Data saved to {fileName}.")
    print(f"Total successful demos: {demo_ctr}/{num_demos}")

if __name__ == "__main__":
    main()
