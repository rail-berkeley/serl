"""
Scripted Controller for Franka FR3 Robot - Demonstration Data Generation

This script implements a scripted controller for generating expert demonstration data 
for Deep Reinforcement Learning (DRL) algorithms using the Franka FR3 robot in a 
pick-and-place task. The generated data serves as bootstrapping demonstrations for 
training RL agents with stable-baselines3.

The controller uses a 4-phase hierarchical approach:
1. Approach Object (move gripper above object)
2. Grasp Object (move to object and close gripper)  
3. Transport to Goal (move grasped object to target)
4. Maintain Position (hold final position)

Output: Compressed NPZ file containing action, observation, and info sequences
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from time import sleep
from panda_mujoco_gym.envs import FrankaPickAndPlaceEnv

# Global variables to store episode data across all iterations
observations = []   # List storing observation sequences for each episode  
actions = []        # List storing action sequences for each episode
rewards = []        # List storing reward sequences for each episode
infos = []          # List storing info dictionaries for each episode
terminateds = []    # List storing terminated flags for each episode
truncateds = []     # List storing truncated flags for each episode
dones = []          # List storing done flags (terminated or truncated) for each episode

# Robot configuration
robot = 'franka'    # Robot type used in the environment, can be 'franka' or 'fetch'

# Weld constraint flag
weld_flag = True    # Flag to activate weld constraint during pick-and-place


def activate_weld(env, constraint_name="grasp_weld"):
    """
    Activate a weld constraint during pick portion of a demo
    :param env: The environment containing the model
    :param constraint_name: The name of the weld constraint to activate
    :return: True if the weld was successfully activated, False if the constraint was not found
    """

    try:
        # Activate the weld constraint
        env.unwrapped.model.eq(constraint_name).active = 1     
        print("Activated weld")   
        return True
    
    except KeyError:
        print(f"Warning: Constraint '{constraint_name}' not found")
        return False

def deactivate_weld(env, constraint_name="grasp_weld"):
    """
    Deactivate a weld constraint during place portion of a demo
    :param env: The environment containing the model
    :param constraint_name: The name of the weld constraint to deactivate
    :return: True if the weld was successfully deactivated, False if the constraint was not
    found
    """ 
    
    try:
        # Deactivate the weld constraint
        env.unwrapped.model.eq(constraint_name).active = 0    
        print("Deactivated weld")    
        return True
    
    except KeyError:
        print(f"Warning: Constraint '{constraint_name}' not found")
        return False

def main():
    """
    Orchestrates the data generation process by running multiple episodes 
    of the pick-and-place task.
    
    Creates environment, runs scripted episodes, and saves demonstration data
    to compressed NPZ file for use with stable-baselines3.
    """
    # Initialize Fetch pick-and-place environment
    env = FrankaPickAndPlaceEnv(reward_type="sparse", render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=50)  

    # Adjust physical settings
    # env.model.opt.timestep = 0.001  # Smaller timestep for more accurate physics. Default is 0.002.
    # env.model.opt.iterations = 100  # More solver iterations for better contact resolution. Default is 50.

    # Configuration parameters
    initStateSpace = "random"       # Initial state space configuration

    # Demos configs
    attempted_demos = 1  # Number of demonstration episodes to generate--ADJUST THIS VALUE FOR MORE OR LESS DEMOS**  

    num_demos = 0         # Counter for successful demonstration episodes 
    
    # Reset environment to initial state - render for the first time.
    obs, _ = env.reset()
    print("Reset!")
    
    # Generate demonstration episodes
    while len(actions) < attempted_demos:
        obs,_ = env.reset() # Reset environment for new episode
        print(f"We will run a total of: {attempted_demos} demos!!")
        print("Demo: #", len(actions)+1)

        # Execute pick-and-place task
        res = pick_and_place_demo(env, obs)

        # Print success message
        if res:                        
            num_demos += 1
            print("Episode completed successfully!")
            print(f"Total successful demos: {num_demos}/{attempted_demos}")
    
    ## Write data to demos folder
    # 1. Get the absolute path of this script
    #script_path = os.path.abspath(__file__)

    # 2. Extract its directory
    #script_dir = os.path.dirname(script_path)
    script_dir = '/home/student/data/franka_baselines/demos/pick_n_place' # Assumes data folder in user directory.

    # 3. Create output filename with configuration details
    fileName = "data_" + robot
    fileName += "_" + initStateSpace
    fileName += "_" + str(attempted_demos)
    fileName += ".npz"

    # 3. Build a filename in that same directory
    out_path = os.path.join(script_dir, fileName)    
    
    # Save collected data to compressed numpy NPZ file
    # Set acs,obs,info as keys in dict 
    np.savez_compressed(out_path, 
                        acs = actions, 
                        obs = observations, 
                        rewards = rewards,
                        info = infos,
                        terminateds = terminateds,
                        truncateds = truncateds,
                        dones = dones)
    
    print(f"Data saved to {fileName}.")

def pick_and_place_demo(env, lastObs):
    """
    Executes a scripted pick-and-place sequence using a hierarchical approach.
    
    Implements 4-phase control strategy:
    1. Approach: Move gripper above object (3cm offset)
    2. Grasp: Move to object and close gripper  
    3. Transport: Move grasped object to goal position
    4. Maintain: Hold position until episode ends

    Store observations, actions, and info in global lists for later replay buffer inclusion.
    
    Args:
        env: Gymnasium environment instance
        lastObs: Last observation containing goal and object state information
            - desired_goal: Target position for object placement
            - observations:
                ee_position[0:3],
                ee_velocity[3:6],
                fingers_width[6],
                object_position[7:10],
                object_rotation[10:13],
                object_velp[13:16],
                object_velr[16:19],
    """

    ## Init goal, current_pos, and object position from last observation
    goal             = np.zeros(3, dtype=np.float32)
    current_pos      = np.zeros(3, dtype=np.float32)
    object_pos       = np.zeros(3, dtype=np.float32)
    object_rel_pos   = np.zeros(3, dtype=np.float32)
    fgr_pos          = np.zeros(1, dtype=np.float32)

    # Initialize episode data collection
    episodeObs  = []        # Observations for this episode  
    episodeAcs  = []        # Actions for this episode
    episodeRews = []        # Rewards for this episode
    episodeInfo = []        # Info for this episode
    episodeTerminated = []  # Terminated flags for this episode
    episodeTruncated = []   # Truncated flags for this episode
    episodeDones = []       # Done flags (terminated or truncated) for this episode

    # Proportional control gain for action scaling -- empirically tuned
    Kp = 8.0            

    # pre_pick_offset
    pre_pick_offset = np.array([0,0,0.03], dtype=float)  # Offset to approach object safely (3cm)
    
    # Error thresholds
    error_threshold = 0.011  # Threshold for stopping condition (Xmm)

    finger_delta_fast = 0.05    # Action delta for fingers 5cm per step (will get clipped by controller)... more of a scalar. 
    finger_delta_slow = 0.005   # Franka has a range from 0 to 4cm per finger

    ## Extract data
    # Extract desired position from desired_goal dict
    goal = lastObs["desired_goal"][0:3]
    
    # Current robot end-effector position from observation dict
    current_pos = lastObs["observation"][0:3]
    
    # Current object position from observation dict: 
    object_pos = lastObs["observation"][7:10]
    
    # Relative position between end-effector and object
    object_rel_pos = object_pos - current_pos  
    
    ## Phase 1: Approach Object (Above)
    # Create target position 3cm above the object. Use copy() method.
    error = object_rel_pos.copy() 
    error+=pre_pick_offset  # Move 3cm above object for safe approach. Fingers should still end up surrounding object.
    
    timeStep = 0  # Track total timesteps in episode
    episodeObs.append(lastObs)
    
    # Phase 1: Move gripper to position above object
    # Terminate when distance to above-object position < 5mm
    print(f"----------------------------------------------- Phase 1: Approach Object -----------------------------------------------")
    while np.linalg.norm(error) >= error_threshold and timeStep <= env._max_episode_steps:
        env.render()  # Visual feedback
        
        # Initialize action vector [x, y, z, gripper]
        action = np.array([0., 0., 0., 0.])
        
        # Proportional control with gain of 6
        # action = Kp * error
        action[:3] = error * Kp
        
        # Open gripper for approach
        action[ len(action)-1 ] = 0.05
        
        # Unpack new Gymnasium step API
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        timeStep += 1
        
        # Store episode data
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeRews.append(reward)
        episodeObs.append(new_obs)
        episodeTerminated.append(terminated)
        episodeTruncated.append(truncated)
        episodeDones.append(done)

        # Update state information
        fgr_pos          = new_obs["observation"][6]  
        current_pos = new_obs["observation"][0:3]
        object_pos  = new_obs['observation'][7:10]  
        error            = (object_pos+pre_pick_offset) - current_pos  # Error with regard to offset position         

        # Print debug information
        print(
                f"Time Step: {timeStep}, Error: {np.linalg.norm(error):.4f}, "
                f"Eff_pos: {np.array2string(current_pos, precision=3)}, "
                f"obj_pos: {np.array2string(object_pos, precision=3)}, "
                f"fgr_pos: {np.array2string(fgr_pos, precision=2)}, "
                f"Error: {np.array2string(error, precision=3)}, "
                f"Action: {np.array2string(action, precision=3)}"
                )
        
    # Phase 2: Descend Grasp Object
    # Move gripper directly to object and close gripper
    # Terminate when relative distance to object < 5mm
    print(f"----------------------------------------------- Phase 2: Grip -----------------------------------------------")
    error = object_pos - current_pos # remove offset
    while (np.linalg.norm(error) >= error_threshold or fgr_pos>=0.39) and timeStep <= env._max_episode_steps: # Cube of width 4cm, each finger open to 2cm
        env.render()
        
        # Initialize action vector [x, y, z, gripper]
        action = np.array([0., 0., 0., 0.])
        
        # Direct proportional control to object position
        action[:3] = error * Kp
        
        # Close gripper to grasp object
        action[len(action)-1] = -finger_delta_fast * 2
        
        # Execute action and collect data
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        timeStep += 1
        
        # Store episode data
        episodeObs.append(new_obs)
        episodeRews.append(reward)
        episodeAcs.append(action)
        episodeInfo.append(info)     
        episodeTerminated.append(terminated)
        episodeTruncated.append(truncated)
        episodeDones.append(done)           

        # Update state information
        fgr_pos = new_obs["observation"][6]  
        current_pos = new_obs["observation"][0:3]
        object_pos = new_obs['observation'][7:10]
        error = object_pos - current_pos #- np.array([0.,0.,0.01]) # Grab lower

       # Print debug information
        print(
                f"Time Step: {timeStep}, Error: {np.linalg.norm(error):.4f}, "
                f"Eff_pos: {np.array2string(current_pos, precision=3)}, "
                f"obj_pos: {np.array2string(object_pos, precision=3)}, "
                f"fgr_pos: {np.array2string(fgr_pos, precision=3)}, "
                f"Error: {np.array2string(error, precision=3)}, "
                f"Action: {np.array2string(action, precision=3)}"
                )    
        #sleep(0.5)  # Optional: Slow down for better visualization
    
    # Phase 3: Transport to Goal
    # Move grasped object to desired goal position
    # Terminate when distance between object and goal < 1cm
    print(f"----------------------------------------------- Phase 3: Transport to Goal -----------------------------------------------")

    # Weld activation
    if weld_flag:
        activate_weld(env, constraint_name="grasp_weld")

    # Set error between goal and hand assuming the object is grasped
    gh_error = goal - current_pos        # Error between goal and hand position
    ho_error = object_pos - current_pos  # Error between object and hand position
    while np.linalg.norm(gh_error) >= 0.01 and timeStep <= env._max_episode_steps:
        env.render()
        
        action = np.array([0., 0., 0., 0.])
        
        # Proportional control toward goal position
        action[:3] = gh_error[:3] * Kp
        
        # Maintain grip on object
        #action[len(action)-1] = 0  
        
        # Execute action and collect data
        new_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        timeStep += 1
        
        # Store episode data
        episodeObs.append(new_obs)
        episodeRews.append(reward)
        episodeAcs.append(action)
        episodeInfo.append(info)
        episodeTerminated.append(terminated)
        episodeTruncated.append(truncated)
        episodeDones.append(done)        
        
        # Update state information
        fgr_pos = new_obs["observation"][6] 
        current_pos = new_obs["observation"][0:3]
        object_pos = new_obs['observation'][7:10]
        gh_error = goal - current_pos        # Error between goal and hand position
        ho_error = object_pos - current_pos  # Error between object and hand position

        # Print debug information
        print(
                f"Time Step: {timeStep}, Error Norm: {np.linalg.norm(gh_error):.4f}, "
                f"Eff_pos: {np.array2string(current_pos, precision=3)}, "
                f"goal_pos: {np.array2string(goal, precision=3)}, "
                f"fgr_pos: {np.array2string(fgr_pos, precision=2)}, "
                f"Error: {np.array2string(gh_error, precision=3)}, "
                f"Action: {np.array2string(action, precision=3)}"
                )    
    
        sleep(0.5)  # Optional: Slow down for better visualization

        ## Check for success and store episode data
        gh_norm = np.linalg.norm(gh_error)
        ho_nomr = np.linalg.norm(ho_error)
        if  gh_norm < error_threshold and ho_nomr < error_threshold:
        
            # Store complete episode data in global lists only if we succeeded (avoid bad demos)
            actions.append(episodeAcs)
            observations.append(episodeObs)
            infos.append(episodeInfo)
            rewards.append(episodeRews)

            # Optionally, also store the done/terminated/truncated flags globally if needed:
            terminateds.append(episodeTerminated)
            truncateds.append(episodeTruncated)
            dones.append(episodeDones)    

            # Deactivate weld constraint after successful pick
            if weld_flag:
                deactivate_weld(env, constraint_name="grasp_weld")

            # Close mujoco viewer
            env.close()

            # Break out of the loop to start a new episode
            return True
        
    # If we reach here, the episode was not successful
    if weld_flag:
        print("Failed to transport object to goal position. Deactivating weld.")
        deactivate_weld(env, constraint_name="grasp_weld")
    
if __name__ == "__main__":
    main()