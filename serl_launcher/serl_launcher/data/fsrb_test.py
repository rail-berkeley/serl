import gym.wrappers
import numpy as np
import gym
from serl_launcher.utils.launcher import make_replay_buffer
from serl_launcher.data.fractal_symmetry_replay_buffer import FractalSymmetryReplayBuffer
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from absl import app, flags
import franka_sim
# import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_integer("capacity", 10, "Replay buffer capacity.")
flags.DEFINE_string("branch_method", "constant", "Method for determining the number of transforms per dimension (x,y)")
flags.DEFINE_string("split_method", "never", "Method for determining whether to change the number of transforms per dimension (x,y)")
flags.DEFINE_float("workspace_width", 0.5, "workspace width in meters")
flags.DEFINE_integer("max_depth", 4, "Maximum level of depth") # For fractal_branch only
flags.DEFINE_integer("max_steps",100,"Maximum steps")
flags.DEFINE_integer("branching_factor", 3, "Rate of change of number of transforms per dimension (x,y)") # For fractal_branch only
flags.DEFINE_integer("starting_branch_count", 1, "Initial number of transforms per dimension (x,y)") # For constant_branch only
flags.DEFINE_integer("alpha",1,"alpha value")
# Density Workspace width
flags.DEFINE_string("workspace_width_method",'increase', 'Controls workspace width dimensions configurations')

def main(_):

    x_obs_idx = np.array([0, 4])
    y_obs_idx = np.array([1, 5])

    # Initialize replay buffer
    env = gym.make("PandaPickCubeVision-v0")
    env = SERLObsWrapper(env)
    env = ChunkingWrapper(env, obs_horizon=3, act_exec_horizon=None)

    # env = gym.make("PandaReachCube-v0")
    # env = gym.wrappers.FlattenObservation(env)

    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    their_buffer = make_replay_buffer(
        env,
        type="memory_efficient_replay_buffer",
        capacity=FLAGS.capacity,
        image_keys=image_keys,

    )

    replay_buffer = make_replay_buffer(
        env,
        type="fractal_symmetry_replay_buffer",
        capacity=FLAGS.capacity,
        split_method=FLAGS.split_method,
        branch_method=FLAGS.branch_method,
        workspace_width=FLAGS.workspace_width,
        x_obs_idx=x_obs_idx,
        y_obs_idx= y_obs_idx,
        image_keys=image_keys,
        # max_depth=FLAGS.max_depth,
        max_traj_length = 100,
        # branching_factor=FLAGS.branching_factor,
        # alpha = FLAGS.alpha,
        starting_branch_count = FLAGS.starting_branch_count,
    )

    observation, info = env.reset()
    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, info = env.step(action)
    
    # observation = np.zeros_like(observation)
    for k in observation.keys():
        observation[k] = np.zeros_like(observation[k])
        next_observation[k] = np.ones_like(next_observation[k])
    
    action = np.ones_like(action)
    # next_observation = np.ones_like(next_observation)
    reward = 1

    data_dict = dict(
        observations=observation,
        next_observations=next_observation,
        actions=action,
        rewards=reward,
        masks=not truncated and not terminated,
        dones=truncated or terminated,
    )

    del env, observation, next_observation, action, reward, truncated, terminated, info, y_obs_idx, x_obs_idx, _

    for i in range(6):

        replay_buffer.insert(data_dict)
        their_buffer.insert(data_dict)
        assert(replay_buffer.dataset_dict["observations"]["state"][i % 10].all() == their_buffer.dataset_dict["observations"]["state"][(i + 3) % 10].all())
        assert(replay_buffer.dataset_dict["next_observations"]["state"][i % 10].all() == their_buffer.dataset_dict["next_observations"]["state"][(i + 3) % 10].all())
        assert(replay_buffer.img_buffer["front"][replay_buffer.dataset_dict["observations"]["front"][i % 10]].all() == their_buffer.dataset_dict["observations"]["front"][(i + 3) % 10].all())

        data_dict["observations"]["state"] += 1
        data_dict["next_observations"]["state"] += 1
        for k in image_keys:
            data_dict["observations"][k] += 1
            data_dict["next_observations"][k] += 1
            
    replay_buffer.sample(batch_size=3, indx=np.array([2,3,4]))
    their_buffer.sample(batch_size=3, indx=np.array([5,6,7]))

    

    # branch() tests

    #-------------------------------------------------------------------
    # Fractal Associative Expansions
    #-------------------------------------------------------------------
    replay_buffer.branching_factor = 3

    replay_buffer.current_depth = 1
    result = replay_buffer.fractal_branch()
    expected = 3
    assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    replay_buffer.current_depth = 2
    result = replay_buffer.fractal_branch()
    expected = 9
    assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    replay_buffer.current_depth = 3
    result = replay_buffer.fractal_branch()
    expected = 27
    assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    replay_buffer.current_depth = 4
    result = replay_buffer.fractal_branch()
    expected = 81
    assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    replay_buffer.current_depth = 0

    del result, expected

    print("\033[32mTEST PASSED \033[0m fractal_branch() tests passed")

    #-------------------------------------------------------------------
    # Fractal Associative Contractions
    #-------------------------------------------------------------------    
    replay_buffer.branching_factor = 3

    replay_buffer.current_depth = 1
    result = replay_buffer.fractal_contraction()
    expected = 81
    assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    replay_buffer.current_depth = 2
    result = replay_buffer.fractal_contraction()
    expected = 27
    assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    replay_buffer.current_depth = 3
    result = replay_buffer.fractal_contraction()
    expected = 9
    assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    replay_buffer.current_depth = 4
    result = replay_buffer.fractal_contraction()
    expected = 3
    assert result == expected, f"\033[31mTEST FAILED\033[0m fractal_branch() test failed (expected {expected} but got {result})"
    
    replay_buffer.current_depth = 0

    del result, expected

    print("\033[32mTEST PASSED \033[0m fractal_contraction() tests passed")

    #-------------------------------------------------------------------
    # split() tests
    #-------------------------------------------------------------------

    ## time
    replay_buffer.max_steps = 100
    replay_buffer.max_depth = 4

    replay_buffer.timestep = 0
    result = replay_buffer.time_split(data_dict)
    expected = True
    assert result == expected, f"\033[31mTEST FAILED\033[0m split() test failed (expected {expected} but got {result})"
    
    replay_buffer.timestep = 25
    result = replay_buffer.time_split(data_dict)
    expected = True
    assert result == expected, f"\033[31mTEST FAILED\033[0m split() test failed (expected {expected} but got {result})"

    replay_buffer.timestep = 50
    result = replay_buffer.time_split(data_dict)
    expected = True
    assert result == expected, f"\033[31mTEST FAILED\033[0m split() test failed (expected {expected} but got {result})"

    replay_buffer.timestep = 75
    result = replay_buffer.time_split(data_dict)
    expected = True
    assert result == expected, f"\033[31mTEST FAILED\033[0m split() test failed (expected {expected} but got {result})"


    replay_buffer.timestep = 100
    result = replay_buffer.time_split(data_dict)
    expected = False
    assert result == expected, f"\033[31mTEST FAILED\033[0m split() test failed (expected {expected} but got {result})"

    del result, expected

    print("\033[32mTEST PASSED \033[0m time_split() test passed")

    # insert() tests
    # insert() tests
    initial_size = len(replay_buffer.dataset_dict['observations'][0]) * replay_buffer._insert_index % len(replay_buffer.dataset_dict['observations'])

    replay_buffer.insert(data_dict)
    final_size = len(replay_buffer.dataset_dict['observations'][0]) * replay_buffer._insert_index % len(replay_buffer.dataset_dict['observations'])
    
    result = final_size > initial_size
    expected = True
    assert result == expected, f"\033[31mTEST FAILED\033[0m insert() test failed (expected buffer size to increase from {initial_size} to {final_size})"
    del result, expected, initial_size, final_size


    print("\033[32mTEST PASSED \033[0m insert() tests passed")

    #-------------------------------------------------------------------
    # Fractal Expansions with workspace_width_modification
    #-------------------------------------------------------------------
    print('\nWorkspace width tests....')

    replay_buffer.branching_factor = 3
    replay_buffer.current_depth = 1

    if FLAGS.workspace_width_method == 'constant':
        result = replay_buffer.get_workspace_width()
        expected = FLAGS.workspace_width
        assert result == expected, f"\033[31mTEST FAILED\033[0m get_workspace_width() test failed (expected {expected} but got {result})"
    
    elif FLAGS.workspace_width_method == 'decrease':
        result = replay_buffer.get_workspace_width()
        expected = FLAGS.workspace_width - 0.05*replay_buffer.current_depth
        assert result == expected, f"\033[31mTEST FAILED\033[0m get_workspace_width() test failed (expected {expected} but got {result})"
    
    elif FLAGS.workspace_width_method == 'increase':
        result = replay_buffer.get_workspace_width()
        expected = FLAGS.workspace_width + 0.05*replay_buffer.current_depth
        assert result == expected, f"\033[31mTEST FAILED\033[0m get_workspace_width() test failed (expected {expected} but got {result})"

    else:
        raise NameError('There is no workspace width method with that name.')
    

    replay_buffer.current_depth = 2

    if FLAGS.workspace_width_method == 'constant':
        result = replay_buffer.get_workspace_width()
        expected = FLAGS.workspace_width
        assert result == expected, f"\033[31mTEST FAILED\033[0m get_workspace_width() test failed (expected {expected} but got {result})"
    
    elif FLAGS.workspace_width_method == 'decrease':
        result = replay_buffer.get_workspace_width()
        expected = FLAGS.workspace_width - 0.05*replay_buffer.current_depth
        assert result == expected, f"\033[31mTEST FAILED\033[0m get_workspace_width() test failed (expected {expected} but got {result})"
    
    elif FLAGS.workspace_width_method == 'increase':
        result = replay_buffer.get_workspace_width()
        expected = FLAGS.workspace_width + 0.05*replay_buffer.current_depth
        assert result == expected, f"\033[31mTEST FAILED\033[0m get_workspace_width() test failed (expected {expected} but got {result})"

    else:
        raise NameError('There is no workspace width method with that name.')


    replay_buffer.current_depth = 3

    if FLAGS.workspace_width_method == 'constant':
        result = replay_buffer.get_workspace_width()
        expected = FLAGS.workspace_width
        assert result == expected, f"\033[31mTEST FAILED\033[0m get_workspace_width() test failed (expected {expected} but got {result})"
    
    elif FLAGS.workspace_width_method == 'decrease':
        result = replay_buffer.get_workspace_width()
        expected = FLAGS.workspace_width - 0.05*replay_buffer.current_depth
        assert result == expected, f"\033[31mTEST FAILED\033[0m get_workspace_width() test failed (expected {expected} but got {result})"
    
    elif FLAGS.workspace_width_method == 'increase':
        result = replay_buffer.get_workspace_width()
        expected = FLAGS.workspace_width + 0.05*replay_buffer.current_depth
        assert result == expected, f"\033[31mTEST FAILED\033[0m get_workspace_width() test failed (expected {expected} but got {result})"

    else:
        raise NameError('There is no workspace width method with that name.')
    
    replay_buffer.current_depth = 0

    del result, expected

    print("\n\033[32mTEST PASSED \033[0m workspace_width_method() test passed")    

        
    print("\nfinished!\n")



if __name__ == "__main__":
    app.run(main)