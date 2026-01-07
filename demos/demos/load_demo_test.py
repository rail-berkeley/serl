# Updated and advanced train.py that includes logging, vectorized environments, and periodic recorded evaluations
import os
from pathlib import Path
import numpy as np

def load_demos_to_her_buffer_gymnasium(data_store, demo_npz_path: str, combine_done: bool = True):
    """
    Load a raw Gymnasium-style .npz of expert episodes into model.replay_buffer.

    demo_npz_path must contain at least these arrays:
      - 'episodeObs'         : shape (T+1, *obs_shape*), list of observations
      - 'episodeAcs'         : shape (T, *act_shape*),   list of actions
      - 'episodeRews'        : shape (T,),               list of rewards
      - 'episodeTerminated'  : shape (T,),               list of terminated flags
      - 'episodeTruncated'   : shape (T,),               list of truncated flags
      - 'episodeInfo'        : shape (T,),               list of info dicts

    Parameters
    ----------
    data_store : DataStore
        The data store to which the demo transitions will be added.
    demo_npz_path : str
        Path to the .npz file you saved from your demo collector.
    combine_done : bool, default=True
        If True, `done = terminated or truncated`.  If False, `done = terminated` only.
    """
    
    # Load all demo data. Structure: var_name[num_demo][time_step][key if dict] = value
    data = np.load(demo_npz_path, allow_pickle=True)

    obs_buffer       = data['obs']          # length T+1
    act_buffer       = data['acs']          # length T
    rew_buffer       = data['rewards']      # length T
    term_buffer      = data['terminateds']  # length T
    trunc_buffer     = data['truncateds']   # length T
    info_buffer      = data['info']         # length T
    done_buffer      = data['dones']        # length T, if available

    # Extract number of demonstrations
    num_demos = obs_buffer.shape[0]

    # Extract rollout data for a single episode
    for ep in range(num_demos):
        ep_obs   = obs_buffer[ep]    # this is a length‐(T+1) array of dicts
        ep_acts  = act_buffer[ep]    # length‐T array of actions
        ep_rews  = rew_buffer[ep]
        ep_terms = term_buffer[ep]
        ep_trunc = trunc_buffer[ep]
        ep_done  = done_buffer[ep]
        ep_info  = info_buffer[ep]   # length‐T array of dicts

        # Length of episode:
        T = len(ep_acts)

        # Extract single transitions from the episode data
        for t in range(T):
            # raw single‐step data:
            obs_t      = ep_obs[t]       # dict[str, np.ndarray]  (obs_dim,)
            next_obs_t = ep_obs[t+1]
            a_t        = ep_acts[t]      # np.ndarray (action_dim,)
            r_t        = float(ep_rews[t])
            done_t     = bool(ep_done[t] or ep_terms[t] or ep_trunc[t])

            # Rehydrate info dict and inject the timeout flag
            raw_info     = ep_info[t]      # dict[str,Any]
            if isinstance(raw_info, str):
                import ast
                info_t = ast.literal_eval(raw_info)
            else:
                info_t = raw_info.copy()  
            # Append truncated information to info_t
            info_t["TimeLimit.truncated"] = bool(ep_trunc[t])                      

            # Enter transition into data_store or QueuedDataStore
            data_store.insert(
                            dict(
                                observations=obs_t,
                                actions=a_t,
                                next_observations=next_obs_t,
                                rewards=r_t,
                                masks=1.0 - done_t,
                                dones=done_t
                            )
                        )       

    print(f"Can load {num_demos} transitions successfullly from {demo_npz_path}."
          f"(combine_done={combine_done}).")

def get_demo_path(relative_path: str) -> str:
    """
    Given a path relative to this script file, return
    the absolute, normalized path as a string.

    Example:
        # If your demos live at ../../../demos/data.npz
        demo_file = get_demo_path("../../../demos/data_franka_random_10.npz")
    """
    # 1) Resolve this script’s directory
    script_dir = Path(__file__).resolve().parent

    # 2) Join with the user-supplied relative path and normalize
    full_path = (script_dir / relative_path).resolve()

    return str(full_path)


def main():
    """
    Load demos    
    """

    # Get abs path to demo file
    script_dir = '/data/data/serl/demos' 
    default_file = 'data_franka_reach_random_20.npz'

    prompt = f"Please input the name of the file to load [{default_file}]: "

    file_name = input(prompt) or default_file
    demo_file = os.path.join(script_dir, file_name)

    # Load the demo file into data_store as in async_sac_state.py
    from agentlace.data.data_store import QueuedDataStore
    data_store = QueuedDataStore(2000) 
    load_demos_to_her_buffer_gymnasium(data_store, demo_file, combine_done=True)

if __name__ == "__main__":
    main()