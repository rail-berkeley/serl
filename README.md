# SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning

![](https://github.com/rail-berkeley/serl/workflows/pre-commit/badge.svg)

## Installation
1. Conda Environment:
    - create an environment with `conda create -n serl python=3.10`

2. Install RL library
    - the examples here use jaxrl-minimal as the RL library.
    - To install jaxrl-minimal, `git clone https://github.com/rail-berkeley/jaxrl_minimal/tree/serl_dev`, the `serl_dev` branch is based off the latest `main` branch.
    - `cd` into the `jaxrl_minimal` path
    - run `pip install -e .` and `pip install -r requirements.txt` to install the jaxrl-minimal library and its dependencies.

3. Install the serl_launcher
    - `cd` into the `serl_launcher` folder
    - run `pip install -e .` to install the launcher package and its dependencies.

4. Install Franka Sim library (Optional)
    - `cd` into `franka_sim` path.
    - run `pip install -e .` and `pip install -r requirements.txt` to install the franka arm simulation package and its dependencies.


## Quick Start with Franka Arm in Sim
1. `cd` into `examples/async_sac_state_sim` folder
2. run `. ./run_learner.sh ` to launch the learner node
3. run `. ./run_actor.sh` to launch the actor node with rendering window.
4. You can optionally launch learner and actor on separate machines. For example, if learner node is running on a PC with `ip=x.x.x.x`, you can launch the actor node on a different machine with internet access to `ip=x.x.x.x` and add `--ip x.x.x.` to the commands in `run_actor.sh`.
