# SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning

![](https://github.com/rail-berkeley/serl/workflows/pre-commit/badge.svg)

## Installation
1. Conda Environment:
    create an environment with
    ```bash
    conda create -n serl python=3.10
    ```

2. Install RL library
    - the examples here use jaxrl-minimal as the RL library.
    - To install jaxrl-minimal, with `serl_dev` branch is based off the latest `main` branch.
        ```bash
        git clone https://github.com/rail-berkeley/jaxrl_minimal/tree/serl_dev
        ```
    - install and its dependencies
        ```bash
        cd jaxrl_minimal
        pip install -e .
        pip install -r requirements.txt
        ```
    - For GPU: (change cuda12 to cuda11 if you are using older driver versions)
        ```bash
        pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ```

    - For TPU
        ```bash
        pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        ```
    - See the [Jax Github page](https://github.com/google/jax) for more details on installing Jax.

3. Install the serl_launcher
    ```bash
    cd serl_launcher
    pip install -e .
    ```

1. Install Franka Sim library (Optional)
    ```bash
    cd franka_sim
    pip install -e .
    pip install -r requirements.txt
    ```

Try if franka_sim is running via `python franka_sim/franka_sim/test/test_gym_env_human.py`

## Quick Start with Franka Arm in Sim
### 1. Training from state observation example
```bash
cd examples/async_sac_state_sim
```

Run learner node:
```bash
bash run_learner.sh
```

Run actor node with rendering window:
```bash
# add --ip x.x.x.x if running on a different machine
bash run_actor.sh
```

You can optionally launch learner and actor on separate machines. For example, if learner node is running on a PC with `ip=x.x.x.x`, you can launch the actor node on a different machine with internet access to `ip=x.x.x.x` and add `--ip x.x.x.` to the commands in `run_actor.sh`.

### 2. Training from image observation example
```bash
cd examples/async_drq_sim
```

Run learner node:
```bash
bash run_learner.sh
```

Run actor node with rendering window:
```bash
# add --ip x.x.x.x if running on a different machine
bash run_actor.sh
```

### 2. Training from image observation with 20 demo trajectories example
```bash
cd examples/async_rlpd_drq_sim
# download 20 demo trajectories
wget \
https://github.com/rail-berkeley/serl/releases/download/franka_sim_lift_cube_demos/franka_lift_cube_image_20_trajs.pkl
```

Run learner node:
```bash
bash run_learner.sh
```

Run actor node with rendering window:
```bash
# add --ip x.x.x.x if running on a different machine
bash run_actor.sh
```
