# Quick Start with SERL in Sim

![](./images/franka_sim.png)

## Installation

**Install Franka Sim library**
```bash
    cd franka_sim
    pip install -e .
    pip install -r requirements.txt
```

Try if `franka_sim` is running via `python franka_sim/franka_sim/test/test_gym_env_human.py`.

Before beginning, please make sure that the simulation environment with `franka_sim` is working.

*Note: to set `MUJOCO_GL` as egl if you are doing off-screen rendering.
You can do so by ```export MUJOCO_GL=egl``` and remember to set the rendering argument to False in the script.
If receives `Cannot initialize a EGL device display due to GLIBCXX not found` error, try run `conda install -c conda-forge libstdcxx-ng` ([ref](https://stackoverflow.com/a/74132234))*


Optionally install `tmux`: `sudo apt install tmux`

## 1. Training from state observation example

**✨ One-liner launcher (requires `tmux`) ✨**
```bash
bash examples/async_sac_state_sim/tmux_launch.sh
```

To kill the tmux session, run `tmux kill-session -t serl_session`.

### Without using one-liner tmux launcher

You can opt for running the commands individually in 2 different terminals.

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

You can optionally launch the learner and actor on separate machines. For example, if the learner node is running on a PC with `ip=x.x.x.x`, you can launch the actor node on a different machine with internet access to `ip=x.x.x.x` and add `--ip x.x.x.` to the commands in `run_actor.sh`.

Remove `--debug` flag in `run_learner.sh` to upload training stats to `wandb`.

## 2. Training from image observation example

**✨ One-liner launcher (requires `tmux`) ✨**

```bash
bash examples/async_drq_sim/tmux_launch.sh
```

### Without using one-liner tmux launcher

You can opt for running the commands individually in 2 different terminals.

```bash
cd examples/async_drq_sim

# to use pre-trained ResNet weights, please download
wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl
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

## 3. Training from image observation with 20 demo trajectories example

**✨ One-liner launcher (requires `tmux`) ✨**
```bash
bash examples/async_rlpd_drq_sim/tmux_launch.sh
```

### Without using one-liner tmux launcher

You can opt for running the commands individually in 2 different terminals.

```bash
cd examples/async_rlpd_drq_sim

# to use pre-trained ResNet weights, please download
# note manual download is only for now, once repo is public, auto download will work
wget https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl

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
