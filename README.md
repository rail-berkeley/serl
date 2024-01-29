# SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning

![](https://github.com/rail-berkeley/serl/workflows/pre-commit/badge.svg)

![](./docs/tasks-banner.gif)

**Webpage: https://serl-robot.github.io/**

SERL provides a set of libraries, env wrappers, and examples to train RL policies for robotic manipulation tasks. The following sections describe how to use SERL. We will illustate the usage with examples.

**Table of Contents**
- [SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning](#serl-a-software-suite-for-sample-efficient-robotic-reinforcement-learning)
  - [Installation](#installation)
  - [Quick Start with Franka Arm in Sim](#quick-start-with-franka-arm-in-sim)
    - [1. Training from state observation example](#1-training-from-state-observation-example)
    - [2. Training from image observation example](#2-training-from-image-observation-example)
    - [3. Training from image observation with 20 demo trajectories example](#3-training-from-image-observation-with-20-demo-trajectories-example)
  - [Run with Franka Arm on Real Robot](#run-with-franka-arm-on-real-robot)
    - [1. Peg Insertion 📍](#1-peg-insertion-)
    - [2. PCB Insertion 🖥️](#2-pcb-insertion-️)
    - [3. Cable Routing 🔌](#3-cable-routing-)
    - [4. Bin Relocation 🗑️](#4-bin-relocation-️)
  - [Citation](#citation)

---

## Installation
1. **Setup Conda Environment:**
    create an environment with
    ```bash
    conda create -n serl python=3.10
    ```

2. **Install Jax as follows:**
    - For CPU (not recommended):
        ```bash
        pip install --upgrade "jax[cpu]"
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

3. **Install the serl_launcher**
    ```bash
    cd serl_launcher
    pip install -e .
    pip install -r requirements.txt
    ```

4. **Install Franka Sim library (Optional)**
    ```bash
    cd franka_sim
    pip install -e .
    pip install -r requirements.txt
    ```

Try if franka_sim is running via `python franka_sim/franka_sim/test/test_gym_env_human.py`

---

## Quick Start with Franka Arm in Sim

Before beginning, please make sure that the simulation environment with `franka_sim` is working. Please refer to the [Quick Start with Franka Arm in Sim](#quick-start-with-franka-arm-in-sim) section for more details.

Note to set `MUJOCO_GL`` as egl if you are doing off-screen rendering.
You can do so by ```export MUJOCO_GL=egl``` and remember to set the rendering argument to False in the script.

### 1. Training from state observation example

One-liner launcher (requires `tmux`, `sudo apt install tmux`):):
```bash
bash examples/async_sac_state_sim/tmux_launch.sh
```

<details>
  <summary>Click to show detailed commands</summary>

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

</details>

### 2. Training from image observation example

One-liner launcher (requires `tmux`, `sudo apt install tmux`):
```bash
bash examples/async_drq_sim/tmux_launch.sh
```

<details>
  <summary>Click to show detailed commands</summary>

```bash
cd examples/async_drq_sim

# to use pre-trained ResNet weights, please download
# note manual download is only for now, once repo is public, auto download will work
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

</details>

### 3. Training from image observation with 20 demo trajectories example

One-liner launcher (requires `tmux`):
```bash
bash examples/async_sac_image_sim/tmux_launch.sh
```

<details>
  <summary>Click to show detailed commands</summary>

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

</details>

---

## Run with Franka Arm on Real Robot

We demonstrate how to use SERL with real robot manipulators with 4 different tasks. Namely: peg insertion, pcb insertion, cable routing, and bin relocation.

When running with a real robot, a separate gym env is needed. For our examples, we isolated the gym env as a client to a robot server. The robot server is a Flask server which sends commands to the robot via ROS. The gym env communicates with the robot server via post requests.

```mermaid
graph LR
A[Robot] <-- ROS --> B[Robot Server]
B <-- HTTP --> C[Gym Env]
C <-- Lib --> D[RL Policy]
```

This requires installation of the following packages:

- [serl_franka_controller](https://github.com/rail-berkeley/serl_franka_controller)
- `serl_robot_infra`: [readme](serl_robot_infra/README.md)


*NOTE: the following code will not run as it is, since it will requires custom datas, checkpoints and robot env. We provide the code as a reference for how to use SERL with real robots. Learn this section in incremental order, starting from the first task (peg insertion) to the last task (bin relocation). Modify the code according to your need*

### 1. Peg Insertion 📍

> Example is located in `examples/async_peg_insert_drq/`

> Env and default config is located in `franka_env/envs/peg_env/`

We record 20 demo trajectories with the robot. The trajectories are saved in `examples/async_peg_insert_drq/peg_insertion_20_trajs_{UUID}.pkl`.
```bash
python record_demo.py
```

The `franka_env.envs.wrappers.SpacemouseIntervention` gym wrapper provides the ability to intervene the robot with a spacemouse.

With the demo trajectories, we then use [DRQ](https://arxiv.org/abs/2004.13649) as the agent, and run both learner and actor nodes.
```bash
bash run_learner.sh
bash run_actor.sh
```

Reward is given when the peg is inserted into the hole. This is done by checking the target pose of the peg and the current pose of the peg, defined in the `peg_env/config.py`


### 2. PCB Insertion 🖥️

> Example is located in `examples/async_pcb_insert_drq`

> Env and default config is located in `franka_env/envs/pcb_env/`

Similar to peg insertion, here we record demo trajectories with the robot, then run the learner and actor nodes.
```bash
# record demo trajectories
python record_demo.py

# run learner and actor nodes
bash run_learner.sh
bash run_actor.sh
```

A baseline of using BC as policy is also provided. To train BC, simply run the following command:
```bash
python3 examples/bc_policy.py ....TODO_ADD_ARGS.....
```

To run the BC policy, simply run the following command:
```bash
bash run_bc.sh
```

### 3. Cable Routing 🔌

> Example is located in `examples/async_cable_routing_drq`

> Env and default config is located in `franka_env/envs/cable_env/`

In this cable routing task, we provided an example of a reward classfier. This replaced hardcoded reward classifier which depends on known `TARGET_POSE` defined in the `config.py`. The reward classifier is an image-based classifier (pretrained resnet), which is trained to classify whether the cable is routed successfully or not. The reward classifier is trained with demo trajectories of successful and failed samples.

```bash
# NOTE: custom paths are used in this script
python train_reward_classifier.py
```

The reward classifier is used as a gym wrapper `franka_env.envs.wrapper.BinaryRewardClassifier`. The wrapper classifies the current observation and return a reward of 1 if the observation is classified as successful, and 0 otherwise.

The reward classifier is then used in the BC policy and DRQ policy for the actor node, path is provided as `--reward_classifier_ckpt_path` argument in `run_bc.sh` and `run_actor.sh`


### 4. Bin Relocation 🗑️

> Example is located in `examples/async_bin_relocation_fwbw_drq`

> Env and default config is located in `franka_env/envs/bin_env/`

This bin relocation example demonstrates the usage of a forward and backward policies. This is helpful for RL tasks, which requires the robot to "reset". In this case, the robot is moving an object from one bin to another. The forward policy is used to move the object from left bin to right bin, and the backward policy is used to move the object from right bin to left bin.

1. Record demo trajectories

Multiple utility scripts has been provided to record demo trajectories. (e.g. `record_demo.py`: for rlpd, `record_transitions.py`: for reward classifier, `reward_bc_demos.py`: for bc policy). Note that both forward and backward trajectories requires different demo trajectories.

2. Reward Classifier

Similar to the cable routing example, we need to train two reward classifiers for both forward and backward policies, shown in `train_fwd_reward_classifier.sh` and `train_bwd_reward_classifier.sh`. The reward classifiers are then used in the BC and DRQ policy for the actor node, checkpoint path is provided as `--reward_classifier_ckpt_path` argument in `run_bc.sh` and `run_actor.sh`.

3. Run 2 learners and 1 actor with 2 policies

Finally, 2 learners node will learn both forward and backward policies respectively. The actor node will switch between running the forward and backward policies with their respective reward classifiers during the RL training process.

```bash
bash run_actor.sh

# run 2 learners
bash run_fw_learner.sh
bash run_bw_learner.sh
```

## Citation

If you use this code for your research, please cite our paper:

```
TODO
```
