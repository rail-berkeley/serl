# SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning

![](https://github.com/rail-berkeley/serl/workflows/pre-commit/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://serl-robot.github.io/)

![](./docs/images/tasks-banner.gif)

**Webpage: [https://serl-robot.github.io/](https://serl-robot.github.io/)**

SERL provides a set of libraries, env wrappers, and examples to train RL policies for robotic manipulation tasks. The following sections describe how to use SERL. We will illustrate the usage with examples.

**Table of Contents**
- [SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning](#serl-a-software-suite-for-sample-efficient-robotic-reinforcement-learning)
  - [Installation](#installation)
  - [Overview and Code Structure](/docs/overview_structure.md)
  - [Quick Start with Franka Arm in Sim](/docs/sim_quick_start.md)
    - [1. Training from state observation example](/docs/sim_quick_start.md#1-training-from-state-observation-example)
    - [2. Training from image observation example](/docs/sim_quick_start.md#2-training-from-image-observation-example)
    - [3. Training from image observation with 20 demo trajectories example](/docs/sim_quick_start.md#3-training-from-image-observation-with-20-demo-trajectories-example)
  - [Run with Franka Arm on Real Robot](/docs/real_franka_peg.md#run-with-franka-arm-on-real-robot)
    - [1. Peg Insertion 📍](/docs/real_franka_peg.md#peg-insertion-📍)
      - [Detailed Procedure](/docs/real_franka_peg.md#procedure)
    - [2. PCB Component Insertion 🖥️](/docs/real_franka_pcb.md)
    - [3. Cable Routing 🔌](/docs/real_franka_cable_route.md)
    - [4. Object Relocation 🗑️](/docs/real_franka_bin_relocation.md)
  - [Contribution](#contribution)
  - [Citation](#citation)

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

## Contribution

We welcome contributions to this repository! Fork and submit a PR if you have any improvements to the codebase. Before submitting a PR, please run `pre-commit run --all-files` to ensure that the codebase is formatted correctly.

## Citation

If you use this code for your research, please cite our paper:

```bibtex
@misc{luo2024serl,
      title={SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning},
      author={Jianlan Luo and Zheyuan Hu and Charles Xu and You Liang Tan and Jacob Berg and Archit Sharma and Stefan Schaal and Chelsea Finn and Abhishek Gupta and Sergey Levine},
      year={2024},
      eprint={2401.16013},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
