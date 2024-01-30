# Object Relocation 🗑️

![](./images/forward.png)

![](./images/backward.png)

> Example is located in [examples/async_bin_relocation_fwbw_drq/](../examples/async_bin_relocation_fwbw_drq/)

> Env and default config are located in `serl_robot_infra/franka_env/envs/bin_env/`

This bin relocation example demonstrates the usage of forward and backward policies. This is helpful for RL tasks, which require the robot to "reset". In this case, the robot is moving an object from one bin to another. The forward policy is used to move the object from the right bin to the left bin, and the backward policy is used to move the object from the left bin to the right bin.

1. Record demo trajectories

Multiple utility scripts have been provided to record demo trajectories. (e.g. `record_demo.py`: for RLPD, `record_transitions.py`: for reward classifier, `reward_bc_demos.py`: for bc policy). Note that both forward and backward trajectories require different demo trajectories.

2. Reward Classifier

Similar to the cable routing example, we need to train two reward classifiers for both forward and backward policies, shown in `train_fwd_reward_classifier.sh` and `train_bwd_reward_classifier.sh`. The reward classifiers are then used in the BC and DRQ policy for the actor node, checkpoint path is provided as `--reward_classifier_ckpt_path` argument in `run_bc.sh` and `run_actor.sh`.

3. Run 2 learners and 1 actor with 2 policies

Finally, 2 learner nodes will learn both forward and backward policies respectively. The actor node will switch between running the forward and backward policies with their respective reward classifiers during the RL training process.

```bash
bash run_actor.sh

# run 2 learners
bash run_fw_learner.sh
bash run_bw_learner.sh
```
