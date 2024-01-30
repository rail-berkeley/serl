## Overview and Code Structure

SERL provides a set of common libraries for users to train RL policies for robotic manipulation tasks. The main structure of running the RL experiments involves having an actor node and a learner node, both of which interact with the robot gym environment. Both nodes run asynchronously, with data being sent from the actor to the learner node via the network using [agentlace](https://github.com/youliangtan/agentlace). The learner will periodically synchronize the policy with the actor. This design provides flexibility for parallel training and inference.

<p align="center">
  <img src="./images/software_design.png" width="80%"/>
</p>

**Table for code structure**

| Code Directory | Description |
| --- | --- |
| [serl_launcher](../serl_launcher) | Main code for SERL |
| [serl_launcher.agents](../serl_launcher/serl_launcher/agents/) | Agent Policies (e.g. DRQ, SAC, BC) |
| [serl_launcher.wrappers](../serl_launcher/serl_launcher/wrappers) | Gym env wrappers |
| [serl_launcher.data](../serl_launcher/serl_launcher/data) | Replay buffer and data store |
| [serl_launcher.vision](../serl_launcher/serl_launcher/vision) | Vision related models and utils |
| [franka_sim](../franka_sim) | Franka mujoco simulation gym environment |
| [serl_robot_infra](../serl_robot_infra/) | Robot infra for running with real robots |
| [serl_robot_infra.robot_servers](../serl_robot_infra/robot_servers/) | Flask server for sending commands to robot via ROS |
| [serl_robot_infra.franka_env](../serl_robot_infra/franka_env/) | Gym env for real franka robot |
