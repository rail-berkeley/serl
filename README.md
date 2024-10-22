# Voxel SERL

## Contributions

| Code Directory                                                                                             | Description                                |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| [robot_controllers](https://github.com/nisutte/voxel-serl/tree/develop/serl_robot_infra/robot_controllers) | Impedance controller for the UR5 robot arm |
| [box_picking_env](https://github.com/nisutte/voxel-serl/tree/develop/serl_robot_infra/box_picking_env)     | Environment setup for the box picking task |
| [vision](https://github.com/nisutte/voxel-serl/tree/develop/serl_launcher/serl_launcher/vision)            | Point-Cloud based encoders                 |
| [utils](https://github.com/nisutte/voxel-serl/blob/develop/serl_robot_infra/ur_env/camera/utils.py)        | Point-Cloud fusion and voxelization        |

## Quick start guide for box picking with a UR5 robot arm

### Without cameras (TODO modify the bash files)

1. Follow the installation in the official [SERL repo](https://github.com/rail-berkeley/serl).
2. Check [envs](https://github.com/nisutte/voxel-serl/blob/develop/serl_robot_infra/ur_env/envs) and either use the provided [box_picking_env](https://github.com/nisutte/voxel-serl/blob/develop/serl_robot_infra/ur_env/envs/camera_env/box_picking_camera_env.py) or set up a new environment using the one mentioned as a template. (New environments have to be registered [here](https://github.com/nisutte/voxel-serl/blob/develop/serl_robot_infra/ur_env/__init__.py))
2. Use the [config](https://github.com/nisutte/voxel-serl/blob/develop/serl_robot_infra/ur_env/envs/camera_env/config.py) file to configure all the robot-arm specific parameters, as well as gripper and camera infos.
3. Go to the [box picking](https://github.com/nisutte/voxel-serl/blob/develop/examples/box_picking_drq) folder and modify the bash files ```run_learner.py``` and ```run_actor.py```. If no images are used, set ```camera_mode``` to ```none``` . WandB logging can be deactivated if ```debug``` is set to True.
4. Record 20 demostrations using [record_demo.py](https://github.com/nisutte/voxel-serl/blob/develop/examples/box_picking_drq/record_demo.py) in the same folder. Double check that the ```camera_mode``` and all environment-wrappers are identical to [drq_policy.py](https://github.com/nisutte/voxel-serl/blob/develop/examples/box_picking_drq/drq_policy.py).
5. Update the demonstrations path in the ```run_learner.py``` and execute it and ```run_actor.py``` simultaneously to start the RL training.
6. To evaluate on a policy, modify and execute ```run_evaluation.py``` with the specified checkpoint path and step. 

## Modaliy examples
<p>
  <img src="./docs/images/trajectory%20timeline.png" width="50%"/>
</p>

## TODO's
- [ ] improve readme
- [ ] add paper link
- [ ] document how to use in a real setting

## Notes
- clu needs to be installed for a parameter overview of the model