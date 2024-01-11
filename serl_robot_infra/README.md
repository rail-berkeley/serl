# SERL Robot Infra

All robot code is structured as follows:
There is a Flask server which sends commands to the robot via ROS. There is a gym env for the robot which communicates with the Flask server via post requests.

- `robot_server`: hosts a Flask server which sends commands to the robot via ROS
- `franka_env`: gym env for the robot which communicates with the Flask server via post requests

### Installation

First, make sure the NUC meets the specifications [here](https://frankaemika.github.io/docs/requirements.html), and install the real time kernel, and `libfranka` and `franka_ros` as described [here](https://frankaemika.github.io/docs/installation_linux.html).

You'll then want to copy the following files from `launchers` to your catkin workspace:
- copy the two `.launch` files to a `catkin_ws/scripts` folder in your ros workspace
- copy the `.cfg` files to `catkin_ws/src/franka_ros/franka_example_controllers/cfg`
- copy the two `.cpp` files to `catkin_ws/src/franka_ros/franka_example_controllers/src`
- copy the two `.h` files to `catkin_ws/src/franka_ros/franka_example_controllers/include/franka_example_controllers`

### Usage

To start using the robot, first power on the robot (small switch on the back of robot control box on the floor). Unlock the robot from the browser interface by going to robot IP address in your browser, then press the black and white button to put the robot in FCI control mode (blue light).

From there you should be able to navigate to `robot_infra` and then simply run `python franka_server.py`. This should start the impedence controller and the HTTP server. You can test that things are running by trying to move the end effector around, if the impedence controller is running it should be compliant.

Lastly, any code you write can interact with the robot via the gym interface defined in this repo under `env`. Simply run `pip install -e .` in the `robot_infra` directory, and in your code simply initialize the env via `gym.make("Franka-{ENVIRONMENT NAME}-v0)`.
