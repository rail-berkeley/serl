"""
This file starts a control server running on the real time PC connected to the franka robot.
In a screen run `python franka_server.py`
"""
import os
import sys
from flask import Flask, request, jsonify
import numpy as np
import rospy
import time
import subprocess
from scipy.spatial.transform import Rotation as R

from sensor_msgs.msg import JointState
from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState, ZeroJacobian
import geometry_msgs.msg as geom_msg
from dynamic_reconfigure.client import Client



app = Flask(__name__)
RESET_JOINT_TARGET = [-0.07, -0.1, 0.0, -2.5, -0.1, 2.5, -0.6]
ROBOT_IP = "172.16.0.2"
GRIPPER_IP = None
GRIPPER_TYPE = "Franka" # "Robotiq", "Franka", or "None"

if GRIPPER_TYPE == "Robotiq":
    from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
    assert GRIPPER_IP != None
elif GRIPPER_TYPE == "Franka":
    from franka_gripper.msg import GraspActionGoal, MoveActionGoal
elif GRIPPER_TYPE == "None":
    pass
else:
    raise NotImplementedError("Gripper Type Not Implemented")



class RobotServer:
    """Handles the starting and stopping of the impedence controller
    (as well as backup) joint recovery policy."""

    def __init__(self):
        self.eepub = rospy.Publisher(
            "/cartesian_impedance_controller/equilibrium_pose",
            geom_msg.PoseStamped,
            queue_size=10,
        )
        self.resetpub = rospy.Publisher(
            "/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, queue_size=1
        )
        self.state_sub = rospy.Subscriber(
            "franka_state_controller/franka_states", FrankaState, self.set_currpos
        )
        self.jacobian_sub = rospy.Subscriber(
            "/cartesian_impedance_controller/franka_jacobian",
            ZeroJacobian,
            self.set_jacobian,
        )

    def start_impedence(self):
        """Launches the impedence controller"""
        self.imp = subprocess.Popen(
            [
                "roslaunch",
                "serl_franka_controllers",
                "impedence.launch",
                "robot_ip:=" + ROBOT_IP,
                f"load_gripper:={'true' if GRIPPER_TYPE == 'Franka' else 'false'}",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(5)

    def stop_impedence(self):
        """Stops the impedence controller"""
        self.imp.terminate()
        time.sleep(1)

    def clear(self):
        """Clears any errors"""
        msg = ErrorRecoveryActionGoal()
        self.resetpub.publish(msg)

    def set_currpos(self, msg):
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3])
        pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])
        self.pos = pose
        self.dq = np.array(list(msg.dq)).reshape((7,))
        self.q = np.array(list(msg.q)).reshape((7,))
        self.force = np.array(list(msg.O_F_ext_hat_K)[:3])
        self.torque = np.array(list(msg.O_F_ext_hat_K)[3:])
        self.vel = self.jacobian @ self.dq

    def set_jacobian(self, msg):
        jacobian = np.array(list(msg.zero_jacobian)).reshape((6, 7), order="F")
        self.jacobian = jacobian

    def reset_joint(self):
        """Resets Joints (needed after running for hours)"""
        # First Stop Impedence
        try:
            self.stop_impedence()
            self.clear()
        except:
            print("Impedence Not Running")
        time.sleep(3)
        self.clear()
        
        # Launch joint controller reset
        rospy.set_param("/target_joint_positions", RESET_JOINT_TARGET)
        self.j = subprocess.Popen(
            [
                "roslaunch",
                "serl_franka_controllers",
                "joint.launch",
                "robot_ip:=" + ROBOT_IP,
                f"load_gripper:={'true' if GRIPPER_TYPE == 'Franka' else 'false'}",
            ],
            stdout=subprocess.PIPE,
        )
        time.sleep(1)
        print("RUNNING JOINT RESET")
        self.clear()
        
        # Wait until target joint angles are reached
        count = 0
        time.sleep(1)
        while not np.allclose(
            np.array(RESET_JOINT_TARGET) - np.array(self.q), 0, atol=1e-2, rtol=1e-2
        ):
            time.sleep(1)
            count += 1
            if count > 100:
                break
            
        # Stop joint controller
        print("RESET DONE")
        self.j.terminate()
        time.sleep(1)
        self.clear()
        print("KILLED JOINT RESET", self.pos)
        
        # Restart impedece controller
        self.start_impedence()
        print("IMPEDENCE STARTED")

    def move(self, pose):
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(pose[0], pose[1], pose[2])
        msg.pose.orientation = geom_msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
        self.eepub.publish(msg)
    
    def open(self):
        pass
    
    def close(self):
        pass
    
    def activate_gripper(self):
        pass
    
    def reset_gripper(self):
        pass

class FrankaGripperServer(RobotServer):
    def __init__(self):
        super().__init__()
        self.grippermovepub = rospy.Publisher(
            "/franka_gripper/move/goal", MoveActionGoal, queue_size=1
        )
        self.grippergrasppub = rospy.Publisher(
            "/franka_gripper/grasp/goal", GraspActionGoal, queue_size=1
        )
        self.gripper_sub = rospy.Subscriber(
            "/franka_gripper/joint_states", JointState, self.update_gripper
        )

    def open(self):
        msg = MoveActionGoal()
        msg.goal.width = 0.09
        msg.goal.speed = 0.3
        self.grippermovepub.publish(msg)

    def close(self):
        msg = GraspActionGoal()
        msg.goal.width = 0.01
        msg.goal.speed = 0.3
        msg.goal.epsilon.inner = 1
        msg.goal.epsilon.outer = 1
        msg.goal.force = 130
        self.grippergrasppub.publish(msg)

    def update_gripper(self, msg):
        self.gripper_dist = np.sum(msg.position)
        
class RobotiqServer(RobotServer):
    def __init__(self):
        super().__init__()
        self.gripper = subprocess.Popen(
            [
                "rosrun",
                "robotiq_2f_gripper_control",
                "Robotiq2FGripperTcpNode.py",
                GRIPPER_IP,
            ],
            stdout=subprocess.PIPE,
        )
        self.gripperpub = rospy.Publisher(
            "Robotiq2FGripperRobotOutput", outputMsg.Robotiq2FGripper_robot_output, queue_size=1
        )
        self.gripper_command = outputMsg.Robotiq2FGripper_robot_output()
        
    def generate_gripper_command(self, char, command):
        """Update the gripper command according to the character entered by the user."""
        if char == "a":
            command = outputMsg.Robotiq2FGripper_robot_output()
            command.rACT = 1
            command.rGTO = 1
            command.rSP = 255
            command.rFR = 150

        if char == "r":
            command = outputMsg.Robotiq2FGripper_robot_output()
            command.rACT = 0

        if char == "c":
            command.rPR = 255

        if char == "o":
            command.rPR = 0

        # If the command entered is a int, assign this value to rPR 
        # (i.e., move to this position)
        try:
            command.rPR = int(char)
            if command.rPR > 255:
                command.rPR = 255
            if command.rPR < 0:
                command.rPR = 0
        except ValueError:
            pass
        return command
    
    def activate_gripper(self):
        self.gripper_command = self.generate_gripper_command("a", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)
    
    def reset_gripper(self):
        self.gripper_command = self.generate_gripper_command("r", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)
        self.activate_gripper()
    
    def open(self):
        self.gripper_command = self.generate_gripper_command("o", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)

    def close(self):
        self.gripper_command = self.generate_gripper_command("c", self.gripper_command)
        self.gripperpub.publish(self.gripper_command)

    def update_gripper(self, msg):
        raise NotImplementedError("Not implemented for Robotiq Gripper")



try:
    roscore = subprocess.Popen("roscore")
    time.sleep(1)
except:
    pass

"""Starts Impedence controller"""
if GRIPPER_TYPE == "Franka":
    l = FrankaGripperServer()
elif GRIPPER_TYPE == "Robotiq":
    l = RobotiqServer()
elif GRIPPER_TYPE == "None":
    l = RobotServer()
else:
    raise NotImplementedError("Gripper Type Not Implemented")

l.start_impedence()
rospy.init_node("franka_control_api")

## Defines the ros topics to publish to
client = Client(
    "/cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node"
)

## Route for Starting Impedence
@app.route("/startimp", methods=["POST"])
def si():
    l.clear()
    l.start_impedence()
    return "Started Impedence"

## Route for Stopping Impedence
@app.route("/stopimp", methods=["POST"])
def sti():
    l.stop_impedence()
    return "Stopped Impedence"

## Route for getting pose
@app.route("/getpos", methods=["POST"])
def gp():
    return jsonify({"pose": np.array(l.pos).tolist()})

## Route for getting velocity
@app.route("/getvel", methods=["POST"])
def gv():
    return jsonify({"vel": np.array(l.vel).tolist()})

## Route for getting force
@app.route("/getforce", methods=["POST"])
def gf():
    return jsonify({"force": np.array(l.force).tolist()})

## Route for getting torque
@app.route("/gettorque", methods=["POST"])
def gt():
    return jsonify({"torque": np.array(l.torque).tolist()})

## Route for getting joint angles
@app.route("/getq", methods=["POST"])
def gq():
    return jsonify({"q": np.array(l.q).tolist()})

## Route for getting joint velocities
@app.route("/getdq", methods=["POST"])
def gdq():
    return jsonify({"dq": np.array(l.dq).tolist()})

## Route for getting jacobian
@app.route("/getjacobian", methods=["POST"])
def gj():
    return jsonify({"jacobian": np.array(l.jacobian).tolist()})

## Route for getting gripper distance
@app.route("/getgripper", methods=["POST"])
def gg():
    return jsonify({"gripper": l.gripper_dist})

## Route for getting all state information
@app.route("/getstate", methods=["POST"])
def gs():
    return jsonify(
        {
            "pose": np.array(l.pos).tolist(),
            "vel": np.array(l.vel).tolist(),
            "force": np.array(l.force).tolist(),
            "torque": np.array(l.torque).tolist(),
            "q": np.array(l.q).tolist(),
            "dq": np.array(l.dq).tolist(),
            "jacobian": np.array(l.jacobian).tolist(),
            "gripper": l.gripper_dist,
        }
    )

## Route for running joint reset
@app.route("/jointreset", methods=["POST"])
def jr():
    l.clear()
    l.reset_joint()
    return "Reset Joint"

##Route for activating the Robotiq gripper
@app.route("/activate_gripper", methods=["POST"])
def activate_gripper():
    print("activate gripper")
    l.activate_gripper()
    return "Activated"

## Route for resetting the Robotiq gripper. It will reset and activate the gripper
@app.route("/reset_gripper", methods=["POST"])
def reset_gripper():
    print("reset gripper")
    l.reset_gripper()
    return "Reset"

## Route for closing the gripper
@app.route("/close", methods=["POST"])
def closed():
    print("close")
    l.close()
    return "Closed"

## Route for opening the gripper
@app.route("/open", methods=["POST"])
def open():
    print("open")
    l.open()
    return "Opened"


## Route for clearing errors (communcation constraints, etc.)
@app.route("/clearerr", methods=["POST"])
def clear():
    l.clear()
    return "Clear"


## Route for sending a pose command
@app.route("/pose", methods=["POST"])
def pose():
    pos = np.array(request.json["arr"])
    print("Moving to", pos)
    l.move(pos)
    return "Moved"


## Route for increasing controller gain
@app.route("/precision_mode", methods=["POST"])
def precision_mode():
    client.update_configuration({"translational_Ki": 30})
    client.update_configuration({"rotational_Ki": 10})
    for direction in ["x", "y", "z", "neg_x", "neg_y", "neg_z"]:
        client.update_configuration({"translational_clip_" + direction: 0.1})
        client.update_configuration({"rotational_clip" + direction: 0.1})
    return "Precision"

## Route for decreasing controller gain
@app.route("/compliance_mode", methods=["POST"])
def compliance_mode():
    client.update_configuration({"translational_Ki": 10})
    client.update_configuration({"rotational_Ki": 0})
    for direction in ["x", "y", "z", "neg_x", "neg_y", "neg_z"]:
        client.update_configuration({"translational_clip" + direction: 0.007})
        client.update_configuration({"rotational_clip" + direction: 0.04})
    return "Compliance"

## Route for updating custom controller parameters
@app.route("/update_params", methods=["POST"])
def update_parameter():
    client.update_configuration(request.json)
    return "Updated Controller Parameters"


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0")
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
        sys.exit()
