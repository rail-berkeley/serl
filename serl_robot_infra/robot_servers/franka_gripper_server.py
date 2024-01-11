import rospy
from franka_gripper.msg import GraspActionGoal, MoveActionGoal
from sensor_msgs.msg import JointState
import numpy as np

from robot_servers.gripper_server import GripperServer


class FrankaGripperServer(GripperServer):
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
        self.gripper_pos = np.sum(msg.position)
