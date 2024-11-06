import time

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np

from geometry_msgs.msg import Pose, Point, Quaternion
import rclpy.task
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from control_msgs.action import FollowJointTrajectory
import os
import sys
from moveit_msgs.msg import (
    RobotState,
    RobotTrajectory,
    MoveItErrorCodes,
    Constraints,
    JointConstraint,
)
from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetPositionFK
from moveit_msgs.action import ExecuteTrajectory
from kuka_server.wait_for_message import wait_for_message


from kuka_server.utils import quat_to_euler, convert_wrench_to_numpy, euler_to_quat
from lbr_fri_idl.msg import LBRState
import copy

class RobotInterfaceNode(Node):
    timeout_sec_ = 5.0

    move_group_name_ = "arm"
    namespace_ = "lbr"

    joint_state_topic_ = "joint_states"
    lbr_state_topic_ = "state"
    plan_srv_name_ = "plan_kinematic_path"
    ik_srv_name_ = "compute_ik"
    fk_srv_name_ = "compute_fk"
    execute_action_name_ = "execute_trajectory"
    fri_execute_action_name_ = "joint_trajectory_controller/follow_joint_trajectory"
    WRENCH_TOPIC = "/lbr/force_torque_broadcaster/wrench"

    base_ = "link_0"  ###Changed for compatibility with latest lbr stack
    end_effector_ = "link_ee"

    def __init__(self) -> None:
        super().__init__("robot_interface_node", namespace=self.namespace_)

        self.fk_client_callback = ReentrantCallbackGroup()
        self.ik_client_callback = MutuallyExclusiveCallbackGroup()
        self.plan_client_callback = MutuallyExclusiveCallbackGroup()
        self.execute_client_callback = MutuallyExclusiveCallbackGroup()

        self.robot_state = {}

        self.wrench_sub = self.create_subscription(
            WrenchStamped,
            self.WRENCH_TOPIC,
            self.wrench_callback,
            10
        )

        self.ik_client_ = self.create_client(GetPositionIK, self.ik_srv_name_, callback_group=self.ik_client_callback)
        if not self.ik_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("IK service not available.")
            exit(1)

        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_, callback_group=self.fk_client_callback)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)

        self.plan_client_ = self.create_client(GetMotionPlan, self.plan_srv_name_, callback_group=self.plan_client_callback)
        if not self.plan_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("Plan service not available.")
            exit(1)

        self.execute_client_ = ActionClient(
            self, ExecuteTrajectory, self.execute_action_name_, callback_group=self.execute_client_callback
        )
        if not self.execute_client_.wait_for_server(timeout_sec=self.timeout_sec_):
            self.get_logger().error("Execute action not available.")
            exit(1)

        self.fri_execute_client_ = ActionClient(
            self,
            FollowJointTrajectory,
            self.fri_execute_action_name_,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        if not self.fri_execute_client_.wait_for_server(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FRI Execute action not available.")
            exit(1)

    def get_ik(self, target_pose: Pose) -> JointState | None:
        request = GetPositionIK.Request()

        request.ik_request.group_name = self.move_group_name_
        tf_prefix = self.get_namespace()[1:]
        request.ik_request.pose_stamped.header.frame_id = f"{tf_prefix}/{self.base_}"
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose = target_pose
        request.ik_request.avoid_collisions = True

        future = self.ik_client_.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to get IK solution")
            return None

        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None

        return response.solution.joint_state
    
    def wrench_callback(self, msg):
        self.latest_wrench = convert_wrench_to_numpy(msg)
        # self.get_logger().info(f"Received wrench data: {self.latest_wrench}")
        return


    def get_fk(self) -> Pose | None:
        current_joint_state = self.get_joint_state()
        if current_joint_state is None:
            self.get_logger().error("Failed to get joint state")
            return None

        current_robot_state = RobotState()
        current_robot_state.joint_state = current_joint_state

        request = GetPositionFK.Request()
        self.get_logger().info(f"{self.namespace_}/{self.base_}")
        request.header.frame_id = f"{self.namespace_}/{self.base_}"
        request.header.stamp = self.get_clock().now().to_msg()

        request.fk_link_names.append(self.end_effector_)
        request.robot_state = current_robot_state

        future = self.fk_client_.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to get FK solution")
            return None
        
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"Failed to get FK solution: {response.error_code.val}"
            )
            return None
        
        return response.pose_stamped[0].pose, current_joint_state

    def get_fk_lbr(self, commanded:bool = False) -> Pose | None:
        lbr_state_set, lbr_state = wait_for_message(LBRState, self, self.lbr_state_topic_)
        joint_position = lbr_state.measured_joint_position.tolist()
        if commanded:
            joint_position = lbr_state.commanded_joint_position.tolist()
        print(joint_position)
        joint_position[2], joint_position[3] = joint_position[3], joint_position[2]

        current_robot_state = RobotState()
        current_robot_state.joint_state = self.get_joint_state()
        current_robot_state.joint_state.position = joint_position

        request = GetPositionFK.Request()

        request.header.frame_id = f"{self.namespace_}/{self.base_}"
        request.header.stamp = self.get_clock().now().to_msg()

        request.fk_link_names.append(self.end_effector_)
        request.robot_state = current_robot_state

        future = self.fk_client_.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to get FK solution")
            return None
        
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"Failed to get FK solution: {response.error_code.val}"
            )
            return None
        
        return response.pose_stamped[0].pose

    def sum_of_square_diff(
        self, joint_state_1: JointState, joint_state_2: JointState
    ) -> float:
        return np.sum(
            np.square(np.subtract(joint_state_1.position, joint_state_2.position))
        )

    def get_current_state(self):
        ##Return a dictionary with all the corresponding state variables
        #Pose, Velocity, Force, Torque, Jacobian, Joint Angles, Joint Velocity, Gripper Pose
        self.robot_state["pose"] = np.zeros((6,))
        self.robot_state["vel"] = np.zeros((6,))
        self.robot_state["force"] = np.zeros((3,))
        self.robot_state["torque"] = np.zeros((3,))
        self.robot_state["q"] = np.zeros((7,))
        self.robot_state["dq"] = np.zeros((7,))

        ##TCP pose
        current_ee_geom_pose, current_joint_state = self.get_fk()
        self.robot_state["pose"][:3] = np.array([current_ee_geom_pose.position.x,current_ee_geom_pose.position.y,current_ee_geom_pose.position.z])
        euler_angles = quat_to_euler( np.array([current_ee_geom_pose.orientation.x,
                                               current_ee_geom_pose.orientation.y,
                                               current_ee_geom_pose.orientation.z,
                                               current_ee_geom_pose.orientation.w]))
        self.robot_state["pose"][3:] = euler_angles
        
        ##Getting Joint Angles
        joint_angles = current_joint_state.position
        self.robot_state["q"] = joint_angles
        joint_velocity = current_joint_state.velocity
        self.robot_state["dq"] = joint_velocity

        ##Getting Force and Wrench
        current_wrench = copy.deepcopy(self.latest_wrench)
        self.robot_state["force"] = current_wrench[:3]
        self.robot_state["torque"] = current_wrench[3:]

        


        return

    def move_to_pose(self, pose:np.ndarray):

        target_pose = Pose()
        
        target_pose.position.x = pose[0]
        target_pose.position.y = pose[1]
        target_pose.position.z = pose[2]

        target_pose_quaternion = euler_to_quat(pose[3:])
        target_pose.orientation.x = target_pose_quaternion[0]
        target_pose.orientation.y = target_pose_quaternion[1]
        target_pose.orientation.z = target_pose_quaternion[2]
        target_pose.orientation.w = target_pose_quaternion[3]


        traj = self.get_motion_plan(target_pose, True)
        if traj:
            client = self.get_motion_execute_client()
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = traj

            future = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Failed to execute trajectory")
            else:
                self.get_logger().info("Trajectory accepted")

            
            result_future = goal_handle.get_result_async()

            expect_duration = traj.joint_trajectory.points[-1].time_from_start
            expect_time = time.time() + 2 * expect_duration.sec
            while not result_future.done() and time.time() < expect_time:
                time.sleep(0.01)

            self.get_logger().info("Trajectory executed")
        
            self.get_logger().info("Current pose: "  + str(self.get_fk()[0]) )


        return



    def get_best_ik(self, target_pose: Pose, attempts: int = 100) -> JointState | None:
        current_joint_state = self.get_joint_state()
        if current_joint_state is None:
            self.get_logger().error("Failed to get joint state")
            return None

        best_cost = np.inf
        best_joint_state = None

        for _ in range(attempts):
            joint_state = self.get_ik(target_pose)
            if joint_state is None:
                continue

            cost = self.sum_of_square_diff(current_joint_state, joint_state)
            if cost < best_cost:
                best_cost = cost
                best_joint_state = joint_state

        if not best_joint_state:
            self.get_logger().error("Failed to get IK solution")

        return best_joint_state

    def get_joint_state(self) -> JointState:
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_
        )
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None
        
        return current_joint_state

    def get_motion_plan(
        self, target_pose: Pose, linear: bool = False, scaling_factor: float = .1, attempts: int = 10
    ) -> RobotTrajectory | None:
        current_pose = self.get_fk()[0]
        if current_pose is None:
            self.get_logger().error("Failed to get current pose")

        # if check_same_pose(current_pose, target_pose):
        #     self.get_logger().info("Detected same pose as current pose, neglected")
        #     return RobotTrajectory()

        current_joint_state = self.get_joint_state()
        if current_joint_state is None:
            self.get_logger().error("Failed to get joint state")
            return None

        current_robot_state = RobotState()
        current_robot_state.joint_state.position = current_joint_state.position

        target_joint_state = self.get_best_ik(target_pose)
        if target_joint_state is None:
            self.get_logger().error("Failed to get target joint state")
            return None

        target_constraint = Constraints()
        for i in range(len(target_joint_state.position)):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = target_joint_state.name[i]
            joint_constraint.position = target_joint_state.position[i]
            joint_constraint.tolerance_above = 0.001
            joint_constraint.tolerance_below = 0.001
            joint_constraint.weight = 1.0
            target_constraint.joint_constraints.append(joint_constraint)

        request = GetMotionPlan.Request()
        request.motion_plan_request.group_name = self.move_group_name_
        request.motion_plan_request.start_state = current_robot_state
        request.motion_plan_request.goal_constraints.append(target_constraint)
        request.motion_plan_request.num_planning_attempts = 10
        request.motion_plan_request.allowed_planning_time = 5.0
        request.motion_plan_request.max_velocity_scaling_factor = scaling_factor
        request.motion_plan_request.max_acceleration_scaling_factor = scaling_factor

        if linear:
            request.motion_plan_request.pipeline_id = "pilz_industrial_motion_planner"
            request.motion_plan_request.planner_id = "LIN"
        else:
            request.motion_plan_request.pipeline_id = "ompl"
            request.motion_plan_request.planner_id = "APSConfigDefault"

        for _ in range(attempts):
            plan_future = self.plan_client_.call_async(request)
            rclpy.spin_until_future_complete(self, plan_future)

            if plan_future.result() is None:
                self.get_logger().error("Failed to get motion plan")

            response = plan_future.result()
            if response.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error(
                    f"Failed to get motion plan: {response.motion_plan_response.error_code.val}"
                )
            else:
                return response.motion_plan_response.trajectory
            
        return None

    def get_motion_execute_client(self) -> ActionClient:
        return self.execute_client_
    
    def get_fri_motion_execute(self, traj: RobotTrajectory, short_wait=False) -> bool:
        joint_trajectory_goal = FollowJointTrajectory.Goal()
        joint_trajectory_goal.trajectory = traj.joint_trajectory

        goal_future = self.fri_execute_client_.send_goal_async(joint_trajectory_goal)
        rclpy.spin_until_future_complete(self, goal_future)
        goal_handle = goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            return False
        self.get_logger().info("Trajectory goal accepted")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        execution_finished = False
        expected_timout = time.time() + 30
        if short_wait: 
            expected_timout = time.time() + 2.0     
        while not execution_finished and time.time() < expected_timout:
            _, lbr_state = wait_for_message(LBRState, self, self.lbr_state_topic_)
            if np.max(np.abs(np.subtract(lbr_state.measured_joint_position, traj.joint_trajectory.points[-1].positions))) < 0.0002:
                execution_finished = True

            time.sleep(0.01)

        self.get_logger().info("Trajectory executed")
        return True


def main(args=None):
    rclpy.init(args=args)

    robot_interface_node = RobotInterfaceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(robot_interface_node)
    robot_interface_node.get_logger().info("Robot interface node started.")

    # target_poses = []
    # for i in range(3):
    #     target_poses.append(
    #         Pose(
    #             position=Point(x=0.5, y=-0.1 + 0.1 * i, z=0.6),
    #             orientation=Quaternion(x=0.0, y=-1.0, z=0.0, w=0.0),
    #         )
    #     )

    # traj = robot_interface_node.get_motion_plan(target_poses[1])
    # if traj:
    #     client = robot_interface_node.get_motion_execute_client()
    #     goal = ExecuteTrajectory.Goal()
    #     goal.trajectory = traj

    #     future = client.send_goal_async(goal)
    #     rclpy.spin_until_future_complete(robot_interface_node, future)
        
    #     goal_handle = future.result()
    #     if not goal_handle.accepted:
    #         robot_interface_node.get_logger().error("Failed to execute trajectory")
    #     else:
    #         robot_interface_node.get_logger().info("Trajectory accepted")

    #     result_future = goal_handle.get_result_async()

    #     expect_duration = traj.joint_trajectory.points[-1].time_from_start
    #     expect_time = time.time() + 2 * expect_duration.sec 
    #     while not result_future.done() and time.time() < expect_time:
    #         time.sleep(0.01)

    #     robot_interface_node.get_logger().info("Trajectory executed")

    #     robot_interface_node.get_logger().info("Current pose: " + str(robot_interface_node.get_fk()[0])) 

    # for target_pose in target_poses:
    #     traj = robot_interface_node.get_motion_plan(target_pose, True)
    #     if traj:
    #         client = robot_interface_node.get_motion_execute_client()
    #         goal = ExecuteTrajectory.Goal()
    #         goal.trajectory = traj

    #         future = client.send_goal_async(goal)
    #         rclpy.spin_until_future_complete(robot_interface_node, future)
            
    #         goal_handle = future.result()
    #         if not goal_handle.accepted:
    #             robot_interface_node.get_logger().error("Failed to execute trajectory")
    #         else:
    #             robot_interface_node.get_logger().info("Trajectory accepted")

            
    #         result_future = goal_handle.get_result_async()

    #         expect_duration = traj.joint_trajectory.points[-1].time_from_start
    #         expect_time = time.time() + 2 * expect_duration.sec
    #         while not result_future.done() and time.time() < expect_time:
    #             time.sleep(0.01)

    #         robot_interface_node.get_logger().info("Trajectory executed")
        
    #         robot_interface_node.get_logger().info("Current pose: "  + str(robot_interface_node.get_fk()[0]) )

    # rclpy.spin(robot_interface_node)
    robot_interface_node.get_current_state()
    
    print("Current State: ", robot_interface_node.robot_state)
    executor.spin()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
