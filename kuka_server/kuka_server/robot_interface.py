import time

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import GetParameters
import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
import rclpy.task
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import os
import sys
import socket
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
from moveit_configs_utils.launches import generate_move_group_launch

from kuka_server.utils import quat_to_euler, convert_wrench_to_numpy, euler_to_quat
from lbr_fri_idl.msg import LBRState
import copy
import optas


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
    fri_execute_action_name_ = "/lbr/joint_trajectory_controller/follow_joint_trajectory"
    WRENCH_TOPIC = "/lbr/force_torque_broadcaster/wrench"
    robot_desc_topic_ = "robot_description"
    port_id = 30200
    robot_ip = "192.168.10.122"
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
            WrenchStamped, self.WRENCH_TOPIC, self.wrench_callback, 10
        )

        self.state_sub = self.create_subscription(
            LBRState, self.lbr_state_topic_, self.state_callback, 100
        )

        self._init_jacobian()
        self.session_ok = True

        self.ik_client_ = self.create_client(
            GetPositionIK, self.ik_srv_name_, callback_group=self.ik_client_callback
        )
        if not self.ik_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("IK service not available.")
            exit(1)

        self.fk_client_ = self.create_client(
            GetPositionFK, self.fk_srv_name_, callback_group=self.fk_client_callback
        )
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)

        self.plan_client_ = self.create_client(
            GetMotionPlan, self.plan_srv_name_, callback_group=self.plan_client_callback
        )
        if not self.plan_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("Plan service not available.")
            exit(1)

        self.execute_client_ = ActionClient(
            self,
            ExecuteTrajectory,
            self.execute_action_name_,
            callback_group=self.execute_client_callback,
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

    def state_callback(self, state: LBRState):
        self.session_state = state.session_state
        external_torque = state.external_torque
        # if(np.any(np.abs(external_torque)>10) or self.session_state!=4):
        if self.session_state != 4:
            self.get_logger().info(
                f"Torque Exceeded or State Change to {self.session_state}"
            )
            self.session_ok = False

        return

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

    def _init_jacobian(self):
        print("Initializing Jacobian")

        self._robot_description = self._retrieve_parameter(
            "/lbr/robot_state_publisher/get_parameters", "robot_description"
        ).string_value
        self._robot = optas.RobotModel(
            urdf_string=self._robot_description, time_derivs=[0, 1]
        )

        self.jacobian_func = self._robot.get_link_geometric_jacobian_function(
            link=self.end_effector_, base_link=self.base_, numpy_output=True
        )

        return

    def get_fk(self) -> Pose | None:
        current_joint_state = self.get_joint_state()
        if current_joint_state is None:
            self.get_logger().error("Failed to get joint state")
            return None

        current_robot_state = RobotState()
        current_robot_state.joint_state = current_joint_state

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

        # print(
        #     "In get_fk, printing the current pose after fk: ",
        #     response.pose_stamped[0].pose,
        # )
        return response.pose_stamped[0].pose, current_joint_state

    def get_fk_lbr(self, commanded: bool = False) -> Pose | None:
        lbr_state_set, lbr_state = wait_for_message(
            LBRState, self, self.lbr_state_topic_
        )
        joint_position = lbr_state.measured_joint_position.tolist()
        if commanded:
            joint_position = lbr_state.commanded_joint_position.tolist()

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

    def _retrieve_parameter(self, service: str, parameter_name: str) -> ParameterValue:
        parameter_client = self.create_client(GetParameters, service)
        while not parameter_client.wait_for_service(timeout_sec=1.0):
            if not rclpy.ok():
                self.get_logger().error(
                    "Interrupted while waiting for the service. Exiting."
                )
                return None
            self.get_logger().info(f"Waiting for '{service}' service...")
        request = GetParameters.Request(names=[parameter_name])
        future = parameter_client.call_async(request=request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error(f"Failed to retrieve '{parameter_name}'.")
            return None
        return future.result().values[0]

    def get_current_state(self):
        ##Return a dictionary with all the corresponding state variables
        # Pose, Velocity, Force, Torque, Jacobian, Joint Angles, Joint Velocity, Gripper Pose
        self.robot_state["pose"] = np.zeros((7,))
        self.robot_state["vel"] = np.zeros((6,))
        self.robot_state["force"] = np.zeros((3,))
        self.robot_state["torque"] = np.zeros((3,))
        self.robot_state["q"] = np.zeros((7,))
        self.robot_state["dq"] = np.zeros((7,))

        ##TCP pose
        current_ee_geom_pose, current_joint_state = self.get_fk()
        self.robot_state["pose"] = np.array(
            [
                current_ee_geom_pose.position.x,
                current_ee_geom_pose.position.y,
                current_ee_geom_pose.position.z,
                current_ee_geom_pose.orientation.x,
                current_ee_geom_pose.orientation.y,
                current_ee_geom_pose.orientation.z,
                current_ee_geom_pose.orientation.w,
            ]
        )

        ##Getting Joint Angles
        joint_angles = [
            current_joint_state.position[0],
            current_joint_state.position[1],
            current_joint_state.position[3],
            current_joint_state.position[2],
            current_joint_state.position[4],
            current_joint_state.position[5],
            current_joint_state.position[6],
        ]

        self.robot_state["q"] = joint_angles
        joint_velocity = [
            current_joint_state.velocity[0],
            current_joint_state.velocity[1],
            current_joint_state.velocity[3],
            current_joint_state.velocity[2],
            current_joint_state.velocity[4],
            current_joint_state.velocity[5],
            current_joint_state.velocity[6],
        ]

        self.robot_state["dq"] = joint_velocity

        ##Getting Force and Wrench
        current_wrench = copy.deepcopy(self.latest_wrench)
        self.robot_state["force"] = current_wrench[:3]
        self.robot_state["torque"] = current_wrench[3:]
        self.robot_state["jacobian"] = self.jacobian_func(joint_angles)
        self.robot_state["vel"] = np.dot(self.robot_state["jacobian"], joint_velocity)
        return self.robot_state

    def is_socket_open(self, host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)  # Timeout after 1 second
        try:
            sock.connect((host, port))
            return True  # If the connection succeeds, the socket is open
        except (socket.timeout, socket.error):
            return False  # If there’s an error, the socket is likely closed
        finally:
            sock.close()  # Always close the socket after checking

    def move_to_pose(self, pose: np.ndarray, reset_pose: np.ndarray):

        if not self.session_ok:
            print("ERROR in LBR FRI Restart the node")
            input("Press Enter When Done Resetting")
            input("Are you sure everything is good? Press Enter Again")
            self.session_ok = True

        target_pose = Pose()

        target_pose.position.x = float(pose[0])
        target_pose.position.y = float(pose[1])
        target_pose.position.z = float(pose[2])

        # target_pose_quaternion = euler_to_quat(pose[3:])
        target_pose.orientation.x = float(pose[3])
        target_pose.orientation.y = float(pose[4])
        target_pose.orientation.z = float(pose[5])
        target_pose.orientation.w = float(pose[6])

        traj = self.get_motion_plan(target_pose=target_pose, linear=True, scaling_factor=0.2)
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
                self.get_logger().debug("Trajectory accepted")

            result_future = goal_handle.get_result_async()

            expect_duration = traj.joint_trajectory.points[-1].time_from_start
            expect_time = time.time() + 2 * expect_duration.sec
            while not result_future.done() and time.time() < expect_time:
                time.sleep(0.01)

            self.get_logger().debug("Trajectory executed")

        return

    def move_to_joint_pos(self, joint_pos: np.ndarray):
        
        if not self.session_ok:
            print("ERROR in LBR FRI Restart the node")
            input("Press Enter When Done Resetting")
            input("Are you sure everything is good? Press Enter Again")
            self.session_ok = True

        

        goal_msg = FollowJointTrajectory.Goal()
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ["A1", "A2", "A4", "A3", "A5", "A6", "A7"]

        point = JointTrajectoryPoint()
        point.positions = [float(val) for val in joint_pos]
        point.time_from_start.sec = 2
        trajectory_msg.points.append(point)
        goal_msg.trajectory = trajectory_msg
        self.fri_execute_client_.wait_for_server()

        send_goal_future = self.fri_execute_client_.send_goal_async(
            goal_msg
        )

        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            print(f"Goal for init point was rejected")
            return

        print(f"Goal for init point was accepted")
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result().result
        time.sleep(1)
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
    
    # https://github.com/moveit/moveit/issues/3583#issuecomment-2035540755
    def adjust_trajectory_timings(self, trajectory):
        epsilon = 0.001  # Small time increment to adjust duplicate timings
        for i in range(1, len(trajectory.joint_trajectory.points)):
            if trajectory.joint_trajectory.points[i].time_from_start == trajectory.joint_trajectory.points[i - 1].time_from_start:
                print("THIS IS TRUE HERE")
                trajectory.joint_trajectory.points[i].time_from_start += rclpy.duration.Duration(seconds=epsilon)
        return trajectory

    def get_motion_plan(
        self,
        target_pose: Pose = None,
        linear: bool = False,
        scaling_factor: float = 0.1,
        attempts: int = 10,
        target_joint_state_in: np.ndarray = None,
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
        current_robot_state.joint_state.name = current_joint_state.name

        if target_joint_state_in is not None:
            target_joint_state = JointState()
            target_joint_state.position = [float(val) for val in target_joint_state_in]
            target_joint_state.name = current_joint_state.name
            # print("Printing target joint state: ", target_joint_state)
        
        else:
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

        for i in range(attempts):
            # print(f"Attempt {i} to generate a trajectory")
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
                traj = response.motion_plan_response.trajectory
                # # Adjust timings as mentioned above
                # traj = self.adjust_trajectory_timings(traj)  
                return traj

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
        self.get_logger().debug("Trajectory goal accepted")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        execution_finished = False
        expected_timout = time.time() + 30
        if short_wait:
            expected_timout = time.time() + 2.0
        while not execution_finished and time.time() < expected_timout:
            _, lbr_state = wait_for_message(LBRState, self, self.lbr_state_topic_)
            if (
                np.max(
                    np.abs(
                        np.subtract(
                            lbr_state.measured_joint_position,
                            traj.joint_trajectory.points[-1].positions,
                        )
                    )
                )
                < 0.0002
            ):
                execution_finished = True

            time.sleep(0.01)

        self.get_logger().debug("Trajectory executed")
        return True


def main(args=None):
    rclpy.init(args=args)

    robot_interface_node = RobotInterfaceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(robot_interface_node)
    robot_interface_node.get_logger().info("Robot interface node started.")

    executor.spin()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
