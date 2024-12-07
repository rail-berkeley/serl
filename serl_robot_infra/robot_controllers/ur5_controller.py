import datetime
import time
import threading
import asyncio
import numpy as np
from scipy.spatial.transform import Rotation as R
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from ur_env.utils.vacuum_gripper import VacuumGripper
from ur_env.utils.rotations import rotvec_2_quat, quat_2_rotvec, pose2rotvec, pose2quat

np.set_printoptions(precision=4, suppress=True)


def pos_difference(quat_pose_1: np.ndarray, quat_pose_2: np.ndarray):
    assert quat_pose_1.shape == (7,)
    assert quat_pose_2.shape == (7,)
    p_diff = np.sum(np.abs(quat_pose_1[:3] - quat_pose_2[:3]))

    r_diff = (
        R.from_quat(quat_pose_1[3:]) * R.from_quat(quat_pose_2[3:]).inv()
    ).magnitude()
    return p_diff + r_diff


class UrImpedanceController(threading.Thread):
    def __init__(
        self,
        robot_ip,
        frequency=100,
        kp=10000,
        kd=2200,
        config=None,
        verbose=False,
        *args,
        **kwargs,
    ):
        super(UrImpedanceController, self).__init__(*args, **kwargs)
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._is_ready = threading.Event()
        self._is_truncated = threading.Event()
        self.lock = threading.Lock()

        self.robot_ip = robot_ip
        self.frequency = frequency
        self.kp = kp
        self.kd = kd
        self.gripper_timeout = {
            "timeout": config.GRIPPER_TIMEOUT,
            "last_grip": time.monotonic() - 1e6,
        }
        self.verbose = verbose

        self.target_pos = np.zeros(
            (7,), dtype=np.float32
        )  # new as quat to avoid +- problems with axis angle repr.
        self.target_grip = np.zeros((1,), dtype=np.float32)
        self.curr_pos = np.zeros((7,), dtype=np.float32)
        self.curr_vel = np.zeros((6,), dtype=np.float32)
        self.gripper_state = np.zeros((2,), dtype=np.float32)
        self.curr_Q = np.zeros((6,), dtype=np.float32)
        self.curr_Qd = np.zeros((6,), dtype=np.float32)
        self.curr_force_lowpass = np.zeros((6,), dtype=np.float32)  # force of tool tip
        self.curr_force = np.zeros((6,), dtype=np.float32)

        self.reset_Q = np.array(
            [np.pi / 2.0, -np.pi / 2.0, np.pi / 2.0, -np.pi / 2.0, -np.pi / 2.0, 0.0],
            dtype=np.float32,
        )  # reset state in Joint Space
        self.reset_Pose = np.zeros_like(self.reset_Q)
        self.reset_height = np.array([0.1], dtype=np.float32)  # TODO make customizable

        self.delta = config.ERROR_DELTA
        self.fm_damping = config.FORCEMODE_DAMPING
        self.fm_task_frame = config.FORCEMODE_TASK_FRAME
        self.fm_selection_vector = config.FORCEMODE_SELECTION_VECTOR
        self.fm_limits = config.FORCEMODE_LIMITS

        self.ur_control: RTDEControlInterface = None
        self.ur_receive: RTDEReceiveInterface = None
        self.robotiq_gripper: VacuumGripper = None

        # only temporary to test
        self.hist_data = [[], []]
        self.horizon = [0, 500]
        self.err = 0
        self.noerr = 0

        # log to file (reset every new run)
        with open("/tmp/console2.txt", "w") as f:
            f.write("reset\n")
        self.second_console = open("/tmp/console2.txt", "a")

    def start(self):
        super().start()
        if self.verbose:
            print(f"[RIC] Controller process spawned at {self.native_id}")

    def print(self, msg, both=False):
        self.second_console.write(f"{datetime.datetime.now()} --> {msg}\n")
        if both:
            print(msg)

    async def start_ur_interfaces(self, gripper=True):
        self.ur_control = RTDEControlInterface(self.robot_ip)
        self.ur_receive = RTDEReceiveInterface(self.robot_ip)
        if gripper:
            self.robotiq_gripper = VacuumGripper(self.robot_ip)
            await self.robotiq_gripper.connect()
            await self.robotiq_gripper.activate()
        if self.verbose:
            gr_string = "(with gripper) " if gripper else ""
            print(f"[RIC] Controller connected to robot {gr_string}at: {self.robot_ip}")

    async def restart_ur_interface(self):
        self._is_truncated.set()
        self.print("[RIC] forcemode failed, is now truncated!")

        # disconnect and reconnect, otherwise the controller won't take any commands
        self.ur_control.disconnect()
        try:
            print(f"[RTDE] trying to reconnect")
            self.ur_control.reconnect()
        except RuntimeError:
            self.ur_receive.disconnect()
            for _ in range(10):
                try:
                    self.ur_control.disconnect()
                    self.ur_receive.disconnect()
                    await self.start_ur_interfaces(gripper=False)
                    return
                except Exception as e:
                    print(e)
                    time.sleep(0.2)

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.is_set()

    def is_moving(self):
        return np.linalg.norm(self.get_state()["vel"], 2) > 0.01

    def set_target_pos(self, target_pos: np.ndarray):
        if target_pos.shape == (7,):
            target_orientation = target_pos[3:]
        elif target_pos.shape == (6,):
            target_orientation = rotvec_2_quat(target_pos[3:])
        else:
            raise ValueError(f"[RIC] target pos has shape {target_pos.shape}")

        with self.lock:
            self.target_pos[:3] = target_pos[:3]
            self.target_pos[3:] = target_orientation

            self.print(f"target: {self.target_pos}")

    def set_reset_Q(self, reset_Q: np.ndarray):
        with self.lock:
            self.reset_Q[:] = reset_Q
        self._reset.set()

    def set_reset_pose(self, reset_pose: np.ndarray):
        with self.lock:
            self.reset_Pose[:] = reset_pose
        self._reset.set()

    def set_gripper_pos(self, target_grip: np.ndarray):
        with self.lock:
            self.target_grip[:] = target_grip

    def get_target_pos(self, copy=True):
        with self.lock:
            if copy:
                return self.target_pos.copy()
            else:
                return self.target_pos

    async def _update_robot_state(self):
        pos = self.ur_receive.getActualTCPPose()
        vel = self.ur_receive.getActualTCPSpeed()
        Q = self.ur_receive.getActualQ()
        Qd = self.ur_receive.getActualQd()
        force = self.ur_receive.getActualTCPForce()
        pressure = await self.robotiq_gripper.get_current_pressure()
        obj_status = await self.robotiq_gripper.get_object_status()

        # 3-> no object detected, 0-> sucking empty, [1, 2] obj detected
        grip_status = [-1.0, 1.0, 1.0, 0.0][obj_status.value]

        pressure = (
            pressure if pressure < 99 else 0
        )  # 100 no obj, 99 sucking empty, so they are ignored
        # grip status, 0->neutral, -1->bad (sucking but no obj), 1-> good (sucking and obj)
        grip_status = 1.0 if pressure > 0 else grip_status
        pressure /= 98.0  # pressure between [0, 1]
        with self.lock:
            self.curr_pos[:] = pose2quat(pos)
            self.curr_vel[:] = vel
            self.curr_Q[:] = Q
            self.curr_Qd[:] = Qd
            self.curr_force[:] = np.array(force)
            # use moving average (5), since the force fluctuates heavily
            self.curr_force_lowpass[:] = (
                0.1 * np.array(force) + 0.9 * self.curr_force_lowpass[:]
            )
            self.gripper_state[:] = [pressure, grip_status]

    def get_state(self):
        with self.lock:
            state = {
                "pos": self.curr_pos,
                "vel": self.curr_vel,
                "Q": self.curr_Q,
                "Qd": self.curr_Qd,
                "force": self.curr_force_lowpass[:3],
                "torque": self.curr_force_lowpass[3:],
                "gripper": self.gripper_state,
            }
            return state

    def is_ready(self):
        return self._is_ready.is_set()

    def is_reset(self):
        return not self._reset.is_set()

    def _calculate_force(self):
        target_pos = self.get_target_pos(copy=True)
        with self.lock:
            curr_pos = self.curr_pos
            curr_vel = self.curr_vel

        # calc position for
        kp, kd = self.kp, self.kd
        diff_p = np.clip(
            target_pos[:3] - curr_pos[:3], a_min=-self.delta, a_max=self.delta
        )
        vel_delta = 2 * self.delta * self.frequency
        diff_d = np.clip(-curr_vel[:3], a_min=-vel_delta, a_max=vel_delta)
        force_pos = kp * diff_p + kd * diff_d

        # calc torque
        rot_diff = R.from_quat(target_pos[3:]) * R.from_quat(curr_pos[3:]).inv()
        vel_rot_diff = R.from_rotvec(curr_vel[3:]).inv()
        torque = (
            rot_diff.as_rotvec() * 100 + vel_rot_diff.as_rotvec() * 22
        )  # TODO make customizable

        # check for big downward tcp force and adapt accordingly
        if self.curr_force[2] > 3.5 and force_pos[2] < 0.0:
            force_pos[2] = (
                max((1.5 - self.curr_force_lowpass[2]), 0.0) * force_pos[2]
                + min(self.curr_force_lowpass[2] - 0.5, 1.0) * 20.0
            )

        return np.concatenate((force_pos, torque))

    async def send_gripper_command(self, force_release=False):
        if force_release:
            await self.robotiq_gripper.automatic_release()
            self.target_grip[0] = 0.0
            return

        timeout_exceeded = (
            time.monotonic() - self.gripper_timeout["last_grip"]
        ) * 1000 > self.gripper_timeout["timeout"]
        # target grip above threshold and timeout exceeded and not gripping something already
        if (
            self.target_grip[0] > 0.5
            and timeout_exceeded
            and self.gripper_state[1] < 0.5
        ):
            await self.robotiq_gripper.automatic_grip()
            self.target_grip[0] = 0.0
            self.gripper_timeout["last_grip"] = time.monotonic()
            # print("grip")

        # release if below neg threshold and gripper activated (grip_status not zero)
        elif self.target_grip[0] < -0.5 and abs(self.gripper_state[1]) > 0.5:
            await self.robotiq_gripper.automatic_release()
            self.target_grip[0] = 0.0
            # print("release")

    def _truncate_check(self):
        downward_force = self.curr_force_lowpass[2] > 20.0
        if downward_force:  # TODO add better criteria
            self._is_truncated.set()
        else:
            self._is_truncated.clear()

    def is_truncated(self):
        return self._is_truncated.is_set()

    def run(self):
        try:
            asyncio.run(
                self.run_async()
            )  # gripper has to be awaited, both init and commands
        finally:
            self.stop()

    async def _go_to_reset_pose(self):
        self.ur_control.forceModeStop()

        # first disable vaccum gripper
        if self.robotiq_gripper:
            await self.send_gripper_command(force_release=True)
            time.sleep(0.01)

        # then move up (so no boxes are moved)
        success = True
        while self.curr_pos[2] < self.reset_height:
            if (
                self.curr_Q[2] < 0.5
            ):  # if the shoulder joint is near 180deg --> do not move into singularity
                success = success and self.ur_control.speedJ(
                    [0.0, -1.0, 1.0, 0.0, 0.0, 0.0], acceleration=0.8
                )
            else:
                success = success and self.ur_control.speedL(
                    [0.0, 0.0, 0.25, 0.0, 0.0, 0.0], acceleration=0.8
                )
            await self._update_robot_state()
            time.sleep(0.01)
        self.ur_control.speedStop(a=1.0)

        if self.reset_Pose.std() > 0.001:
            success = success and self.ur_control.moveL(
                self.reset_Pose, speed=0.5, acceleration=0.3
            )
            self.print(
                f"[RIC] moving to {self.reset_Pose} with moveL (task space)",
                both=self.verbose,
            )
            self.reset_Pose[:] = 0.0
        else:
            # then move to desired Jointspace position
            success = success and self.ur_control.moveJ(
                self.reset_Q, speed=1.0, acceleration=0.8
            )
            self.print(
                f"[RIC] moving to {self.reset_Q} with moveJ (joint space)",
                both=self.verbose,
            )

        time.sleep(0.1)  # wait for 100ms
        await self._update_robot_state()
        with self.lock:
            self.target_pos = self.curr_pos.copy()

        self.ur_control.forceModeSetDamping(self.fm_damping)  # less damping = Faster
        self.ur_control.zeroFtSensor()

        if not success:  # restart if not successful
            await self.restart_ur_interface()
        else:
            self._reset.clear()

    async def run_async(self):
        await self.start_ur_interfaces(gripper=True)

        self.ur_control.forceModeSetDamping(self.fm_damping)  # less damping = Faster

        try:
            dt = 1.0 / self.frequency
            self.ur_control.zeroFtSensor()
            await self._update_robot_state()
            self.target_pos = self.curr_pos.copy()
            print(f"[RIC] target position set to curr pos: {self.target_pos}")

            self._is_ready.set()

            while not self.stopped():
                if self._reset.is_set():
                    await self._update_robot_state()
                    await self._go_to_reset_pose()

                t_now = time.monotonic()

                # update robot state and check for truncation
                await self._update_robot_state()
                self._truncate_check()

                # calculate force
                force = self._calculate_force()
                # print(self.target_pos, self.curr_pos, force)
                self.print(
                    f" p:{self.curr_pos}   f:{self.curr_force_lowpass}   gr:{self.gripper_state}"
                )  # log to file

                # send command to robot
                t_start = self.ur_control.initPeriod()
                fm_successful = self.ur_control.forceMode(
                    self.fm_task_frame,
                    self.fm_selection_vector,
                    force,
                    2,
                    self.fm_limits,
                )
                if not fm_successful:  # truncate if the robot ends up in a singularity
                    await self.restart_ur_interface()
                    await self._go_to_reset_pose()

                if self.robotiq_gripper:
                    await self.send_gripper_command()

                self.ur_control.waitPeriod(t_start)

                a = dt - (time.monotonic() - t_now)
                time.sleep(max(0.0, a))
                self.err, self.noerr = self.err + int(a < 0.0), self.noerr + int(
                    a >= 0.0
                )  # some logging
                if a < -0.04:  # log if delay more than 50ms
                    self.print(
                        f"Controller Thread stopped for {(time.monotonic() - t_now)*1e3:.1f} ms"
                    )

        finally:
            if self.verbose:
                print(
                    f"[RTDEPositionalController] >dt: {self.err}     <dt (good): {self.noerr}"
                )
            # mandatory cleanup
            self.ur_control.forceModeStop()

            # release gripper
            if self.robotiq_gripper:
                await self.send_gripper_command(force_release=True)
                time.sleep(0.05)

            # move to real home
            pi = 3.1415
            reset_Q = [0, -pi / 2.0, pi / 2.0, -pi / 2.0, -pi / 2.0, 0.0]
            self.ur_control.moveJ(reset_Q, speed=1.0, acceleration=0.8)

            # terminate
            self.ur_control.disconnect()
            self.ur_receive.disconnect()

            if self.verbose:
                print(
                    f"[RTDEPositionalController] Disconnected from robot: {self.robot_ip}"
                )
