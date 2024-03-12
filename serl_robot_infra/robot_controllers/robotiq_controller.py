import time
import threading
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from robotiq_env.envs.robotiq_env import DefaultEnvConfig
from robotiq_env.utils.vacuum_gripper import VacuumGripper
from robotiq_env.utils.rotations import rotvec_2_quat, quat_2_rotvec

np.set_printoptions(precision=4, suppress=True)


def pose2quat(rotvec_pose) -> np.ndarray:
    return np.concatenate((rotvec_pose[:3], rotvec_2_quat(rotvec_pose[3:])))


def pose2rotvec(quat_pose) -> np.ndarray:
    return np.concatenate((quat_pose[:3], quat_2_rotvec(quat_pose[3:])))


class RobotiqImpedanceController(threading.Thread):
    def __init__(
            self,
            robot_ip,
            frequency=100,
            kp=4e4,
            kd=8e3,
            config: DefaultEnvConfig=None,
            verbose=True,
            *args,
            **kwargs
    ):
        super(RobotiqImpedanceController, self).__init__(*args, **kwargs)
        self._stop = threading.Event()
        """
        frequency: CB2=125, UR3e=500
        max_pos_speed: m/s
        max_rot_speed: rad/s
        """

        self.robot_ip = robot_ip
        self.frequency = frequency
        self.kp = kp
        self.kd = kd
        self.lock = threading.Lock()
        self.verbose = verbose

        self.target_pos = np.zeros((7,))  # new as quat to avoid +- problems with axis angle repr.
        self.target_grip = np.zeros((1,))
        self.curr_pos = np.zeros((7,))
        self.curr_vel = np.zeros((7,))
        self.curr_gripper_state = np.zeros((1,))  # TODO gripper state (sucking or not)
        self.curr_Q = np.zeros((6,))
        self.curr_Qd = np.zeros((6,))
        self.curr_force = np.zeros((6,))  # force of tool tip

        self.fm_damping = config.FORCEMODE_DAMPING
        self.fm_task_frame = config.FORCEMODE_TASK_FRAME
        self.fm_selection_vector = config.FORCEMODE_SELECTION_VECTOR
        self.fm_limits = config.FORCEMODE_LIMITS

        self.action_scale = config.ACTION_SCALE

        self.robotiq_control: RTDEControlInterface = None
        self.robotiq_receive: RTDEReceiveInterface = None
        self.robotiq_gripper: VacuumGripper = None

        self._is_ready = False

        # only temporary to test
        self.hist_data = [[], []]
        self.horizon = [0, 500]
        self.err = 0
        self.noerr = 0

    def start(self):
        super().start()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.native_id}")

    async def start_robotiq_interfaces(self, gripper=True):
        self.robotiq_control = RTDEControlInterface(self.robot_ip)
        self.robotiq_receive = RTDEReceiveInterface(self.robot_ip)
        if gripper:
            self.robotiq_gripper = VacuumGripper(self.robot_ip)
            await self.robotiq_gripper.connect()
            await self.robotiq_gripper.activate()
        if self.verbose:
            gr_string = "(with gripper) " if gripper else ""
            print(f"[RTDEPositionalController] Controller connected to robot {gr_string}at: {self.robot_ip}")

    def init_pose(self):
        target_pose = [-0.25, -0.7, 0.25, 0., np.pi, 0.]
        self.robotiq_control.moveL(target_pose, speed=0.3, acceleration=0.3)

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.is_set()

    def set_target_pos(self, action, gripper_action):  # TODO make action scale
        # action = np.array(action[:6]) * np.array([1.5, 1.5, 1.5, 3, 3, 3]) * 0.01
        action = np.array(action[:6], dtype=np.float32)
        action[:3] *= self.action_scale[0]
        action[3:] *= self.action_scale[1]
        with self.lock:
            self.target_pos[:3] += action[:3]
            self.target_pos[3:] = (R.from_quat(self.target_pos[3:]) * R.from_rotvec(action[3:])).as_quat()
            # print("new orientation: ", self.target_pos[3:], "curr: ", self.curr_pos[3:])
            self.target_grip = gripper_action

    def _get_target_pos(self):
        with self.lock:
            return self.target_pos

    async def _update_robot_state(self):
        with self.lock:
            self.curr_pos = pose2quat(self.robotiq_receive.getActualTCPPose())
            self.curr_vel = pose2quat(self.robotiq_receive.getActualTCPSpeed())
            self.curr_Q = np.array(self.robotiq_receive.getActualQ())
            self.curr_Qd = np.array(self.robotiq_receive.getActualQd())
            self.curr_force = np.array(self.robotiq_receive.getActualTCPForce())
            self.curr_gripper_state = np.array(await self.robotiq_gripper.get_current_pressure())

    def get_state(self):
        with self.lock:
            state = {
                "pos": self.curr_pos,
                "vel": self.curr_vel,
                "Q": self.curr_Q,
                "Qd": self.curr_Qd,
                "force": self.curr_force[:3],
                "torque": self.curr_force[3:],
                "pressure": self.curr_gripper_state
            }
            return state

    def is_ready(self):
        with self.lock:
            return self._is_ready

    def _calculate_force(self):
        target_pos = self._get_target_pos()

        # calc position force
        kp, kd = self.kp, self.kd
        diff_p = target_pos[:3] - self.curr_pos[:3]
        diff_d = - self.curr_vel[:3]
        force_pos = kp * diff_p + kd * diff_d

        # calc torque
        # rot_diff = rotvec_diff(self.curr_pos[3:], target_pos[3:])
        # rot_diff_vel = rotvec_diff(self.curr_vel[3:], [0, 0, 0])
        # torque = 500 * rot_diff

        # calc torque new
        rot_diff = R.from_quat(self.target_pos[3:]) * R.from_quat(self.curr_pos[3:]).inv()
        torque = 200 * rot_diff.as_rotvec()

        # np.clip(force_pos, a_min=-124., a_max=124.)
        # np.clip(torque, a_min=-50., a_max=50.)
        return np.concatenate((force_pos, torque))

    def run(self):
        asyncio.run(self.run_async())  # gripper has to be awaited, both init and commands

    def plot(self):
        if self.horizon[0] == 0:
            self.start_t = time.monotonic()
        if self.horizon[0] < self.horizon[1]:
            self.horizon[0] += 1
            # print(self.horizon[0], self.horizon[1], end="  ")
            return

        print(time.monotonic() - self.start_t)
        self.robotiq_control.forceModeStop()

        self.horizon[0] = -1e5
        print("plotting")
        real_pos = np.array([pose2rotvec(q) for q in self.hist_data[0]])
        target_pos = np.array([pose2rotvec(q) for q in self.hist_data[1]])

        plt.figure()
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), dpi=300)
        for i in range(6):
            ax = axes[i % 3, i // 3]
            ax.plot(real_pos[:, i], 'b')
            ax.plot(target_pos[:, i], 'g')

        fig.suptitle(f"params_ kp:{self.kp}  kd:{self.kd}")
        plt.show(block=True)
        self.stop()

    async def run_async(self):
        await self.start_robotiq_interfaces(gripper=True)
        self.init_pose()

        self.robotiq_control.forceModeSetDamping(self.fm_damping)     # less damping = Faster

        try:
            dt = 1. / self.frequency
            await self._update_robot_state()
            self.robotiq_control.zeroFtSensor()
            self.target_pos = self.curr_pos.copy()

            with self.lock:
                self._is_ready = True

            while not self.stopped():
                t_now = time.monotonic()

                # update robot state
                await self._update_robot_state()

                # log data
                # self.hist_data[0].append(self.curr_pos.copy())
                # self.hist_data[1].append(self.target_pos.copy())

                # self.plot()

                t_start = self.robotiq_control.initPeriod()

                # send command to robot
                force = self._calculate_force()

                assert self.robotiq_control.forceMode(
                    self.fm_task_frame,
                    self.fm_selection_vector,
                    force,
                    2,
                    self.fm_limits
                )

                if self.robotiq_gripper:
                    if self.target_grip > 0.9:
                        await self.robotiq_gripper.automatic_grip()
                    elif self.target_grip < -0.9:
                        await self.robotiq_gripper.automatic_release()

                self.robotiq_control.waitPeriod(t_start)
                a = (1.0 / self.frequency) - (time.monotonic() - t_now)
                time.sleep(max(0., a))
                if a < 0:
                    self.err += 1
                else:
                    self.noerr += 1

        finally:
            print(f"num errs: {self.err}     no_err: {self.noerr}")
            # mandatory cleanup
            self.robotiq_control.forceModeStop()

            # terminate
            self.robotiq_control.disconnect()
            self.robotiq_receive.disconnect()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {self.robot_ip}")
