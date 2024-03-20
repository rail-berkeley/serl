import time
import threading
import asyncio
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from robotiq_env.utils.vacuum_gripper import VacuumGripper
from robotiq_env.utils.rotations import rotvec_2_quat, quat_2_rotvec, pose2rotvec, pose2quat

np.set_printoptions(precision=4, suppress=True)



def pos_difference(quat_pose_1: np.ndarray, quat_pose_2: np.ndarray):
    assert quat_pose_1.shape == (7,)
    assert quat_pose_2.shape == (7,)
    p_diff = np.sum(np.abs(quat_pose_1[:3] - quat_pose_2[:3]))

    r_diff = (R.from_quat(quat_pose_1[3:]) * R.from_quat(quat_pose_2[3:]).inv()).magnitude()
    return p_diff + r_diff


class RobotiqImpedanceController(threading.Thread):
    def __init__(
            self,
            robot_ip,
            frequency=100,
            kp=10000,
            kd=2200,
            config=None,
            verbose=False,
            plot=False,
            *args,
            **kwargs
    ):
        super(RobotiqImpedanceController, self).__init__(*args, **kwargs)
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._is_ready = threading.Event()
        self.lock = threading.Lock()

        self.robot_ip = robot_ip
        self.frequency = frequency
        self.kp = kp
        self.kd = kd
        self.verbose = verbose
        self.do_plot = plot

        self.target_pos = np.zeros((7,), dtype=np.float32)  # new as quat to avoid +- problems with axis angle repr.
        self.target_grip = np.zeros((1,), dtype=np.float32)
        self.curr_pos = np.zeros((7,), dtype=np.float32)
        self.curr_vel = np.zeros((6,), dtype=np.float32)
        self.gripper_state = np.zeros((2,), dtype=np.float32)
        self.curr_Q = np.zeros((6,), dtype=np.float32)
        self.curr_Qd = np.zeros((6,), dtype=np.float32)
        self.curr_force = np.zeros((6,), dtype=np.float32)  # force of tool tip

        self.reset_Q = np.zeros((6,), dtype=np.float32)  # reset state in Joint Space

        self.delta = config.ERROR_DELTA
        self.fm_damping = config.FORCEMODE_DAMPING
        self.fm_task_frame = config.FORCEMODE_TASK_FRAME
        self.fm_selection_vector = config.FORCEMODE_SELECTION_VECTOR
        self.fm_limits = config.FORCEMODE_LIMITS

        self.robotiq_control: RTDEControlInterface = None
        self.robotiq_receive: RTDEReceiveInterface = None
        self.robotiq_gripper: VacuumGripper = None

        # only temporary to test
        self.hist_data = [[], []]
        self.horizon = [0, 500]
        self.err = 0
        self.noerr = 0

    def start(self):
        super().start()
        if self.verbose:
            print(f"[RobotiqImpedanceController] Controller process spawned at {self.native_id}")

    async def start_robotiq_interfaces(self, gripper=True):
        for _ in range(5):  # try it 5 times
            try:
                self.robotiq_control = RTDEControlInterface(self.robot_ip)
                self.robotiq_receive = RTDEReceiveInterface(self.robot_ip)
                if gripper:
                    self.robotiq_gripper = VacuumGripper(self.robot_ip)
                    await self.robotiq_gripper.connect()
                    await self.robotiq_gripper.activate()
                if self.verbose:
                    gr_string = "(with gripper) " if gripper else ""
                    print(f"[RobotiqImpedanceController] Controller connected to robot {gr_string}at: {self.robot_ip}")
                    break
            except RuntimeError as e:
                print("[RobotiqImpedanceController] ", e.__str__())
                continue
        # raise RuntimeError(f"[RobotiqImpedanceController] Could not connect to robot [{self.robot_ip}]")

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.is_set()

    def set_target_pos(self, target_pos: np.ndarray):
        if target_pos.shape == (7,):
            target_orientation = target_pos[3:]
        elif target_pos.shape == (6,):
            target_orientation = rotvec_2_quat(target_pos[3:])
        else:
            raise ValueError(f"[RobotiqImpedanceController] target pos has shape {target_pos.shape}")

        with self.lock:
            self.target_pos[:3] = target_pos[:3]
            self.target_pos[3:] = target_orientation

        # print("target: ", self.target_pos)

    def set_reset_Q(self, reset_Q: np.ndarray):
        self._reset.set()
        with self.lock:
            self.reset_Q[:] = reset_Q

    def set_gripper_pos(self, target_grip: np.ndarray):
        with self.lock:
            self.target_grip[:] = target_grip

    def get_target_pos(self):
        with self.lock:
            return self.target_pos

    async def _update_robot_state(self):
        pos = self.robotiq_receive.getActualTCPPose()
        vel = self.robotiq_receive.getActualTCPSpeed()
        Q = self.robotiq_receive.getActualQ()
        Qd = self.robotiq_receive.getActualQd()
        force = self.robotiq_receive.getActualTCPForce()
        pressure = await self.robotiq_gripper.get_current_pressure()
        obj_status = await self.robotiq_gripper.get_object_status()
        with self.lock:
            self.curr_pos[:] = pose2quat(pos)
            self.curr_vel[:] = vel
            self.curr_Q[:] = Q
            self.curr_Qd[:] = Qd
            self.curr_force[:] = force
            self.gripper_state[:] = [pressure, float(obj_status.value)]

    def get_state(self):
        with self.lock:
            state = {
                "pos": self.curr_pos,
                "vel": self.curr_vel,
                "Q": self.curr_Q,
                "Qd": self.curr_Qd,
                "force": self.curr_force[:3],
                "torque": self.curr_force[3:],
                "gripper": self.gripper_state
            }
            return state

    def is_ready(self):
        return self._is_ready.is_set()

    def is_reset(self):
        return not self._reset.is_set()

    def _calculate_force(self):
        target_pos = self.get_target_pos()
        with self.lock:
            curr_pos = self.curr_pos
            curr_vel = self.curr_vel

        # calc position force
        kp, kd = self.kp, self.kd
        diff_p = np.clip(target_pos[:3] - curr_pos[:3], a_min=-self.delta, a_max=self.delta)
        vel_delta = 2 * self.delta * self.frequency
        diff_d = np.clip(- curr_vel[:3], a_min=-vel_delta, a_max=vel_delta)
        force_pos = kp * diff_p + kd * diff_d

        # calc torque
        rot_diff = R.from_quat(target_pos[3:]) * R.from_quat(curr_pos[3:]).inv()
        vel_rot_diff = R.from_rotvec(curr_vel[3:]).inv()
        torque = rot_diff.as_rotvec() * 100 + vel_rot_diff.as_rotvec() * 22  # TODO make customizable

        return np.concatenate((force_pos, torque))

    def plot(self):
        if self.horizon[0] < self.horizon[1]:
            self.horizon[0] += 1
            self.hist_data[0].append(self.curr_pos.copy())
            self.hist_data[1].append(self.target_pos.copy())
            return

        self.robotiq_control.forceModeStop()

        print("[RobotiqImpedanceController] plotting")
        real_pos = np.array([pose2rotvec(q) for q in self.hist_data[0]])
        target_pos = np.array([pose2rotvec(q) for q in self.hist_data[1]])

        plt.figure()
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), dpi=200)
        ax_label = [{'x': f'time {self.frequency} [Hz]', 'y': ylabel} for ylabel in ["[mm]", "[rad]"]]
        plot_label = "X Y Z RX RY RZ".split(' ')

        for i in range(6):
            ax = axes[i % 3, i // 3]
            ax.plot(real_pos[:, i], 'b', label=plot_label[i])
            ax.plot(target_pos[:, i], 'g')
            ax.set(xlabel=ax_label[i // 3]['x'], ylabel=ax_label[i // 3]['y'])
            ax.legend()

        fig.suptitle(f"params-->  kp:{self.kp}  kd:{self.kd}")
        plt.show(block=True)
        self.stop()

    async def send_gripper_command(self):
        if self.target_grip[0] > 0.9 and self.gripper_state[0] == 100:
            await self.robotiq_gripper.automatic_grip()
            self.target_grip[0] = 0.0
            # print("grip")

        elif self.target_grip[0] < -0.9 and self.gripper_state[1] != 3:  # only release if obj detected
            await self.robotiq_gripper.automatic_release()
            self.target_grip[0] = 0.0
            # print("release")

    def run(self):
        try:
            asyncio.run(self.run_async())  # gripper has to be awaited, both init and commands
        finally:
            self.stop()

    async def run_async(self):
        await self.start_robotiq_interfaces(gripper=True)

        self.robotiq_control.forceModeSetDamping(self.fm_damping)  # less damping = Faster
        self.start_t = time.monotonic()

        try:
            dt = 1. / self.frequency
            self.robotiq_control.zeroFtSensor()
            await self._update_robot_state()
            self.target_pos = self.curr_pos.copy()

            self._is_ready.set()

            while not self.stopped():
                if self._reset.is_set():  # move to reset joint space pose with moveJ
                    # first disable vaccum gripper
                    if self.robotiq_gripper:
                        self.target_grip[0] = -1.
                        await self.send_gripper_command()
                        time.sleep(0.1)

                    # then move to Jointspace position
                    print(f"[RobotiqImpedanceController] moving to {self.reset_Q} with moveJ (joint space)")
                    self.robotiq_control.forceModeStop()
                    self.robotiq_control.moveJ(self.reset_Q, speed=1., acceleration=0.5)

                    await self._update_robot_state()
                    with self.lock:
                        self.target_pos = self.curr_pos.copy()

                    self.robotiq_control.forceModeSetDamping(self.fm_damping)  # less damping = Faster
                    self.robotiq_control.zeroFtSensor()

                    self._reset.clear()

                t_now = time.monotonic()

                # update robot state
                await self._update_robot_state()

                # only used for plotting
                if self.do_plot:
                    self.plot()

                # calculate force
                force = self._calculate_force()
                # print(self.target_pos, self.curr_pos, force)

                # send command to robot
                t_start = self.robotiq_control.initPeriod()
                assert self.robotiq_control.forceMode(
                    self.fm_task_frame,
                    self.fm_selection_vector,
                    force,
                    2,
                    self.fm_limits
                )
                # TODO try to reset with moveJ if forcemode returns False

                if self.robotiq_gripper:
                    await self.send_gripper_command()

                self.robotiq_control.waitPeriod(t_start)

                a = dt - (time.monotonic() - t_now)
                time.sleep(max(0., a))
                if a < 0:  # log how slow the loop runs
                    self.err += 1
                else:
                    self.noerr += 1

        finally:
            print(f"[RTDEPositionalController] >dt: {self.err}     <dt (good): {self.noerr}")
            # mandatory cleanup
            self.robotiq_control.forceModeStop()

            # terminate
            self.robotiq_control.disconnect()
            self.robotiq_receive.disconnect()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot: {self.robot_ip}")
