"""
MIT License

Copyright (c) 2019 Anders Prier Lindvig - SDU Robotics
Copyright (c) 2020 Fabian Freihube - DavinciKitchen GmbH

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Module to control Robotiq's gripper 2F-85 and Hand-E.
Originally from here: https://sdurobotics.gitlab.io/ur_rtde/_static/gripper_2f85.py
Adjusted for use with asyncio
"""

import asyncio
from enum import Enum
from typing import Union, Tuple, OrderedDict

# TODO: add blocking to release, gripping

class VacuumGripper:
    """
    Communicates with the gripper directly, via socket with string commands, leveraging string names for variables.
    """

    # WRITE VARIABLES (CAN ALSO READ)
    ACT = "ACT"  # act : activate (1 while activated, can be reset to clear fault status)
    GTO = "GTO"  # gto : go to (will perform go to with the actions set in pos, for, spe)
    ATR = "ATR"  # atr : auto-release (emergency slow move)
    FOR = "FOR"  # for : vacuum minimum relative pressure (0-255)
    SPE = "SPE"  # spe : grip timeout/release delay
    POS = "POS"  # pos : vacuum max pressure (0-255)
    MOD = "MOD"  # mod : mode - automatic vs advanced mode

    # READ VARIABLES
    STA = "STA"  # status (0 = is reset, 1 = activating, 3 = active)
    PRE = "PRE"  # position request (echo of last commanded position)
    OBJ = "OBJ"  # object detection (0 = unknown, 1 = minimum pressure value reached, 2 = maximum pressure reached, 3 = no obj detected)
    FLT = "FLT"  # fault (0=ok, see manual for errors if not zero)

    ENCODING = "UTF-8"  # ASCII and UTF-8 both seem to work

    class GripperStatus(Enum):
        """Gripper status reported by the gripper. The integer values have to match what the gripper sends."""

        RESET = 0
        ACTIVATING = 1
        # UNUSED = 2  # This value is currently not used by the gripper firmware
        ACTIVE = 3

    class ObjectStatus(Enum):
        """Object status reported by the gripper. The integer values have to match what the gripper sends."""

        MOVING = 0
        DETECTED_MIN = 1
        DETECTED_MAX = 2
        NO_OBJ_DETECTED = 3

    def __init__(self, hostname: str, port: int = 63352) -> None:
        """Constructor.

        :param hostname: Hostname or ip of the robot arm.
        :param port: Port.

        """
        self.socket_reader = None
        self.socket_writer = None
        self.command_lock = asyncio.Lock()

        self.hostname = hostname
        self.port = port

    async def connect(self) -> None:
        """Connects to a gripper on the provided address"""
        # print(self.hostname, self.port)
        self.socket_reader, self.socket_writer = await asyncio.open_connection(self.hostname, self.port)

    async def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        self.socket_writer.close()
        await self.socket_writer.wait_closed()

    async def _set_vars(self, var_dict: 'OrderedDict[str, Union[int, float]]') -> bool:
        """Sends the appropriate command via socket to set the value of n variables, and waits for its 'ack' response.

        :param var_dict: Dictionary of variables to set (variable_name, value).
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        # construct unique command
        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {str(value)}"
        cmd += "\n"  # new line is required for the command to finish
        # atomic commands send/rcv
        async with self.command_lock:
            self.socket_writer.write(cmd.encode(self.ENCODING))
            await self.socket_writer.drain()
            response = await self.socket_reader.read(1024)
        return self._is_ack(response)

    async def _set_var(self, variable: str, value: Union[int, float]) -> bool:
        """Sends the appropriate command via socket to set the value of a variable, and waits for its 'ack' response.

        :param variable: Variable to set.
        :param value: Value to set for the variable.
        :return: True on successful reception of ack, false if no ack was received, indicating the set may not
        have been effective.
        """
        return await self._set_vars(OrderedDict([(variable, value)]))

    async def _get_var(self, variable: str) -> int:
        """Sends the appropriate command to retrieve the value of a variable from the gripper, blocking until the
        response is received or the socket times out.

        :param variable: Name of the variable to retrieve.
        :return: Value of the variable as integer.
        """
        # atomic commands send/rcv
        async with self.command_lock:
            cmd = f"GET {variable}\n"
            self.socket_writer.write(cmd.encode(self.ENCODING))
            await self.socket_writer.drain()
            data = await self.socket_reader.read(1024)

        # expect data of the form 'VAR x', where VAR is an echo of the variable name, and X the value
        # note some special variables (like FLT) may send 2 bytes, instead of an integer. We assume integer here
        var_name, value_str = data.decode(self.ENCODING).split()
        if var_name != variable:
            raise ValueError(f"Unexpected response {data} ({data.decode(self.ENCODING)}): does not match '{variable}'")
        value = int(value_str)
        return value

    @staticmethod
    def _is_ack(data: str) -> bool:
        return data == b"ack"

    async def activate(self) -> None:
        """Resets the activation flag in the gripper, and sets it back to one, clearing previous fault flags.

        :param auto_calibrate: Whether to calibrate the minimum and maximum positions based on actual motion.
        """
        # stop the vacuum generator
        await self._set_var(self.GTO, 0)
        #await self._set_var(self.GTO, 1)

        # to clear fault status
        await self._set_var(self.ACT, 0)
        await self._set_var(self.ACT, 1)

        # wait for activation to go through
        while not await self.is_active():
            await asyncio.sleep(0.01)

    async def is_active(self) -> bool:
        """Returns whether the gripper is active."""
        status = await self._get_var(self.STA)
        return VacuumGripper.GripperStatus(status) == VacuumGripper.GripperStatus.ACTIVE

    async def get_current_pressure(self) -> int:
        """Returns the current pressure as returned by the physical hardware, max pressure if not gripping."""
        return await self._get_var(self.POS)

    async def get_object_status(self) -> ObjectStatus:
        a = await self._get_var(self.OBJ)
        return VacuumGripper.ObjectStatus(a)

    async def get_fault_status(self) -> int:
        value = await self._get_var(self.FLT)
        return value
    
    async def automatic_grip(self) -> bool:
        """Sends commands to grip using automatic mode.
        In automatic mode, the pressure byte is used to send a grip/release request

        :return: A tuple with a bool indicating whether the action it was successfully sent, and an integer with
        the actual position that was requested, after being adjusted to the min/max calibrated range.
        """

        # activate sets GTO to 0 and makes sure that the gripper is activated
        await self.activate()

        # in automatic mode, any pressure (POS) value < 100 will lead to the grip command
        var_dict = OrderedDict([(self.POS, 50), (self.MOD, 0)])

        # first set the values, then set GTO
        await self._set_vars(var_dict)
        await self._set_var(self.GTO, 1)

    async def advanced_grip(self, min_pressure, max_pressure, timeout) -> bool:
        """ Sends commands to grip in advanced mode.
        min pressure is [0, 99]
        max pressure is [10, 78]
        timeout is in ms [0, 255]
        """

        # activate sets GTO to 0 and makes sure that the gripper is activated
        await self.activate()

        def clip_val(min_val, val, max_val):
            return max(min_val, min(val, max_val))

        clip_min_pressure = 100 - clip_val(0, min_pressure, 99)
        clip_max_pressure = 100 - clip_val(10, max_pressure, 78)
        clip_timeout = clip_val(0, timeout, 255)

        #val = await self.get_fault_status()
        #print(val)

        # moves to the given position with the given speed and force
        var_dict = OrderedDict([
            (self.MOD, 1),
            (self.POS, clip_max_pressure), 
            (self.FOR, clip_min_pressure), 
            (self.SPE, clip_timeout)])

        await self._set_vars(var_dict)
        await self._set_var(self.GTO, 1)

    async def continuous_grip(self, timeout) -> bool:
        """ Sends commands to grip in advanced mode.
        min pressure is [0, 99]
        max pressure is [10, 78]
        timeout is in ms [0, 255]
        """

        def clip_val(min_val, val, max_val):
            return max(min_val, min(val, max_val))

        clip_timeout = clip_val(0, timeout, 255)

        # moves to the given position with the given speed and force
        var_dict = OrderedDict([
            (self.MOD, 1),
            (self.POS, 0), 
            (self.SPE, clip_timeout),
            (self.GTO, 1)])

        return await self._set_vars(var_dict)

    async def advanced_release(self, min_pressure, max_pressure, timeout) -> bool:
        """ Sends commands to do advanced release. This allows to do a more controlled release than the automatic one.
        """
        var_dict = OrderedDict([
            (self.POS, 255), 
            (self.GTO, 1)])

        return await self._set_vars(var_dict)

    async def automatic_release(self) -> bool:
        """ Sends commands to do automatic release. 
        """
        var_dict = OrderedDict([(self.ACT, 1), (self.ATR, 1)])
        return await self._set_vars(var_dict)
