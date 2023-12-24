"""Driver class for SpaceMouse controller.

This class provides a driver support to SpaceMouse on Mac OS X.
In particular, we assume you are using a SpaceMouse Wireless by default.

To set up a new SpaceMouse controller:
    1. Download and install driver from https://www.3dconnexion.com/service/drivers.html
    2. Install hidapi library through pip
       (make sure you run uninstall hid first if it is installed).
    3. Make sure SpaceMouse is connected before running the script
    4. (Optional) Based on the model of SpaceMouse, you might need to change the
       vendor id and product id that correspond to the device.

For Linux support, you can find open-source Linux drivers and SDKs online.
    See http://spacenav.sourceforge.net/

"""

import time
import threading
from collections import namedtuple
import numpy as np
# try:
import hid
# except ModuleNotFoundError as exc:
#     raise ImportError("Unable to load module hid, required to interface with SpaceMouse. "
#                       "Only Mac OS X is officially supported. Install the additional "
#                       "requirements with `pip install -r requirements-ik.txt`") from exc
import math

AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

SPACE_MOUSE_SPEC = {
    "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
    "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
    "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
    "roll": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
    "pitch": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
    "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
}

def unit_vector(data, axis=None, out=None):
    """
    Returns ndarray normalized by length, i.e. eucledian norm, along axis.

    Examples:

        >>> v0 = numpy.random.random(3)
        >>> v1 = unit_vector(v0)
        >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
        True
        >>> v0 = numpy.random.rand(5, 4, 3)
        >>> v1 = unit_vector(v0, axis=-1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = unit_vector(v0, axis=1)
        >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
        >>> numpy.allclose(v1, v2)
        True
        >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float32)
        >>> unit_vector(v0, axis=1, out=v1)
        >>> numpy.allclose(v1, v2)
        True
        >>> list(unit_vector([]))
        []
        >>> list(unit_vector([1.0]))
        [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float32, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data * data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    """
    Returns matrix to rotate about axis defined by point and direction.

    Examples:

        >>> angle = (random.random() - 0.5) * (2*math.pi)
        >>> direc = numpy.random.random(3) - 0.5
        >>> point = numpy.random.random(3) - 0.5
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
        >>> is_same_transform(R0, R1)
        True
        >>> R0 = rotation_matrix(angle, direc, point)
        >>> R1 = rotation_matrix(-angle, -direc, point)
        >>> is_same_transform(R0, R1)
        True
        >>> I = numpy.identity(4, numpy.float32)
        >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
        True
        >>> numpy.allclose(2., numpy.trace(rotation_matrix(math.pi/2,
        ...                                                direc, point)))
        True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(
        ((cosa, 0.0, 0.0), (0.0, cosa, 0.0), (0.0, 0.0, cosa)), dtype=np.float32
    )
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        (
            (0.0, -direction[2], direction[1]),
            (direction[2], 0.0, -direction[0]),
            (-direction[1], direction[0], 0.0),
        ),
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float32, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def to_int16(y1, y2):
    """Convert two 8 bit bytes to a signed 16 bit integer."""
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def scale_to_control(x, axis_scale=350., min_v=-1.0, max_v=1.0):
    """Normalize raw HID readings to target range."""
    x = x / axis_scale
    x = min(max(x, min_v), max_v)
    return x


def convert(b1, b2):
    """Converts SpaceMouse message to commands."""
    return scale_to_control(to_int16(b1, b2))


class SpaceMouse():
    """A minimalistic driver class for SpaceMouse with HID library."""

    def __init__(self, vendor_id=9583, product_id=50741):
        """Initialize a SpaceMouse handler.

        Args:
            vendor_id: HID device vendor id
            product_id: HID device product id

        Note:
            Use hid.enumerate() to view all USB human interface devices (HID).
            Make sure SpaceMouse is detected before running the script.
            You can look up its vendor/product id from this method.
        """

        print("Opening SpaceMouse device")
        self.device = hid.device()
        # for x in hid.enumerate():
        #     print()
        #     for key in sorted(x.keys()):
        #         print(key, x[key])
        self.device.open(vendor_id, product_id)  # SpaceMouse
        # self.device.open(1133, 50732)
        # self.device.open(1452, 627)

        print("Manufacturer: %s" % self.device.get_manufacturer_string())
        print("Product: %s" % self.device.get_product_string())

        self._display_controls()

        # pause = input()

        self.single_click_and_hold = False

        self._control = [0., 0., 0., 0., 0., 0.]
        self._right = 0
        self.rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
        self._enabled = False

        # launch a new listener thread to listen to SpaceMouse
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

        self.start_control()

    def _device_info(self):
        pass

    def _display_controls(self):
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("Control", "Command")
        print_command("Right button", "reset simulation")
        print_command("Left button (hold)", "close gripper")
        print_command("Move mouse laterally", "move arm horizontally in x-y plane")
        print_command("Move mouse vertically", "move arm vertically")
        print_command(
            "Twist mouse about an axis", "rotate arm about a corresponding axis"
        )
        print_command("ESC", "quit")
        print("")

    def _reset_internal_state(self):
        """
        Resets internal state of controller, except for the reset signal.
        """
        self.rotation = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])

    def start_control(self):
        """
        Method that should be called externally before controller can
        start receiving commands.
        """
        self._reset_internal_state()
        self._reset_state = 0
        self._enabled = True

    def get_controller_state(self):
        """Returns the current state of the 3d mouse, a dictionary of pos, orn, grasp, and reset."""
        dpos = self.control[:3] * 0.005
        roll, pitch, yaw = self.control[3:] * 0.005

        # convert RPY to an absolute orientation
        drot1 = rotation_matrix(angle=-pitch, direction=[1., 0, 0], point=None)[:3, :3]
        drot2 = rotation_matrix(angle=roll, direction=[0, 1., 0], point=None)[:3, :3]
        drot3 = rotation_matrix(angle=yaw, direction=[0, 0, 1.], point=None)[:3, :3]

        self.rotation = self.rotation.dot(drot1.dot(drot2.dot(drot3)))

        return dict(
            dpos=dpos, rotation=self.rotation, raw_drotation=np.array([roll, pitch, yaw]),
            grasp=self._right, reset=self.single_click_and_hold
        )

    def run(self):
        """Listener method that keeps pulling new messages."""

        t_last_click = -1

        while True:
            d = self.device.read(14)
            if d is not None and self._enabled:
                if d[0] == 1:  ## readings from 6-DoF sensor
                    self.y = convert(d[1], d[2])
                    self.x = convert(d[3], d[4])
                    self.z = convert(d[5], d[6]) * -1.0

                    # self.roll = convert(d[7], d[8])
                    # self.pitch = convert(d[9], d[10])
                    # self.yaw = convert(d[11], d[12])

                    # self._control = [
                    #     self.x,
                    #     self.y,
                    #     self.z,
                    #     self.roll,
                    #     self.pitch,
                    #     self.yaw,
                    # ]

                elif d[0] == 2: ## readings from 6-DoF sensor
                    # self.y = convert(d[1], d[2])
                    # self.x = convert(d[3], d[4])
                    # self.z = convert(d[5], d[6]) * -1.0

                    self.roll = convert(d[1], d[2])
                    self.pitch = convert(d[3], d[4])
                    self.yaw = convert(d[5], d[6])
                    try:
                        self._control = [
                            self.x,
                            self.y,
                            self.z,
                            self.roll,
                            self.pitch,
                            self.yaw,
                        ]
                    except:
                        exit('Cannot launch spacemouse. Try again')

                elif d[0] == 3:  ## readings from the side buttons
                    # print(d)
                    # press left button
                    if d[1] == 1:
                        t_click = time.time()
                        elapsed_time = t_click - t_last_click
                        t_last_click = t_click
                        self.single_click_and_hold = True

                    # release left button
                    if d[1] == 0:
                        self.single_click_and_hold = False
                        self._right = 0

                    # right button is for reset
                    if d[1] == 2:
                        self._right = 1
                        # self._enabled = False
                        # self._reset_internal_state()

    @property
    def control(self):
        """Returns 6-DoF control."""
        return np.array(self._control)

    @property
    def control_gripper(self):
        """Maps internal states into gripper commands."""
        if self.single_click_and_hold:
            return 1.0
        return 0


if __name__ == "__main__":

    space_mouse = SpaceMouse()
    for i in range(1000):
        print(space_mouse.control, space_mouse.control_gripper)
        time.sleep(0.02)
