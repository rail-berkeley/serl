import threading
import pyspacemouse
import numpy as np
from typing import Tuple


class SpaceMouseExpert:
    """
    This class provides an interface to the SpaceMouse.
    It continuously reads the SpaceMouse state and provide
    a "get_action" method to get the latest action and button state.
    """

    def __init__(self):
        pyspacemouse.open()

        self.state_lock = threading.Lock()
        self.latest_data = {"action": np.zeros(6), "buttons": [0, 0]}
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_spacemouse)
        self.thread.daemon = True
        self.thread.start()

    def _read_spacemouse(self):
        while True:
            state = pyspacemouse.read()
            with self.state_lock:
                self.latest_data["action"] = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                )  # spacemouse axis matched with robot base frame
                self.latest_data["buttons"] = state.buttons

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self.state_lock:
            return self.latest_data["action"], self.latest_data["buttons"]
