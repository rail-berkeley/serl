# Test the SpaceMouseExpert class

import time
from serl_robot_infra.franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
import numpy as np
np.set_printoptions(precision=3, suppress=True)

spacemouse = SpaceMouseExpert()
while True:
    action, buttons = spacemouse.get_action()
    print(f'Spacemouse action: {action}, buttons: {buttons}')
    time.sleep(0.1)
