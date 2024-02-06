""" Test the spacemouse output. """

import time
import numpy as np
from serl_robot_infra.franka_env.spacemouse.spacemouse_expert import SpaceMouseExpert
np.set_printoptions(precision=3, suppress=True)

def test_spacemouse():
    """ Test the SpaceMouseExpert class. """
    spacemouse = SpaceMouseExpert()
    while True:
        action, buttons = spacemouse.get_action()
        print(f'Spacemouse action: {action}, buttons: {buttons}')
        time.sleep(0.1)

def main():
    """ Call spacemouse test."""
    test_spacemouse()

if __name__ == '__main__':
    main()
