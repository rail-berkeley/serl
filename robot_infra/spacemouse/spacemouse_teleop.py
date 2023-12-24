"""Uses a spacemouse as action input into the environment.
You will likely have to `pip install hidapi` and Spacemouse drivers.
"""
from spacemouse import SpaceMouse

import numpy as np
import os
import argparse
import datetime
from PIL import Image
import pickle

class SpaceMouseExpert():
    def __init__(self, xyz_dims=3, xyz_remap=[0, 1, 2], xyz_scale=[1, 1, 1], rot_scale=1, all_angles = False):
        """TODO: fill in other params"""

        self.xyz_dims = xyz_dims
        self.xyz_remap = np.array(xyz_remap)
        self.xyz_scale = np.array(xyz_scale)
        self.device = SpaceMouse()
        self.grasp_input = 0.
        self.grasp_output = 1.
        self.rot_scale = rot_scale
        self.all_angles = all_angles

    def get_action(self):
        """Must return (action, valid, reset, accept)"""
        state = self.device.get_controller_state()
        dpos, rotation, rot, grasp, reset = (
            state["dpos"],
            state["rotation"],
            state["raw_drotation"],
            state["grasp"],
            state["reset"],
        )
        # detect button press
        if grasp and not self.grasp_input:
            # open/close gripper
            self.grasp_output = 1. if self.grasp_output <= 0. else -1.
        self.grasp_input = grasp
        # import pdb; pdb.set_trace()
        xyz = dpos[self.xyz_remap] * self.xyz_scale
        pitch, roll, yaw = tuple(list(rot * self.rot_scale))
        a = xyz[:self.xyz_dims]
        if self.all_angles:
            # 0 1 2, 3(grasp), 4, 5, 6
            a = np.concatenate([a, [self.grasp_output, roll, pitch, yaw]])
        else:
            a = np.concatenate([a, [self.grasp_output, yaw]])

        valid = not np.all(np.isclose(a, 0))

        return (a, valid, reset, grasp)