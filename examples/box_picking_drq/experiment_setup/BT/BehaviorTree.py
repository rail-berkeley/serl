import numpy as np
from queue import Queue


class TreeState():
    def __init__(self):
        self.down = np.array([0., 0., 1., 0., 0., 0., 0.])
        self.up = -self.down
        self.suck = np.array([0., 0., 1., 0., 0., 0., 1.])
        self.random_direction = np.zeros_like(self.down)
        self.random_orientation = np.zeros_like(self.down)
        self.re_sample()

        self.current = np.zeros_like(self.down)

    def re_sample(self):
        rand = np.random.rand(2, 2) - 0.5
        self.random_direction[0:2] = rand[0] / np.linalg.norm(rand[0])
        self.random_orientation[3:5] = rand[1] / np.linalg.norm(rand[1])

    def reset(self):
        self.current = self.down

    def __call__(self, *args, **kwargs):
        return self.current.copy()


class BehaviorTree():
    """
    simple behavior tree for picking boxes

    start: move down
    if force in z: suck
        if not successful:
            maybe orientation rot before? (test)
            move up, random direction xy, goto start
        if successful:
            move up, wait for end
    """

    def __init__(self):
        self.tree_state: TreeState = TreeState()
        self.queue = Queue()

    def reset(self):
        self.tree_state.reset()
        print("down")
        return self.tree_state()

    def sample_actions(self, observations):
        obs = observations["state"].reshape(-1)
        if not self.queue.empty():
            return self.queue.get()

        if obs[8] > 0.5:
            if np.all(self.tree_state.current == self.tree_state.up):
                pass
            else:
                print("go up")
                self.tree_state.current = self.tree_state.up

        elif obs[11] < -3.:  # force check
            if obs[8] < -0.5:  # if sucking
                print("do random direction")
                return self._fill_random_xy_queue()
            else:
                print("suck")
                self.tree_state.current = self.tree_state.suck
                return self._fill_suck_queue()
        else:
            self.tree_state.reset()

        return self.tree_state()

    def _fill_random_xy_queue(self):
        for _ in range(4):
            self.queue.put(self.tree_state.up)
        self.tree_state.re_sample()
        for _ in range(6):
            self.queue.put(self.tree_state.random_direction)

        return self.queue.get()

    def _fill_suck_queue(self):
        for _ in range(3):
            self.queue.put(self.tree_state.suck)
        return self.queue.get()
