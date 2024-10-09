import numpy as np


class TemporalActionEnsemble:
    def __init__(self, activated=True, action_shape=(7,), ensemble=None):
        if ensemble is None:
            ensemble = [0.5, 0.3, 0.2, 0.1]
        self.activated = activated
        self.ensemble = np.asarray(ensemble)
        self.buffer = np.zeros((len(ensemble), action_shape[0]))

        if activated:
            print(f"Temporal Action Ensemble enabled: {self.ensemble}")

    def reset(self):
        self.buffer[...] = 0.

    def sample(self, curr_action: np.ndarray):
        if not self.activated:
            return curr_action

        curr_action = curr_action.reshape(-1)
        assert curr_action.shape[0] == self.buffer.shape[1]

        self.buffer = np.roll(self.buffer, axis=0, shift=1)
        self.buffer[0, :] = curr_action
        return np.dot(self.ensemble, self.buffer)

    def is_activated(self):
        return self.activated
