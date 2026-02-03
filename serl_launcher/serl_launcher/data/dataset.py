from functools import partial           # unused.
from typing import Dict, Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

# frozen_dict is an immutable & nested dictionary-like structure to manage parameters and states in NNs. Key in Flax, model params passed explicitly vs stored in mutable objects.
from flax.core import frozen_dict
from gym.utils import seeding

# Nested data structures where the leaves are NumPy arrays, and internal nodes are dictionaries with string keys.
DataType = Union[np.ndarray, Dict[str, "DataType"]]
DatasetDict = Dict[str, DataType]


# Utility functions to check lengths and subselect data from a dataset dictionary.
def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    """
    Check the lengths of items in a dataset dictionary. 
    
    Upon initializing a Dataset, _check_lengths is invoked to assert that all data arrays are of equal length. This is critical because:
    The Dataset assumes a uniform length across features to support indexing, sampling, and batching.
    Inconsistent lengths would lead to silent errors or runtime failures during sampling, model training, or evaluation.

    If all items are of the same length, return that length.
    If items are of different lengths, raise an assertion error.
    If the dataset is empty, return 0.
    If the dataset is not a dictionary, raise a TypeError.
    Args:
        dataset_dict (DatasetDict): The dataset dictionary to check.
        dataset_len (Optional[int]): The length to compare against, if provided.
    Returns:
        int: The length of the dataset if all items are of the same length.
    Raises:
        TypeError: If the dataset is not a dictionary or contains unsupported types.
    """
    
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        
        else:
            raise TypeError("Unsupported type.")
    return dataset_len


def _subselect(dataset_dict: DatasetDict, index: np.ndarray) -> DatasetDict:
    """
    Subselect enables flexible, consistent indexing into complex datasets to either split or filter data.
    It is especially important when working with nested dictionary-based datasets — a common structure in modern machine learning.
    Our dataset will be deeply structure and we want indexing to be applied consistently at every depth.
    Used by Dataset.split() and Dataset.filter() methods to extract subsets of data based on indices.

    Args:
        dataset_dict (DatasetDict): The dataset dictionary to subselect from.
        index (np.ndarray): The indices to select from the dataset.
    Returns:
        DatasetDict: A new dataset dictionary with items selected based on the index.
    Raises:
        TypeError: If the dataset contains unsupported types.
    """


    new_dataset_dict = {}
    for k, v in dataset_dict.items():
        if isinstance(v, dict):
            new_v = _subselect(v, index)

        elif isinstance(v, np.ndarray):
            new_v = v[index]

        else:
            raise TypeError("Unsupported type.")
        
        new_dataset_dict[k] = new_v
    return new_dataset_dict


def _sample(dataset_dict: Union[np.ndarray, DatasetDict], indx: np.ndarray) -> DatasetDict:
    """
    This function is used to extract a subset of data from the dataset dictionary, which can be either a NumPy array or a nested dictionary structure.
    Args:
        dataset_dict (Union[np.ndarray, DatasetDict]): The dataset dictionary or array to sample from.
        indx (np.ndarray): The indices to sample from the dataset.
    Returns:
        DatasetDict: A new dataset dictionary with items sampled based on the indices.
    Raises:
        TypeError: If the dataset is not a NumPy array or a dictionary.
    """

    if isinstance(dataset_dict, np.ndarray):
        return dataset_dict[indx]
    elif isinstance(dataset_dict, dict):
        batch = {}
        for k, v in dataset_dict.items():
            batch[k] = _sample(v, indx)
    else:
        raise TypeError("Unsupported type.")
    return batch


class Dataset(object):

    def __init__(self, 
                 dataset_dict: DatasetDict, 
                 seed: Optional[int] = None ):
        
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        # Seeding similar to OpenAI Gym:
        # https://github.com/openai/gym/blob/master/gym/spaces/space.py#L46
        self._np_random = None
        self._seed = None
        if seed is not None:
            self.seed(seed)

    # @property decorator is used here to expose np_random as a read-only attribute 
    @property
    def np_random(self) -> np.random.RandomState:
        if self._np_random is None:
            self.seed()
        return self._np_random

    def seed(self, seed: Optional[int] = None) -> list:
        """
        Set the random seed for reproducibility. Ensures a valid RNG is encapsulated behind an attribute-like interface.
        Users do not need to call self.seed() or self.get_np_random() explicitly. Just self.np_random above.

        Args:
            seed (Optional[int]): The seed to set. If None, a random seed will be generated.
        Returns:
            list: A list containing the seed used.
        """
        self._np_random, self._seed = seeding.np_random(seed)
        return [self._seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(self,
               batch_size: int,
               keys: Optional[Iterable[str]] = None,
               indx: Optional[np.ndarray] = None,
              ) -> frozen_dict.FrozenDict:
        """
        Sample a random batch of data from the dataset. 
        
        This method allows for flexible sampling of data, either by specifying keys or using random indices.
        This is useful for training models, where you might want to sample a batch of data points from a larger dataset.
        Args:
            batch_size (int): The number of samples to return.
            keys (Optional[Iterable[str]]): The keys to sample from the dataset. If None, all keys will be sampled.
            indx (Optional[np.ndarray]): Specific indices to sample from. If None, random indices will be generated.
        Returns:
            frozen_dict.FrozenDict: A frozen dictionary containing the sampled data.
        """
      
        if indx is None:
            if hasattr(self.np_random, "integers"):

                # Generate batch_size num of rand ints, each sampled uniformly at random from the range [0, len(self) - 1].
                indx = self.np_random.integers(len(self), size=batch_size)
            else:
                indx = self.np_random.randint(len(self), size=batch_size)

        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            if isinstance(self.dataset_dict[k], dict):
                batch[k] = _sample(self.dataset_dict[k], indx)
            else:
                batch[k] = self.dataset_dict[k][indx]

        return frozen_dict.freeze(batch)

    def sample_jax(self, 
                   batch_size: int, 
                   keys: Optional[Iterable[str]] = None):
        """
        Sample a batch of data from the dataset using JAX. This method is optimized for performance and can be used in JAX-based training loops.
        Args:
            batch_size (int): The number of samples to return.
            keys (Optional[Iterable[str]]): The keys to sample from the dataset. If None, all keys will be sampled.
        Returns:
            Tuple[int, frozen_dict.FrozenDict]: A tuple containing the maximum index sampled and a frozen dictionary with the sampled data.
        """
        if not hasattr(self, "rng"):
            self.rng = jax.random.PRNGKey(self._seed or 42)

            if keys is None:
                keys = self.dataset_dict.keys()

            # jax_dataset_dict = {k: self.dataset_dict[k] for k in keys}
            # jax_dataset_dict = jax.device_put(jax_dataset_dict)

            @jax.jit
            def _sample_jax(rng, src, max_indx: int):
                key, rng = jax.random.split(rng)
                indx = jax.random.randint(key, (batch_size,), minval=0, maxval=max_indx)
                return (
                    rng,
                    indx.max(),
                    jax.tree.map(lambda d: jnp.take(d, indx, axis=0), src),
                )

            self._sample_jax = _sample_jax

        self.rng, indx_max, sample = self._sample_jax(
            self.rng, self.dataset_dict, len(self)
        )
        return indx_max, sample

    def split(self, ratio: float) -> Tuple["Dataset", "Dataset"]:
        """
        Split the dataset into two parts based on the given ratio. The first part will contain a fraction of the dataset specified by the ratio, and the second part will contain the rest.
        This method is useful for creating training and testing datasets, where you want to split the data into two parts for model evaluation.
        Args:
            ratio (float): The fraction of the dataset to include in the first part. Must be between 0 and 1.
        Returns:
            Tuple[Dataset, Dataset]: A tuple containing two Dataset objects, the first part and the second part of the split dataset.
        Raises:
            AssertionError: If the ratio is not between 0 and 1.
        """

        assert 0 < ratio and ratio < 1
        train_index = np.index_exp[: int(self.dataset_len * ratio)] # First part of the dataset.
        test_index = np.index_exp[int(self.dataset_len * ratio) :]  # Second part of the dataset.

        # Shuffle the indices to ensure random sampling.
        index = np.arange(len(self), dtype=np.int32)
        self.np_random.shuffle(index)
        
        train_index = index[: int(self.dataset_len * ratio)]
        test_index = index[int(self.dataset_len * ratio) :]

        train_dataset_dict = _subselect(self.dataset_dict, train_index)
        test_dataset_dict = _subselect(self.dataset_dict, test_index)
        return Dataset(train_dataset_dict), Dataset(test_dataset_dict)

    def _trajectory_boundaries_and_returns(self) -> Tuple[list, list, list]:
        """
        This method computes the boundaries of episodes in the dataset and calculates the returns for each episode.
        It identifies the start and end indices of each episode based on the 'dones' array in the dataset.
        The returns for each episode are calculated by summing the rewards within each episode.
        This is useful for reinforcement learning tasks where episodes are defined by sequences of states, actions, and rewards.
        Returns:
            Tuple[list, list, list]: A tuple containing three lists:
                - episode_starts: The starting indices of each episode.
                - episode_ends: The ending indices of each episode.
                - episode_returns: The total returns for each episode.
        """

        # Initialize lists (note plural) to store episode boundaries and returns.
        episode_starts = [0]
        episode_ends = []

        # Initialize variables to track the current episode return and a list to store returns.
        episode_return = 0
        episode_returns = []

        # Iterate through the dataset to find episode boundaries and calculate returns.
        # The dataset_dict is expected to have 'rewards' and 'dones' keys.
        for i in range(len(self)):
            episode_return += self.dataset_dict["rewards"][i]

            # If the current index indicates the end of an episode, store the return and update boundaries.
            if self.dataset_dict["dones"][i]:
                episode_returns.append(episode_return)
                episode_ends.append(i + 1)  # Store the end index of the episode including the current index.

                # If this is not the last episode, set the start of the next episode. 
                if i + 1 < len(self):
                    episode_starts.append(i + 1)
                episode_return = 0.0

        return episode_starts, episode_ends, episode_returns

    def filter(self, 
               take_top: Optional[float] = None, 
               threshold: Optional[float] = None
    ):
        """
        Filter the dataset based on episode returns. This method allows you to keep only the episodes that meet a certain return threshold or are among the top returns.
        This is useful for focusing on high-performing episodes in reinforcement learning tasks.
        Args:
            take_top (Optional[float]): If specified, keep only the top N percent of episodes based on their returns.
            threshold (Optional[float]): If specified, keep only the episodes with returns greater than or equal to this value.
        Raises:
            AssertionError: If both take_top and threshold are specified, or if neither is specified.
        """
        assert (take_top is None and threshold is not None) or (
            take_top is not None and threshold is None
        )

        # Create a tupe of lists of episode boundaries and returns.
        (
            episode_starts,
            episode_ends,
            episode_returns,
        ) = self._trajectory_boundaries_and_returns()

        # If no threshold is specified, calculate it based on the top N percent of returns.
        if take_top is not None:
            # np.percentile gives the value below which XX% of the data lies 
            threshold = np.percentile(episode_returns, 100 - take_top)

        # create a boolean index array to filter episodes based on the threshold.
        bool_indx = np.full((len(self),), False, dtype=bool)

        for i in range(len(episode_returns)):
            if episode_returns[i] >= threshold:
                bool_indx[episode_starts[i] : episode_ends[i]] = True

        # Return a new dataset dictionary containing only the episodes that meet the threshold.
        self.dataset_dict = _subselect(self.dataset_dict, bool_indx)

        # Update the dataset length after filtering.
        self.dataset_len = _check_lengths(self.dataset_dict)

    def normalize_returns(self, scaling: float = 1000):
        """
        Normalize the returns in the dataset to a specified scaling factor. This is useful for stabilizing training in reinforcement learning tasks.
        Normally done per batch of episodes before training a model to update the policy.

        Args:
            scaling (float): The scaling factor to normalize the returns. Default is 1000.
        Raises:
            AssertionError: If the dataset does not contain 'rewards' or 'dones' keys.
        """

        # Extract episode returns
        (_, _, episode_returns) = self._trajectory_boundaries_and_returns()

        # Normalize rewards in the dataset from 0-1 by dividing by the max range of returns.
        self.dataset_dict["rewards"] /= np.max(episode_returns) - np.min(episode_returns)

        # Scale rewards. Note that large scaling factors can lead to numerical instability, small scaleing factors can lead to poor learning. 
        # Scaling allows you to control the learning dynamics.
        self.dataset_dict["rewards"] *= scaling
