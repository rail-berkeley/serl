#!/usr/bin/env python3

from __future__ import annotations

from abc import abstractmethod
from collections import deque
from threading import Lock
from typing import Any, Dict, List, Tuple, Union

##############################################################################


class DataStoreBase:
    def __init__(self, capacity: int):
        self.capacity = capacity

    @abstractmethod
    def latest_data_id() -> Any:
        """Return the id of the latest data"""
        pass

    @abstractmethod
    def get_latest_data(self, from_id: Any) -> List[Any]:
        """
        provide the all data from the given id
            :return a list of data
        """
        pass

    @abstractmethod
    def __len__(self):
        """Length of the valid data in the store"""
        pass

    @abstractmethod
    def insert(self, data: Any):
        pass

    def batch_insert(self, batch_data: List[Any]):
        """
        Insert a batch of data by using `insert()`
          :param batch_data: a list of data

        NOTE: override this function to create a more efficient
        implementation for custom data store, e.g. slicing in np
        """
        if len(batch_data) > self.capacity:
            batch_data = batch_data[-self.capacity :]
        for data in batch_data:
            self.insert(data)


##############################################################################


class QueuedDataStore(DataStoreBase):
    """
    A simple queue-based data store. This can be used as a simple
    queue for sending a sequence of data to the trainer server replay buffer
    """

    def __init__(self, capacity: int):
        DataStoreBase.__init__(self, capacity)
        self._seq_id_queue = deque(maxlen=capacity)
        self._data_queue = deque(maxlen=capacity)
        self.latest_seq_id = -1
        self._lock = Lock()

    def latest_data_id(self) -> int:
        return self.latest_seq_id

    def insert(self, data: Any):
        self._lock.acquire()
        self.latest_seq_id += 1
        self._seq_id_queue.append(self.latest_seq_id)
        self._data_queue.append(data)
        self._lock.release()

    def get_latest_data(self, from_id: int) -> List[Any]:
        self._lock.acquire()
        return_data = []
        if not self._seq_id_queue or from_id >= self.latest_seq_id:
            pass
        elif from_id < self._seq_id_queue[0]:
            return_data = list(self._data_queue)
        else:
            # calc the index where the required data starts in the queue.
            start_idx = from_id - self._seq_id_queue[0] + 1
            return_data = list(self._data_queue)[start_idx:]
        self._lock.release()
        return return_data

    def __len__(self):
        return len(self._seq_id_queue)
