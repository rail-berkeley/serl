#!/usr/bin/env python3

import hashlib
import pickle
import sys
import zlib
from typing import Callable, Tuple

import cv2
import lz4.frame
import numpy as np


def mat_to_jpeg(img):
    """Compresses a numpy array into a JPEG byte array."""
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def jpeg_to_mat(buf):
    """Decompresses a JPEG byte array into a numpy array."""
    return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)


def compute_hash(obj):
    pickle_str = pickle.dumps(obj)
    # Compute MD5 hash of the string
    return hashlib.md5(pickle_str).hexdigest()


def print_size(obj):
    size = sys.getsizeof(obj)
    mb_size = size / 1024**2
    print(f"The size of the object is {mb_size} MB")


def print_error(text):
    # Red color
    print(f"\033[91m{text}\033[00m")


def print_warning(text):
    # Yellow color
    print(f"\033[93m{text}\033[00m")


def make_compression_method(compression: str) -> Tuple[Callable, Callable]:
    """
    NOTE: lz4 is faster than zlib, but zlib has better compression ratio
        :return: compress, decompress functions
            def compress(object) -> bytes
            def decompress(data) -> object
    """
    if compression == "lz4":

        def compress(data):
            return lz4.frame.compress(pickle.dumps(data))

        def decompress(data):
            return pickle.loads(lz4.frame.decompress(data))

    elif compression == "zlib":

        def compress(data):
            return zlib.compress(pickle.dumps(data))

        def decompress(data):
            return pickle.loads(zlib.decompress(data))

    else:
        raise Exception(f"Unknown compression algorithm: {compression}")
    return compress, decompress
