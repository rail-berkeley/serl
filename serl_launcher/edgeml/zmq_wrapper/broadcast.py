#!/usr/bin/env python3

import argparse
import logging
import threading
from typing import Callable

import zmq
from edgeml.internal.utils import make_compression_method

##############################################################################


class BroadcastServer:
    def __init__(self, port=5557, log_level=logging.DEBUG, compression: str = "lz4"):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 3)  # queue size 3 for send buffer
        self.socket.bind(f"tcp://*:{port}")
        self.compress, _ = make_compression_method(compression)
        logging.basicConfig(level=log_level)
        logging.debug(f"Publisher server is broadcasting on port {port}")

    def broadcast(self, message: dict):
        serialized = self.compress(message)
        self.socket.send(serialized)


##############################################################################


class BroadcastClient:
    def __init__(
        self, ip: str, port=5557, log_level=logging.DEBUG, comppression: str = "lz4"
    ):
        self.context = zmq.Context()
        logging.basicConfig(level=log_level)
        logging.debug(f"Subscriber client is connecting to {ip}:{port}")

        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{ip}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages
        _, self.decompress = make_compression_method(comppression)
        # Set a timeout for the recv method (e.g., 1.5 second)
        self.socket.setsockopt(zmq.RCVTIMEO, 1500)

        # ZMQ will queue up msg with slow connections. This resulted in
        # the new updated weights being queued up, actor is not able to learn
        # https://stackoverflow.com/questions/59542620/zmq-drop-old-messages
        self.socket.setsockopt(zmq.CONFLATE, True)
        self.socket.setsockopt(zmq.RCVHWM, 3)  # queue size 3 for receive buffer
        self.is_kill = False
        self.thread = None

    def async_start(self, callback: Callable[[dict], None]):
        def async_listen():
            while not self.is_kill:
                try:
                    serialized = self.socket.recv()
                    message = self.decompress(serialized)
                    callback(message)
                except zmq.Again:
                    # Timeout occurred, check is_kill flag again
                    continue

        self.thread = threading.Thread(target=async_listen)
        self.thread.start()

    def stop(self):
        self.is_kill = True  # kill the thread
        if self.thread:
            self.thread.join()  # ensure the thread exits
        self.socket.close()


##############################################################################

if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    args = parser.parse_args()

    if args.server:
        ps = BroadcastServer(port=args.port)
        while True:
            ps.broadcast({"message": "Hello World"})
            time.sleep(1)
    elif args.client:
        pc = BroadcastClient(ip=args.ip, port=args.port)
        pc.async_start(callback=lambda x: print(x))
        print("Listening... asynchonously")
    else:
        raise Exception("Must specify --server or --client")
