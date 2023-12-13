#!/usr/bin/env python3

import argparse
import logging
import pickle
import zlib
from typing import Callable, Dict, Optional

import zmq
from typing_extensions import Protocol

##############################################################################


class PairServer:
    def __init__(self, port=5557, log_level=logging.DEBUG):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind(f"tcp://*:{port}")

        self.callback = None

        logging.basicConfig(level=log_level)
        logging.debug(f"Pair server is listening on port {port}")

    def register_callback(self, callback: Callable):
        self.callback = callback

    def broadcast(self, message: dict):
        serialized = pickle.dumps(message)
        serialized = zlib.compress(serialized)
        self.socket.send(serialized)

    def run(self):
        while True:
            try:
                # Wait for the next message from the client
                message = self.socket.recv(flags=zmq.NOBLOCK)
                message = zlib.decompress(message)
                message = pickle.loads(message)
                logging.debug(f"Received message: {message}")

                if self.callback:
                    self.callback(message)
            except zmq.Again as e:
                pass


##############################################################################


class PairClient:
    def __init__(self, ip: str, port=5557, log_level=logging.DEBUG):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect(f"tcp://{ip}:{port}")

        self.callback = None

        logging.basicConfig(level=log_level)
        logging.debug(f"Pair client is connecting to {ip}:{port}")

    def register_callback(self, callback: Callable):
        self.callback = callback

    def broadcast(self, message: dict):
        serialized = pickle.dumps(message)
        serialized = zlib.compress(serialized)
        self.socket.send(serialized)

    def run(self):
        while True:
            try:
                # Wait for the next message from the server
                message = self.socket.recv(flags=zmq.NOBLOCK)
                message = zlib.decompress(message)
                message = pickle.loads(message)
                logging.debug(f"Received message: {message}")

                if self.callback:
                    self.callback(message)
            except zmq.Again as e:
                pass


##############################################################################

if __name__ == "__main__":
    # NOTE: This is just for Testing
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5557)
    args = parser.parse_args()

    def message_handler(msg):
        time.sleep(0.5)
        print(f"Callback received: {msg}")

    if args.server:
        ps = PairServer(port=args.port)
        ps.register_callback(message_handler)
        ps.broadcast({"server_message": "Hello from Server!"})
        ps.run()
    elif args.client:
        pc = PairClient(ip=args.ip, port=args.port)
        pc.register_callback(message_handler)
        pc.broadcast({"client_message": "Hello from Client!"})
        pc.run()
    else:
        raise Exception("Must specify --server or --client")
