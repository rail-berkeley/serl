#!/usr/bin/env python3

import argparse
import logging
from threading import Lock
from typing import Dict, Optional

import zmq
from edgeml.internal.utils import make_compression_method
from typing_extensions import Protocol

##############################################################################


class CallbackProtocol(Protocol):
    def __call__(self, message: Dict) -> Dict:
        ...


class ReqRepServer:
    def __init__(
        self,
        port=5556,
        impl_callback: Optional[CallbackProtocol] = None,
        log_level=logging.DEBUG,
        compression: str = "lz4",
    ):
        """
        Request reply server
        """
        self.impl_callback = impl_callback
        self.compress, self.decompress = make_compression_method(compression)
        self.port = port
        self.reset()
        logging.basicConfig(level=log_level)
        logging.debug(f"Req-rep server is listening on port {port}")

    def run(self):
        if self.is_kill:
            logging.debug("Server is prev killed, reseting...")
            self.reset()
        while not self.is_kill:
            try:
                #  Wait for next request from client
                message = self.socket.recv()
                message = self.decompress(message)
                logging.debug(f"Received new request: {message}")

                #  Send reply back to client
                if self.impl_callback:
                    res = self.impl_callback(message)
                    res = self.compress(res)
                    self.socket.send(res)
                else:
                    logging.warning("No implementation callback provided.")
                    self.socket.send(b"World")
            except zmq.Again as e:
                continue
            except zmq.ZMQError as e:
                # Handle ZMQ errors gracefully
                if self.is_kill:
                    logging.debug("Stopping the ZMQ server...")
                    break
                else:
                    raise e

    def stop(self):
        self.is_kill = True
        self.socket.close()
        self.context.term()

    def reset(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{self.port}")
        self.socket.setsockopt(zmq.SNDHWM, 5)

        # Set a timeout for the recv method (e.g., 1.5 second)
        self.socket.setsockopt(zmq.RCVTIMEO, 1500)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.is_kill = False


##############################################################################


class ReqRepClient:
    def __init__(
        self,
        ip: str,
        port=5556,
        timeout_ms=800,
        log_level=logging.DEBUG,
        compression: str = "lz4",
    ):
        """
        :param ip: IP address of the server
        :param port: Port number of the server
        :param timeout_ms: Timeout in milliseconds
        :param log_level: Logging level, defaults to DEBUG
        :param compression: Compression algorithm, defaults to lz4
        """
        self.context = zmq.Context()
        logging.basicConfig(level=log_level)
        logging.debug(f"Req-rep client is connecting to {ip}:{port}")

        self.compress, self.decompress = make_compression_method(compression)
        self.socket = None
        self.ip, self.port, self.timeout_ms = ip, port, timeout_ms
        self._internal_lock = Lock()
        self.reset_socket()

    def reset_socket(self):
        """
        Reset the socket connection, this is needed when REQ is in a
        broken state.
        """
        if self.socket:
            self.socket.close()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.ip}:{self.port}")

    def send_msg(self, request: dict) -> Optional[str]:
        if self.socket is None or self.socket.closed:
            logging.debug("WARNING: Socket is closed, reseting...")
            return None

        serialized = self.compress(request)
        with self._internal_lock:
            try:
                self.socket.send(serialized)
                message = self.socket.recv()
                return self.decompress(message)
            except Exception as e:
                # accepts timeout exception
                logging.warning(f"Failed to send message: {e}")
                logging.debug("WARNING: No res from server. reset socket.")
                self.reset_socket()
                return None


##############################################################################

if __name__ == "__main__":
    # NOTE: This is just for Testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", action="store_true")
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5556)
    args = parser.parse_args()

    def do_something(message):
        return b"World"

    if args.server:
        ss = ReqRepServer(port=args.port, impl_callback=do_something)
        ss.run()
    elif args.client:
        sc = ReqRepClient(ip=args.ip, port=args.port)
        r = sc.send_msg({"hello": 1})
        print(r)
    else:
        raise Exception("Must specify --server or --client")
