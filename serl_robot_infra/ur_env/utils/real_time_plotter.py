from multiprocessing.connection import Listener, Client
import numpy as np
from threading import Thread, Lock
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import use
from time import time_ns

use("TkAgg")


class DataClient:
    def __init__(self):
        self.conn = None
        print("Opening Client to Realtime Plotter")
        self.conn = Client(('localhost', 6000))

    def send(self, msg):
        self.conn.send(msg)

    def __del__(self):
        if self.conn:
            self.conn.send("close")
            self.conn.close()


class DataListener:
    def __init__(self, shape, verbose=True):
        self.listener = Listener(('localhost', 6000))
        self.data = np.zeros(shape)
        self.all_data = []
        self.lock = Lock()
        self.verbose = verbose
        self.last_time = time_ns() * 1e-6
        print("listener opened on localhost:6000, not connected yet")

        # get connection
        self.listener_thread = Thread(target=self._listen, daemon=True).start()  # listen here

    def _listen(self):
        self.conn = self.listener.accept()
        print('connection accepted from', self.listener.last_accepted)

        while True:
            try:
                msg = self.conn.recv()
                if isinstance(msg, str) and msg == "close":
                    print("connection closed, terminating...")
                    return
                with self.lock:
                    self.data = np.roll(self.data, -1, axis=1)
                    self.data[:, -1] = msg
                    self.all_data.append(msg)
                if self.verbose:
                    now = time_ns() * 1e-6
                    print(f"{now - self.last_time} ms   {msg}")
                    self.last_time = now
            except EOFError:
                print("listener Thread terminated!")
                return

    def get_data(self):
        with self.lock:
            return self.data

    def save_history(self):
        with open('plotting_data.npy', 'wb') as f:  # save run
            np.save(f, np.array(self.all_data))


class RealTimePlotter:
    def __init__(self, plots=3, horizon=50, limit=3, figsize=(12, 8)):
        self.data_listener = None
        self.fig, self.axes = plt.subplots(plots, 1, figsize=figsize)
        self.lines, self.text, colors = [], [], plt.rcParams["axes.prop_cycle"]()
        for ax in self.axes:
            ax.set_ylim(-limit, limit)
            color = next(colors)["color"]
            line, = ax.plot(np.arange(horizon), np.zeros(horizon), color, animated=True)
            text = ax.text(0.91, 0.76, "0.0000", fontsize=14, color=color, transform=ax.transAxes)
            self.lines.append(line)
            self.text.append(text)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2)

    def animate(self, j):
        data = self.data_listener.get_data()
        for i, line in enumerate(self.lines):
            line.set_ydata(data[i])
        for i, text in enumerate(self.text):
            text.set_text(str(round(data[i, -1], 4)))
        return *self.lines, *self.text

    def set_listener(self, data_listener):
        self.data_listener = data_listener

    def start_animation(self):
        ani = animation.FuncAnimation(self.fig, self.animate, interval=50, blit=True, save_count=5)
        plt.show()

    def close(self):
        plt.close("all")


if __name__ == '__main__':
    # only a test
    plotter = RealTimePlotter(plots=6, horizon=30, figsize=(12, 8))
    fetcher = DataListener((6, 30))
    plotter.set_listener(fetcher)
    plotter.start_animation()
    # fetcher.save_history()
