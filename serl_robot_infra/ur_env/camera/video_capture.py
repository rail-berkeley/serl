import queue
import threading
import time


class VideoCapture:
    def __init__(self, cap, name=None):
        if name is None:
            name = cap.name
        self.name = name
        self.q = queue.Queue()
        self.cap = cap
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.enable = True
        self.t.start()

        # read frames as soon as they are available, keeping only most recent one

    def _reader(self):
        while self.enable:
            time.sleep(0.01)
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get(timeout=5)

    def close(self):
        self.enable = False
        self.t.join()
        self.cap.close()
