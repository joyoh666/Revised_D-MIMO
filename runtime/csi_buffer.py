from collections import deque

class CSIBuffer:
    def __init__(self,maxlen=5):
        self.buffer = deque(maxlen=maxlen)

    def append(self, frame):
        self.buffer.append(frame)

    def is_ready(self):
        return len(self.buffer) == self.buffer.maxlen

    def get_window(self):
        return list(self.buffer)

    def get_len(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()