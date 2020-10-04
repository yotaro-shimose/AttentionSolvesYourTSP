from multiprocessing import Process, Queue, Pipe, Lock
from cpprb import ReplayBuffer as CPPRB


class Server(Process):
    def __init__(self, size, env_dict):
        super().__init__()

        self.queue = Queue()
        self.client_pipe, self.server_pipe = Pipe()
        self.env_dict = env_dict
        self.buffer = CPPRB(size, env_dict=env_dict)
        self.parameter = None
        self.lock = Lock

    def run(self):
        while True:
            cmd, *args = self.queue.get()
            if cmd == "add":
                self._add(*args)
            elif cmd == "sample":
                self.server_pipe.send(self._sample(*args))
            elif cmd == "upload":
                self._upload(*args)
            elif cmd == "download":
                self.server_pipe.send(self._download())
            else:
                raise ValueError(
                    f"Parameter Server got an unexpected command {cmd}")

    def _download(self):
        return self.parameter

    def _upload(self, parameter):
        self.parameter = parameter

    def _add(self, data):
        for d in data:
            label_array = list(self.env_dict.keys())
            data_dict = {key: value for key, value in zip(label_array, d)}
            self.buffer.add(**data_dict)

    def _sample(self, size):
        return self.buffer.sample(size)

    def download(self):
        cmd = "download"
        self.lock.acquire()
        self.queue.put((cmd, None))
        weights = self.client.pipe.recv()
        self.lock.release()
        return weights

    def upload(self, parameter):
        cmd = "upload"
        self.queue.put((cmd, parameter))

    def add(self, data):
        cmd = "add"
        self.queue.put((cmd, data))

    def sample(self, size):
        cmd = "sample"
        self.lock.acquire()
        self.queue.put((cmd, size))
        sample = self.client.pipe.recv()
        self.lock.release()
        return sample
