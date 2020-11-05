from multiprocessing import Process, Queue, Pipe, Lock
from cpprb import ReplayBuffer as CPPRB


class Server(Process):
    def __init__(self, size, env_dict, min_storage=100):
        super().__init__()

        self.queue = Queue()
        self.size = size
        self.client_pipe, self.server_pipe = Pipe()
        self.env_dict = env_dict
        self.parameter = None
        self.min_storage = min_storage

        # サーバーロックオブジェクト
        self.lock = Lock()

    def run(self):
        self.buffer = CPPRB(self.size, env_dict=self.env_dict)
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
        if self.buffer.get_stored_size() < self.min_storage:
            print(
                f"stored sample {self.buffer.get_stored_size()} is smaller than mininum storage\
                     size {self.min_storage}. Returning None")
            return None
        else:
            return self.buffer.sample(size)

    def download(self):
        cmd = "download"
        self.lock.acquire()
        self.queue.put((cmd, None))
        weights = self.client_pipe.recv()
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
        sample = self.client_pipe.recv()
        self.lock.release()
        return sample
