from multiprocessing import Process, Queue, Pipe
from cpprb import ReplayBuffer as CPPRB


class Server(Process):
    def __init__(self, size, observation_shape):
        super().__init__()
        for value in observation_shape:
            assert isinstance(value, int)

        self.queue = Queue()
        self.client_pipe, self.server_pipe = Pipe()
        env_dict = {
            "obs": {"shape": observation_shape},
            "act": {},
            "rew": {},
            "next_obs": {"shape": observation_shape},
            "done": {},
        }
        self.buffer = CPPRB(size, env_dict=env_dict)
        self.parameter = None

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
            label_array = ["obs", "act", "rew", "next_obs", "done"]
            data_dict = {key: value for key, value in zip(label_array, d)}
            self.buffer.add(**data_dict)

    def _sample(self, size):
        return self.buffer.sample(size)

    def download(self):
        cmd = "download"
        self.queue.put((cmd, None))
        return self.client_pipe.recv()

    def upload(self, parameter):
        cmd = "upload"
        self.queue.put((cmd, parameter))

    def add(self, data):
        cmd = "add"
        self.queue.put((cmd, data))

    def sample(self, size):
        cmd = "sample"
        self.queue.put((cmd, size))
        return self.client_pipe.recv()
