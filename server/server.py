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
                self.add(*args)
            elif cmd == "sample":
                self.server_pipe.send(self.sample(*args))
            elif cmd == "upload":
                self.upload(*args)
            elif cmd == "download":
                self.server_pipe.send(self.download())
            else:
                raise ValueError(
                    f"Parameter Server got an unexpected command {cmd}")

    def download(self):
        return self.parameter

    def upload(self, parameter):
        self.parameter = parameter

    def add(self, data):
        for d in data:
            label_array = ["obs", "act", "rew", "next_obs", "done"]
            data_dict = {key: value for key, value in zip(label_array, d)}
            self.buffer.add(**data_dict)

    def sample(self, size):
        return self.buffer.sample(size)

    def get_access(self):
        return self.queue, self.client_pipe
