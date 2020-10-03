from server.server import Server
import numpy as np


def create_dummy(size):
    return [(
        np.random.random((14, 2)),
        np.random.randint(5),
        np.random.random(),
        np.random.random((14, 2)),
        bool(np.random.randint(2)),
    ) for _ in range(size)]


def test_server():
    server = Server(
        size=200000, observation_shape=(14, 2))
    queue, pipe = server.get_access()
    server.start()
    size = 100
    data = create_dummy(size)
    cmd = "add"
    queue.put((cmd, data))
    cmd = "sample"
    queue.put((cmd, size))
    sample = pipe.recv()
    # obs
    assert sample["obs"].shape == (size, 14, 2)
    # act
    assert sample["act"].shape == (size, 1)
    # rew
    assert sample["rew"].shape == (size, 1)
    # next_obs
    assert sample["next_obs"].shape == (size, 14, 2)
    # done
    assert sample["done"].shape == (size, 1)

    cmd = "download"
    queue.put((cmd, None))
    weight = pipe.recv()
    assert weight is None
    weight_input = [np.random.random((5, 5)) for _ in range(10)]
    cmd = "upload"
    queue.put(("upload", weight_input))
    cmd = "download"
    queue.put((cmd, None))
    weight = pipe.recv()
    for w1, w2 in zip(weight, weight_input):
        assert np.sum(((w1 == w2).astype(np.int) - 1) * (-1)) == 0

    server.terminate()
    server.join()
    server.close()
