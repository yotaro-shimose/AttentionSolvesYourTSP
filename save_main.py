import numpy as np
from multiprocessing import Process
from agents.actor import Actor
from agents.learner import Learner

from gat.environment.env import Env
from gat.model.encoder import Encoder
from gat.model.decoder import Decoder
from server.server import Server

# NNパラメータ
D_MODEL = 128
D_KEY = 16
N_HEAD = 8
DEPTH = 4
TH_RANGE = 10
LEARNING_RATE = 1.0e-3
WEIGHT_BALANCER = 1

# 学習アルゴリズムパラメータ
EPOCH_NUM = 10000
NUM_MCTS_SIMS = 300
BATCH_SIZE = 5
SIGNIFICANCE = 0.05
EVALUATION_NUM = 15
GAMMA = 0.99999
SEARCH_NUM = 10
ACTOR_UPLOAD_INTERVAL = 10
DOWNLOAD_INTERVAL = 10
BUFFER_SIZE = 20000

SYNCHRONIZE_INTERVAL = 10
LEARNER_UPLOAD_INTERVAL = 50

# 環境用パラメータ
SIZE_MIN = 14
SIZE_MAX = 15


def env_builder():
    return Env()


def encoder_builder(
    d_model=D_MODEL,
    d_key=D_KEY,
    n_heads=N_HEAD,
    depth=DEPTH,
    weight_balancer=WEIGHT_BALANCER
):
    return Encoder(d_model, d_key, n_heads, depth, weight_balancer)


def decoder_builder(
    d_model=D_MODEL,
    d_key=D_KEY,
    n_heads=N_HEAD,
    weight_balancer=WEIGHT_BALANCER
):
    return Decoder(d_model, d_key, n_heads, weight_balancer)


if __name__ == "__main__":
    env_dict = {
        "graph": {"shape": (14, 2), "dtype": np.float32},
        "traj": {"shape": (14, ), "dtype": np.int32},
        "act": {"dtype": np.int32},
        "rew": {"dtype": np.float32},
        "next_graph": {"shape": (14, 2), "dtype": np.float32},
        "next_traj": {"shape": (14, ), "dtype": np.int32},
        "done": {"dtype": np.bool},
        "Q": {"shape": (14, )}
    }
    server = Server(
        size=BUFFER_SIZE, env_dict=env_dict)
    server.start()

    learner = Learner(
        encoder_builder=encoder_builder,
        decoder_builder=decoder_builder,
        server=server,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        synchronize_interval=SYNCHRONIZE_INTERVAL,
        upload_interval=LEARNER_UPLOAD_INTERVAL,
    )
    learner_process = Process(target=learner.start)
    learner_process.start()

    actor_process_list = []
    for _ in range(2):
        actor = Actor(
            env_builder=env_builder,
            encoder_builder=encoder_builder,
            decoder_builder=decoder_builder,
            server=server,
            gamma=GAMMA,
            upload_interval=ACTOR_UPLOAD_INTERVAL,
            download_interval=DOWNLOAD_INTERVAL,
            size_min=SIZE_MIN,
            size_max=SIZE_MAX,
            search_num=SEARCH_NUM
        )
        p = Process(target=actor.start)
        actor_process_list.append(p)
        p.start()

    for actor_process in actor_process_list:
        actor_process.join()
    learner_process.join()
