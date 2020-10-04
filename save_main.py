import numpy as np
from multiprocessing import Process, Queue, Pipe
from agents.actor import Actor
from agents.learner import Learner

from gat.environment.env import Env
from gat.model.encoder import Encoder
from gat.model.decoder import Decoder
from mcts.mcts import MCTS
from server.server import Server

# NNパラメータ
D_MODEL = 128
D_KEY = 16
N_HEAD = 8
DEPTH = 2
TH_RANGE = 10
LEARNING_RATE = 9.0e-5
WEIGHT_BALANCER = 0.12

# 学習アルゴリズムパラメータ
EPOCH_NUM = 10000
NUM_MCTS_SIMS = 300
STEP_NUM = 10
BATCH_SIZE = 5
SIGNIFICANCE = 0.05
EVALUATION_NUM = 15
GAMMA = 0.99999
SEARCH_NUM = 10
ACTOR_UPLOAD_INTERVAL = 1000
DOWNLOAD_INTERVAL = 10
BUFFER_SIZE = 20000

SYNCHRONIZE_FREQ = 10
LEARNER_UPLOAD_INTERVAL = 50

# 環境用パラメータ
SIZE_MIN = 14
SIZE_MAX = 15


def env_builder():
    return Env()


def encoder_builder(d_model=D_MODEL, d_key=D_KEY, n_heads=N_HEAD, depth=DEPTH, weight_balancer=WEIGHT_BALANCER):
    return Encoder(d_model, d_key, n_heads, depth, weight_balancer)


def decoder_builder(d_model=D_MODEL, d_key=D_KEY, n_heads=N_HEAD, weight_balancer=WEIGHT_BALANCER):
    return Decoder(d_model, d_key, n_heads, weight_balancer)


if __name__ == "__main__":
    env_dict = {
        "graph": {"shape": (14, 2)},
        "traj": {"shape": (14, )},
        "act": {"dtype": np.int},
        "rew": {"dtype": np.float},
        "next_graph": {"shape": (14, 2)},
        "next_traj": {"shape": (14, )},
        "done": {"dtype": np.bool},
        "Q": {"shape": (14, )}
    }
    server = Server(
        size=BUFFER_SIZE, env_dict=env_dict)
    server.start()

    # learner = Learner(
    #     encoder_builder,
    #     decoder_builder,
    #     server,
    #     BATCH_SIZE,
    #     GAMMA,
    #     SYNCHRONIZE_FREQ,
    #     LEARNER_UPLOAD_INTERVAL
    # )
    # learner_process = Process(target=learner.start)
    # learner_process.start()
    actor_process_list = []
    for _ in range(5):
        actor = Actor(
            env_builder,
            encoder_builder,
            decoder_builder,
            server,
            GAMMA,
            STEP_NUM,
            ACTOR_UPLOAD_INTERVAL,
            DOWNLOAD_INTERVAL,
            SIZE_MIN,
            SIZE_MAX,
            D_MODEL,
            D_KEY,
            N_HEAD,
            DEPTH,
            LEARNING_RATE,
            WEIGHT_BALANCER,
            SEARCH_NUM
        )
        p = Process(target=actor.start)
        actor_process_list.append(p)
        p.start()

    for actor_process in actor_process_list:
        actor_process.join()
    # learner_process.join()
