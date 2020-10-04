import time
import random
import tensorflow as tf
import numpy as np
from copy import deepcopy
from scipy import stats
from tensorflow.python.ops.gen_array_ops import quantize_and_dequantize

from gat.environment.env import Env
from gat.model.encoder import Encoder
from gat.model.decoder import Decoder
from mcts.mcts import MCTS
from server.server import Server


class Actor():
    def __init__(
        self,
        encoder,
        decoder,
        encoder_target,
        decoder_target,
        server,
        gamma=0.999,
        synchronize_freq=10,
        upload_freq=50
    ):
        pass

    def start(self):

        STEP_NUM = 10

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
        GUMMA = 0.99999
        SEARCH_NUM = 10
        UPLOAD_INTERVAL = 1000
        DOWNLOAD_INTERVAL = 10

        # 環境用パラメータ
        SIZE_MIN = 14
        SIZE_MAX = 15

        def eps_greedy(Q, eps, env):
            rand = np.random.rand()
            if eps > rand:
                return np.argmax(np.random.rand(len(Q)) + env.state.trajectory.mask())
            else:
                return np.argmax(Q + env.state.trajectory.mask())

        online_env = Env()

        encoder = Encoder(D_MODEL, D_KEY, N_HEAD, DEPTH, WEIGHT_BALANCER)
        decoder = Decoder(D_MODEL, D_KEY, N_HEAD, TH_RANGE, WEIGHT_BALANCER)

        # データ初期化
        eps = 0.1  # TODO epsはAnnealする
        return_list = []

        upload_step = 0
        download_step = 0

        # この問題のゲームを解く(下を繰り返し)
        mcts = MCTS(online_env, encoder, decoder, GUMMA)

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
            size=200000, env_dict=env_dict)
        server.start()

        while True:
            # 今回の問題を決定する
            graph_size = random.randrange(SIZE_MIN, SIZE_MAX)
            online_env.reset(graph_size)

            if download_step == DOWNLOAD_INTERVAL:
                download_step = 0
                encoder_weight, decoder_weight = server.download()
                encoder.set_weights(encoder_weight)
                decoder.set_weights(decoder_weight)

            for _ in range(graph_size):

                if upload_step == UPLOAD_INTERVAL:
                    upload_step = 0
                    server.add(return_list)
                    return_list = []

                # MCTSによって次の選択肢のQベクトルを受け取る
                search_env = deepcopy(online_env)
                Q = mcts.search(search_env, SEARCH_NUM)

                # ε-greedyによって次の行動を決める
                next_action = eps_greedy(Q, eps, online_env)

                # Envから次の行動を選択したことによって次のS,r,doneを受け取る
                before_trajectory = deepcopy(online_env.state.trajectory)
                next_state, reward, done = online_env.step(next_action)

                # listに追加する
                return_list.append((online_env.state.graph, before_trajectory,
                                    next_action, online_env.state.graph, online_env.state.trajectory, reward, Q))
                upload_step += 1

            download_step += 1
