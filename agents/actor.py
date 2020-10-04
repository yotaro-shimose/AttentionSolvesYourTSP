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
        env_builder,
        encoder_builder,
        decoder_builder,
        server,
        gamma=0.999,
        step_num=10,
        upload_interval=1000,
        download_interval=10,
        size_min=14,
        size_max=15,
        d_model=128,
        d_key=16,
        n_head=8,
        depth=2,
        learning_rate=9.0e-5,
        weight_balancer=0.12,
        search_num=10
    ):
        self.env = env_builder()
        self.server = server
        self.gamma = gamma
        self.step_num = step_num
        self.upload_interval = upload_interval
        self.download_interval = download_interval
        self.size_min = size_min
        self.size_max = size_max
        self.d_model = d_model
        self.d_key = d_key
        self.n_head = n_head
        self.depth = depth
        self.learning_rate = learning_rate
        self.weight_balancer = weight_balancer
        self.search_num = search_num

        self.encoder = encoder_builder(self.d_model, self.d_key,
                                       self.n_head, self.depth, self.weight_balancer)
        self.decoder = decoder_builder(self.d_model, self.d_key, self.n_head,
                                       self.weight_balancer)

    def start(self):

        def eps_greedy(Q, eps, env):
            rand = np.random.rand()
            if eps > rand:
                return np.argmax(np.random.rand(len(Q)) + env.state.trajectory.mask())
            else:
                return np.argmax(Q + env.state.trajectory.mask())

        # データ初期化
        eps = 0.1  # TODO epsはAnnealする
        return_list = []

        upload_step = 0
        download_step = 0
        print('start')
        # この問題のゲームを解く(下を繰り返し)
        while True:
            # 今回の問題を決定する
            graph_size = random.randrange(self.size_min, self.size_max)
            self.env.reset(graph_size)

            mcts = MCTS(self.env, self.encoder, self.decoder, self.gamma)

            if download_step == self.download_interval:
                download_step = 0
                weight = self.server.download()
                if weight:
                    encoder_weight, decoder_weight = weight
                    self.encoder.set_weights(encoder_weight)
                    self.decoder.set_weights(decoder_weight)

            for _ in range(graph_size):

                if upload_step == self.upload_interval:
                    upload_step = 0
                    self.server.add(return_list)
                    return_list = []

                # MCTSによって次の選択肢のQベクトルを受け取る
                search_env = deepcopy(self.env)
                Q = mcts.search(search_env, self.search_num)

                # ε-greedyによって次の行動を決める
                next_action = eps_greedy(Q, eps, self.env)

                # Envから次の行動を選択したことによって次のS,r,doneを受け取る
                before_trajectory = deepcopy(self.env.state.trajectory)
                next_state, reward, done = self.env.step(next_action)

                # listに追加する
                return_list.append((self.env.state.graph, before_trajectory,
                                    next_action, self.env.state.graph, self.env.state.trajectory, reward, Q))
                print(len(return_list))
                upload_step += 1

            download_step += 1
