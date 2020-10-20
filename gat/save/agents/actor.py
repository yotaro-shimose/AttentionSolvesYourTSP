import numpy as np
from copy import deepcopy

from gat.save.mcts.mcts import MCTS
import os
import time


class Actor():
    def __init__(
        self,
        env_builder,
        encoder_builder,
        decoder_builder,
        server,
        logger_builder=None,
        gamma=0.999,
        upload_interval=1000,
        download_interval=10,
        n_nodes=14,
        search_num=10,
        eps_init=1.0,
        eps_end=0.01,
        annealing_step=int(1e4),
    ):
        self.env = env_builder()
        self.server = server
        self.logger_builder = logger_builder
        self.gamma = gamma
        self.upload_interval = upload_interval
        self.download_interval = download_interval
        self.n_nodes = n_nodes
        self.search_num = search_num

        self.eps_init = eps_init
        self.eps_end = eps_end
        self.annealing_step = annealing_step
        self.encoder_builder = encoder_builder
        self.decoder_builder = decoder_builder

    def start(self):
        pid = os.getpid()
        np.random.seed(pid + int(time.time()))

        self.encoder = self.encoder_builder()
        self.decoder = self.decoder_builder()
        self.logger = self.logger_builder() if self.logger_builder else None

        def eps_greedy(Q, eps, env):
            rand = np.random.rand()
            if eps > rand:
                return np.argmax(np.random.rand(len(Q)) + env.state.trajectory.mask())
            else:
                return np.argmax(Q + env.state.trajectory.mask())

        # データ初期化
        self.eps = self.eps_init
        return_list = []

        print('start')

        # ステップカウンタ
        self.step_count = 0

        # ログ用のエピソードカウンタ
        self.episode_count = 0

        # この問題のゲームを解く(下を繰り返し)
        while True:
            self.episode_count += 1
            # 今回の問題を決定する
            self.env.reset(self.n_nodes)

            mcts = MCTS(self.env, self.encoder, self.decoder, self.gamma)

            if self.episode_count % self.download_interval == 0:
                weight = self.server.download()
                if weight:
                    encoder_weight, decoder_weight = weight
                    self.encoder.set_weights(encoder_weight)
                    self.decoder.set_weights(decoder_weight)

            episode_reward = 0
            done = False

            while not done:
                self.step_count += 1

                if self.step_count % self.upload_interval == 0:
                    self.server.add(return_list)
                    return_list = []

                # MCTSによって次の選択肢のQベクトルを受け取る
                search_env = deepcopy(self.env)
                Q = mcts.search(search_env, self.search_num)

                # ε-greedyによって次の行動を決める
                next_action = eps_greedy(Q, self.eps, self.env)

                # Envから次の行動を選択したことによって次のS,r,doneを受け取る
                before_trajectory = deepcopy(self.env.state.trajectory)
                next_state, reward, done = self.env.step(next_action)
                after_trajectory = deepcopy(next_state.trajectory)

                # rewardを記録する
                episode_reward += reward

                # listに追加する
                return_list.append((self.env.state.graph, before_trajectory, next_action,
                                    reward, next_state.graph, after_trajectory, done, Q))

            # Logを出力する
            if self.logger:
                metrics = {
                    "Episode Reward": episode_reward
                }
                self.logger.log(metrics, self.episode_count)

            # epsをAnneal
            self.anneal()

    def anneal(self):
        step = (self.eps_end - self.eps_init) / self.annealing_step
        new_eps = self.eps + step
        if new_eps > self.eps_end:
            self.eps = new_eps
