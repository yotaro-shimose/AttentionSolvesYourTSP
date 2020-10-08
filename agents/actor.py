import random
import numpy as np
from copy import deepcopy
from mcts.mcts import MCTS


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
        size_min=14,
        size_max=15,
        search_num=10
    ):
        self.env = env_builder()
        self.server = server
        self.logger_builder = logger_builder
        self.gamma = gamma
        self.upload_interval = upload_interval
        self.download_interval = download_interval
        self.size_min = size_min
        self.size_max = size_max

        self.search_num = search_num

        self.encoder_builder = encoder_builder
        self.decoder_builder = decoder_builder

    def start(self):

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
        eps = 0.1  # TODO epsはAnnealする
        return_list = []

        upload_step = 0
        download_step = 0
        print('start')

        # ログ用のエピソードカウンタ
        episode_count = 0

        # この問題のゲームを解く(下を繰り返し)
        while True:
            episode_count += 1
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

            episode_reward = 0

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
                after_trajectory = deepcopy(next_state.trajectory)

                # rewardを記録する
                episode_reward += reward

                # listに追加する
                return_list.append((self.env.state.graph, before_trajectory, next_action,
                                    reward, next_state.graph, after_trajectory, done, Q))
                upload_step += 1
            # Logを出力する
            if self.logger:
                metrics = {
                    "Episode Reward": episode_reward
                }
                self.logger.log(metrics, episode_count)

            download_step += 1
