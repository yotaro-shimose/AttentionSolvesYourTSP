import tensorflow as tf
import numpy as np
from math import sqrt
from copy import deepcopy

import time
AVERAGE_CONST = 0
STANDARD_DEVIDATION = 0.000001
C_PUCT = 0.001


class MCTS:

    def __init__(self, env, encoder, decoder, gumma):
        self.env = env
        self.decoder = decoder
        self.input = encoder(tf.constant([env.state.graph], dtype=tf.float32))
        self.gumma = gumma
        self.Q = {}
        self.N = {}
        self.P = {}
        self.visited = []

    def search(self, env, search_num):

        trajectory = deepcopy(env.state.trajectory)
        if trajectory not in self.visited:
            # Nの初期化
            self.N[trajectory] = np.ones(len(trajectory))
            self.Q[trajectory] = self.decoder([self.input, tf.constant(
                [trajectory], dtype=tf.int32)]).numpy().squeeze()
            self.visited.append(trajectory)
        # 決められた回数ゲームを行う
        for k in range(search_num):
            self.play_game(env)

        return self.Q[trajectory]

    def play_game(self, env):
        '''
            現在の状態から再帰的にアクションを選択し、一回ゲームを終了させる
        '''

        trajectory = deepcopy(env.state.trajectory)

        # 初めてきたSの場合、QをNNで初期化
        if trajectory not in self.visited:
            # Nの初期化
            self.N[trajectory] = np.ones(len(trajectory))
            self.Q[trajectory] = self.decoder([self.input, tf.constant(
                [trajectory], dtype=tf.int32)]).numpy().squeeze()
            self.visited.append(trajectory)
            return np.max(self.Q[trajectory])

        # 次の選択を決める
        u_array = C_PUCT * \
            np.sqrt(np.log(sum(self.N[trajectory]))/(self.N[trajectory]))
        next_action = np.argmax(
            self.Q[trajectory] + u_array + trajectory.mask())

        # 次の行動をとる
        next_state, reward, done = env.step(next_action)

        # その行動の回数を増加
        self.N[trajectory][next_action] += 1

        # Q関数の更新
        # Tまでゲームを行い、rewardを計算
        if done:
            total_reward = reward
        else:
            total_reward = self.gumma * self.play_game(env)
        self.Q[trajectory][next_action] = (
            (self.N[trajectory][next_action] - 1) * self.Q[trajectory][next_action] + total_reward)/self.N[trajectory][next_action]

        if done:
            return reward

        return reward + total_reward
