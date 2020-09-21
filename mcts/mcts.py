import tensorflow as tf
import numpy as np
from math import sqrt
from copy import deepcopy

import time
AVERAGE_CONST = 0
STANDARD_DEVIDATION = 0.000001
C_PUCT = 0.001


class MCTS:

    def __init__(self, env, encoder, decoder):
        self.env = env
        self.decoder = decoder
        self.input = encoder(tf.constant([env.state.graph], dtype=tf.float32))
        self.Q = {}
        self.N = {}
        self.P = {}
        self.visited = []

    def search(self, env):

        trajectory = deepcopy(env.state.trajectory)

        start = time.perf_counter()

        # 経験判定
        if trajectory not in self.visited:

            # Qテーブルの初期化
            self.Q[trajectory] = np.zeros(len(trajectory))

            # Nの初期化
            self.N[trajectory] = np.zeros(len(trajectory))

            self.visited.append(trajectory)

            v, self.P[trajectory] = self.decoder([self.input, tf.constant(
                [trajectory], dtype=tf.int32)])
            end = time.perf_counter()
            print(f"消費時間：{end-start}")

            return v

        # 次の選択を判断
        p_array = np.squeeze(self.P[trajectory].numpy())
        random = np.random.normal(
            AVERAGE_CONST, STANDARD_DEVIDATION, len(p_array))
        u_array = self.Q[trajectory] + C_PUCT * p_array * \
            sqrt(sum(self.N[trajectory]))/(1+self.N[trajectory])
        next_action = np.argmax((u_array+random)+trajectory.mask())

        next_state, totalcost, done = env.step(next_action)

        # 終了判定
        if done:
            self.Q[trajectory][next_action] = (
                self.N[trajectory][next_action] * self.Q[trajectory][next_action] + totalcost) / (self.N[trajectory][next_action] + 1)
            self.N[trajectory][next_action] += 1
            return totalcost

        v = self.search(env)

        self.Q[trajectory][next_action] = (
            self.N[trajectory][next_action] * self.Q[trajectory][next_action] + v) / (self.N[trajectory][next_action] + 1)
        self.N[trajectory][next_action] += 1

        return v

    def pi(self, state):
        trajectory = state.trajectory
        return self.N[trajectory]/sum(self.N[trajectory])
