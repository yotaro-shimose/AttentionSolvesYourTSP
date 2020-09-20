import tensorflow as tf
import numpy as np
from math import sqrt

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

        state = env.state

        # 経験判定
        if state not in self.visited:
            # Qテーブルの初期化
            self.Q[state.trajectory] = np.zeros(len(state.trajectory))

            # Nの初期化
            self.N[state.trajectory] = np.zeros(len(state.trajectory))

            self.visited.append(state)
            v, self.P[state.trajectory] = self.decoder([self.input, tf.constant(
                [env.state.trajectory], dtype=tf.int32)])
            return v

        # 次の選択を判断
        p_array = self.P[state].numpy()
        random = np.random.normal(
            AVERAGE_CONST, STANDARD_DEVIDATION, len(p_array))
        u_array = self.Q[state.trajectory] + C_PUCT * p_array * \
            sqrt(sum(self.N[state]))/(1+self.N[state])
        next_action = np.argmax((u_array+random)*env.mask())

        next_state, totalcost, done = env.step(next_action)

        # 終了判定
        if done:
            return totalcost

        v = self.search(next_state, env)

        self.Q[state.trajectory][next_action] = (
            self.N[state.trajectory][next_action] * self.Q[state.trajectory][next_action] + v) / (self.N[state.trajectory][next_action] + 1)
        self.N[state.trajectory][next_action] += 1
        return v

    def pi(self, state):
        trajectory = state.trajectory
        return self.N[trajectory]/sum(self.N[trajectory])
