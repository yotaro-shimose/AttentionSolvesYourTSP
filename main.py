import random
import tensorflow as tf
import numpy as np
from copy import deepcopy
from scipy import stats

from gat.environment.env import Env
from gat.model.encoder import Encoder
from gat.model.decoder import Decoder
from mcts.mcts import Mcts
import pathlib
import time

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

# 環境用パラメータ
SIZE_MIN = 30
SIZE_MAX = 31

# セーブディレクトリ
# SAVE_DIRECTORY_ENCODER = str(
#     pathlib.Path.cwd().joinpath("weights").joinpath("encoder"))
# SAVE_DIRECTORY_DECODER = str(
#     pathlib.Path.cwd().joinpath("weights").joinpath("decoder"))

# LOAD_DIRECTORY_ENCODER = str(
#     pathlib.Path.cwd().joinpath("weights").joinpath("encoder"))
# LOAD_DIRECTORY_DECODER = str(
#     pathlib.Path.cwd().joinpath("weights").joinpath("decoder"))


def synchronize(encoder, decoder, base_encoder, base_decoder):
    base_encoder.set_weights(encoder.get_weights())
    base_decoder.set_weights(decoder.get_weights())


def play_game(encoder, decoder, graph_list, graph_size, env_list, action_list, policy_list,
              isRandomChoice):
    output = encoder(tf.constant(graph_list, dtype=tf.float32))
    for _ in range(graph_size):
        v, p = decoder([output, tf.constant(
            [env.state.trajectory for env in env_list], dtype=tf.int32)])
        policy = tf.squeeze(p, axis=1)
        if isRandomChoice:
            policy_list.append(policy)
        next_action_probability_list = policy.numpy()

        for next_action_probability, env in zip(next_action_probability_list, env_list):
            if isRandomChoice:
                next_action = np.random.choice(
                    a=action_list, p=next_action_probability)
            else:
                next_action = action_list[np.argmax(next_action_probability)]
            env.step(next_action)


def executeEpisode(env, encoder, decoder):
    examples = []
    state = env.state
    mcts = Mcts(env, encoder, decoder)

    while True:
        for _ in range(NUM_MCTS_SIMS):
            mcts.search(env)
        examples.append([state, mcts.pi(state), None])
        next_action = random.choice(len(mcts.pi(state)), p=mcts.pi(state))
        state, totalcost, done = env.step(state, next_action)
        if done:
            return [[exam[0], exam[1], totalcost] for exam in examples]


if __name__ == "__main__":
    # 学習用
    encoder = Encoder(D_MODEL, D_KEY, N_HEAD, DEPTH, WEIGHT_BALANCER)
    decoder = Decoder(D_MODEL, D_KEY, N_HEAD, TH_RANGE, WEIGHT_BALANCER)
    optimizer = tf.optimizers.Adam(lr=LEARNING_RATE)

    # 最強格納用
    base_encoder = Encoder(D_MODEL, D_KEY, N_HEAD, DEPTH)
    base_decoder = Decoder(D_MODEL, D_KEY, N_HEAD, TH_RANGE)

    # if LOAD_DIRECTORY_ENCODER is not None and LOAD_DIRECTORY_DECODER is not None:
    #     encoder.load_weights(LOAD_DIRECTORY_ENCODER)
    #     decoder.load_weights(LOAD_DIRECTORY_DECODER)
    #     synchronize(encoder, decoder, base_encoder, base_decoder)

    for e in range(EPOCH_NUM):
        for t in range(STEP_NUM):
            # 開始時刻
            start = time.time()

            # 今回の問題を決定する
            graph_size = random.randrange(SIZE_MIN, SIZE_MAX)
            online_env = [Env(graph_size)]
            base_env = deepcopy(online_env)
            graph_list = [
                online_env.state.graph]

            action_list = [index for index in range(graph_size)]
            policy_list = []

            examples = []
            with tf.GradientTape() as tape:

                # 現在のNNで問題を解く
                examples += executeEpisode(online_env, encoder, decoder)

                # trajectoryをたどって、尤度確率を計算
                online_env_list_state_tragectory = []

                #  trajectoryデータにバッチナンバー、行動番号を振る
                trajectory = online_env.state.trajectory
                index_and_tragectory = [
                    [0, index, action] for index, action in enumerate(trajectory)]
                online_env_list_state_tragectory.append(
                    index_and_tragectory)

                select_probability = tf.gather_nd(
                    tf.stack(policy_list, axis=1), online_env_list_state_tragectory)

                prob = tf.reduce_mean(tf.multiply(tf.cast(online_env.state.total_cost(), tf.float32), tf.math.log(
                    tf.reduce_prod(select_probability, 1))))

                # 微分
                gradient = tape.gradient(
                    prob, encoder.trainable_variables + decoder.trainable_variables)

                # コスト関数を計算する
                online_cost_list = [env.state.total_cost()
                                    for env in online_env_list]

                # 現最強のNNで問題を解く
                play_game(base_encoder, base_decoder, graph_list,
                          graph_size, base_env_list, action_list, policy_list, False)

                # コスト関数を計算する
                base_cost_list = [env.state.total_cost()
                                  for env in base_env_list]

                # コストの差を計算する
                cost = tf.constant([
                    x - y for (x, y) in zip(online_cost_list, base_cost_list)])

                # trajectoryをたどって、尤度確率を計算
                online_env_list_state_tragectory = []

                #  trajectoryデータにバッチナンバー、行動番号を振る
                for batch_index, env in enumerate(online_env_list):
                    trajectory = env.state.trajectory
                    index_and_tragectory = [
                        [batch_index, index, action] for index, action in enumerate(trajectory)]
                    online_env_list_state_tragectory.append(
                        index_and_tragectory)

                # 選択した確率ベクトル
                # axis=1で行けてる？
                select_probability = tf.gather_nd(
                    tf.stack(policy_list, axis=1), online_env_list_state_tragectory)
                prob = tf.reduce_mean(tf.multiply(tf.cast(cost, tf.float32), tf.math.log(
                    tf.reduce_prod(select_probability, 1))))

                # 微分
                gradient = tape.gradient(
                    prob, encoder.trainable_variables + decoder.trainable_variables)

                # NNを更新する
                optimizer.apply_gradients(
                    zip(gradient, encoder.trainable_variables + decoder.trainable_variables))
            # 終了時刻
            end = time.time()
            print(f'エポック：{e} エピソード：{t} コスト平均：{sum(online_cost_list)/len(online_cost_list)} \
                ゲームサイズ：{graph_size} 時間消費：{end-start}')

        # STEP_NUM回の学習を終了させたNNと現最強のNNより強いかT検定により判定
        # 強ければ最強を更新
        online_env_list = [Env(graph_size) for _ in range(EVALUATION_NUM)]
        base_env_list = deepcopy(online_env_list)
        graph_list = [
            env.state.graph for env in online_env_list]
        play_game(encoder, decoder, graph_list,
                  graph_size, online_env_list, action_list, [], True)

        play_game(base_encoder, base_decoder, graph_list,
                  graph_size, base_env_list, action_list, [], True)

        online_cost_list = [env.state.total_cost()
                            for env in online_env_list]
        base_cost_list = [env.state.total_cost()
                          for env in base_env_list]

        if stats.ttest_rel(np.array(online_cost_list), np.array(base_cost_list)).pvalue\
                < SIGNIFICANCE and np.mean(online_cost_list) < np.mean(base_cost_list):
            print(f"t検定通過 with 優位水準: {SIGNIFICANCE}")
            # encoder.save_weights(SAVE_DIRECTORY_ENCODER)
            # decoder.save_weights(SAVE_DIRECTORY_DECODER)
            synchronize(encoder, decoder, base_encoder, base_decoder)
