import tensorflow as tf
import numpy as np
from copy import deepcopy
from scipy import stats

from gat.environment.env import Env
from gat.modules.models.encoder import Encoder
from gat.modules.models.decoder import PolicyDecoder
import time


class Reinforce:
    def __init__(
        self,
        n_epochs=10000,
        n_iterations=10,
        n_validations=100,
        n_parallels=5,
        n_nodes_min=14,
        n_nodes_max=14,
        learning_rate=1e-3,
        d_model=128,
        d_key=16,
        n_heads=8,
        depth=3,
        th_range=10,
        weight_balancer=1,
        significance=0.05,
        logger=None,
    ):
        self.n_epochs = n_epochs
        self.n_iterations = n_iterations
        self.n_validations = n_validations
        self.online_envs = [Env() for _ in range(n_parallels)]
        self.n_nodes_min = n_nodes_min
        self.n_nodes_max = n_nodes_max
        self.encoder = Encoder(
            d_model=d_model,
            d_key=d_key,
            n_heads=n_heads,
            depth=depth,
            weight_balancer=weight_balancer
        )
        self.decoder = PolicyDecoder(
            d_model,
            d_key,
            n_heads,
            th_range,
            weight_balancer
        )
        self.base_encoder = Encoder(
            d_model=d_model,
            d_key=d_key,
            n_heads=n_heads,
            depth=depth,
            weight_balancer=weight_balancer
        )
        self.base_decoder = PolicyDecoder(
            d_model,
            d_key,
            n_heads,
            th_range,
            weight_balancer
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.significance = significance
        self.logger = logger

    def start(self):
        for epoch in range(self.n_epochs):
            for iteration in range(self.n_iterations):
                start = time.time()
                metrics = self.train_on_episode()
                end = time.time()
                print(f"経過時間: {end - start}(s)")
                step = epoch * self.n_iterations + iteration
                self.logger.log(metrics, step)
            if self.validate():
                print("Validation passed")
                self.synchronize(self.encoder, self.decoder,
                                 self.base_encoder, self.base_decoder)

    def train_on_episode(self):
        """train_on_episode executes parallel episodes at the same time and learn from the experiences.
        """
        # ** Initialization ** #
        # Initilalize state
        self.graph_size = np.random.randint(
            self.n_nodes_min, self.n_nodes_max + 1)
        # Initialize state
        [env.reset(self.graph_size) for env in self.online_envs]
        # Copy env list for baseline.
        base_envs = [deepcopy(env) for env in self.online_envs]

        with tf.GradientTape() as tape:
            # Execute an episode for each online environment
            online_rewards, online_policies = self.play_game(
                envs=self.online_envs,
                encoder=self.encoder,
                decoder=self.decoder,
                greedy=False
            )
            # Greedy rollout
            base_rewards, _ = self.play_game(
                envs=base_envs,
                encoder=self.base_encoder,
                decoder=self.base_decoder,
                greedy=True
            )
            # ** Learn from experience ** #
            # Get likelihood
            trajectories = tf.stack(
                [env.state.trajectory for env in self.online_envs], axis=0)
            likelihood = self.calculate_likelihood(
                online_policies, trajectories)

            trainable_variables = self.encoder.trainable_variables + \
                self.decoder.trainable_variables
            # Get policy gradient to apply to our network
            policy_gradient = tape.gradient((base_rewards - online_rewards) *
                                            tf.math.log(likelihood), trainable_variables)

            # Apply gradient
            self.optimizer.apply_gradients(
                zip(policy_gradient, trainable_variables))
        # metrics
        metrics = {
            "average_reward": tf.reduce_mean(online_rewards)
        }
        return metrics

    def play_game(self, envs, encoder, decoder, greedy=False):
        """play games in parallels

        Args:
            envs ([type]): list of environments which are RESET.
            encoder ([type]): [description]
            decoder ([type]): [description]
            greedy (bool, optional): [description]. Defaults to False.

        Returns:
            tuple(tf.Tensor(batch_size), tf.Tensor(batch_size, graph_size, graph_size)):
                rewards, policies
        """

        # ** Initialization ** #

        # List to store policies used in this episode.
        policy_series = []
        # List to store rewards in this episode for both online agent and baseline
        reward_series = []
        # Get graph
        graphs = [env.state.graph for env in envs]
        graphs = np.stack(graphs, axis=0)
        Hs = encoder(graphs)
        for _ in range(self.graph_size):
            # Get trajectories
            trajectories = np.stack(
                [env.state.trajectory for env in envs], axis=0)

            # Get policy
            policies = decoder(
                [Hs, trajectories])
            policy_series.append(policies)

            # Define action selection strategy (sampling or greedy)
            get_action = np.argmax if greedy else self.sample_action
            # Get actions for online agent
            actions = [get_action(
                policy) for policy in policies.numpy()]

            # Take actions (Note env.step returns state, reward and done, so we index 1)
            rewards = tf.constant([env.step(action)[1] for env, action in zip(
                envs, actions)], dtype=tf.float32)
            # store rewards
            reward_series.append(rewards)
        return tf.reduce_sum(tf.stack(reward_series, 1), axis=1), tf.stack(policy_series, 1)

    def sample_action(self, policy):
        """sample_action samples an action following the action distribution.
        Policy is a single policy, but batch of policies.

        Args:
            policy (np.ndarray): Rank one ndarray action distribution.
        """
        size = 1
        return np.random.choice(range(len(policy)), size, p=policy)

    def calculate_likelihood(self, policy_series, trajectories):
        """calculate_likelihood calculates likelihood to have this trajectory

        Args:
            policy_series (tf.Tensor(batch_size, graph_size, graph_size)):
            trajectories (tf.Tensor(batch_size, graph_size)):
        Return:
            likelihood (tf.Tensor): likelihood to have this trajectory with shape (batch_size)
        """
        b_indice = tf.tile(tf.expand_dims(
            tf.range(trajectories.shape[0]), 1), (1, trajectories.shape[1]))
        t_indice = tf.tile(tf.expand_dims(
            tf.range(trajectories.shape[1]), 0), (trajectories.shape[0], 1))
        indice = tf.stack([b_indice, t_indice, trajectories], axis=2)
        # Get policy for each action with shape (batch_size, graph_size)
        gather = tf.gather_nd(policy_series, indice)

        return tf.reduce_prod(gather, axis=1)

    def synchronize(self, encoder, decoder, base_encoder, base_decoder):
        base_encoder.set_weights(encoder.get_weights())
        base_decoder.set_weights(decoder.get_weights())

    def validate(self):

        online_envs = [Env() for _ in range(self.n_validations)]
        graph_size = (self.n_nodes_min + self.n_nodes_max) // 2
        # ** Initialization ** #

        # Initialize state
        [env.reset(graph_size) for env in online_envs]
        # Copy env list for baseline.
        base_envs = deepcopy(online_envs)
        online_rewards, _ = self.play_game(
            online_envs, self.encoder, self.decoder, greedy=True)
        base_rewards, _ = self.play_game(
            base_envs, self.base_encoder, self.base_decoder, greedy=True)

        online_rewards = online_rewards.numpy()
        base_rewards = base_rewards.numpy()

        return stats.ttest_rel(online_rewards, base_rewards).pvalue < self.significance and \
            np.mean(online_rewards) > np.mean(base_rewards)

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
