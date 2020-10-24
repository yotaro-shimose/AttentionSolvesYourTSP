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
        n_nodes_min,
        n_nodes_max,
        d_model,
        d_key,
        n_heads,
        depth,
        weight_balancer,

    ):
        pass

    def start(self):
        for epoch in self.n_epochs:
            for iteration in self.n_iterations:
                metrics = self.train_on_episode()
                self.logger.log(metrics)
            if self.validate():
                self.synchronize(self.encoder, self.decoder,
                                 self.base_encoder, self.base_decoder)

    def train_on_episode(self):
        """train_on_episode executes parallel episodes at the same time and learn from the experiences.
        """
        # ** Initialization ** #
        # Initilalize state
        self.graph_size = np.random.randint(self.n_nodes_min, self.n_nodes_max)
        # Initialize state
        map(lambda x: x.reset(self.graph_size), self.online_envs)
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
            trajectories = [env.state.trajectory for env in self.online_envs]
            likelihood = self.calculate_likelihood(
                online_policies, trajectories)

            trainable_variables = self.encoder.trainable_variables + \
                self.decoder.trainable_variables
            # Get policy gradient to apply to our network
            policy_gradient = tape.gradient((online_rewards - base_rewards) *
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
            rewards = [env.step(action)[1] for env, action in zip(
                envs, actions)]
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
        online, base = [], []
        for _ in range(self.n_validation):
            graph_size = np.random.randint(self.n_nodes_min, self.n_nodes_max)
            # ** Initialization ** #

            # Initialize state
            map(lambda x: x.reset(graph_size), self.online_envs)
            # Copy env list for baseline.
            base_envs = [deepcopy(env) for env in self.online_envs]
            online_rewards, _ = self.play_game(
                self.online_envs, self.encoder, self.decoder, greedy=True)
            base_rewards, _ = self.play_game(
                base_envs, self.base_encoder, self.base_decoder, greedy=True)
            online.append(tf.reduce_mean(online_rewards).numpy())
            base.append(tf.reduce_mean(base_rewards).numpy())
        online = np.concat(online)
        base = np.concat(base)
        return stats.ttest_rel(online, base).pvalue < self.significance \
            and np.mean(online) > np.mean(base)

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError
