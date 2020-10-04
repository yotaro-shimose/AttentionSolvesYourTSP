import tensorflow as tf


class Learner:
    def __init__(
        self,
        encoder,
        decoder,
        encoder_target,
        decoder_target,
        server,
        batch_size=512,
        gamma=0.999,
        learning_rate=1e-3,
        synchronize_freq=10,
        upload_freq=50,
        beta_q_first=1,
        beta_q_last=0.1,
        beta_a_first=0,
        beta_a_last=0.045,
        annealing_step=int(1e5)
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_target = encoder_target
        self.decoder_target = decoder_target
        self.server = server
        self.batch_size = batch_size
        self.gamma = gamma
        learning_rate = learning_rate
        self.synchronize_freq = synchronize_freq
        self.upload_freq = upload_freq
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # annealing configuration
        self.beta_q_first = beta_q_first
        self.beta_q_last = beta_q_last
        self.beta_a_first = beta_a_first
        self.beta_a_last = beta_a_last
        self.annealing_step = annealing_step

        self.beta_q = self.beta_q_first
        self.beta_a = self.beta_a_first

    def start(self):
        self.step = 0
        while True:
            self.step += 1
            metrics = self.train()
            self.anneal()
            # lossのログ
            # TODO implement tensorboard
            if self.step % 100:
                self.log_metrics(metrics)
            if self.step % self.synchronize_freq == 0:
                self.synchronize()
            if self.step % self.upload_freq == 0:
                self.upload()

    def train(self):

        # Serverからサンプル
        sample = self.server.sample(self.batch_size)
        graph = sample["graph"]
        traj = sample["traj"]
        action = sample["act"]
        reward = sample["rew"]
        next_graph = sample["next_graph"]
        next_traj = sample["next_traj"]
        done = sample["done"]
        Q = sample["Q"]

        # Gradient Step
        loss = self.train_on_batch(
            graph,
            traj,
            action,
            reward,
            next_graph,
            next_traj,
            done,
            Q
        )

        metrics = {"step": self.step, "loss": loss}

        return metrics

    # @tf.function
    def train_on_batch(
        self,
        graph,
        trajectory,
        action,
        reward,
        next_graph,
        next_trajectory,
        done,
        Q_mcts
    ):
        # Qターゲットを計算
        next_action = tf.argmax(self.network(
            [next_graph, next_trajectory]), axis=1)
        indice = tf.stack(
            [tf.range(next_action.shape[0]), next_action], axis=1)
        next_Q = tf.reshape(tf.gather_nd(self.network_target(
            [next_graph, next_trajectory])), (-1, 1))
        target = reward + self.gamma * next_Q * \
            ((tf.cast(done, tf.int32) - 1) * (-1))

        # TD Loss と Amortized Loss を計算
        indice = tf.stack([tf.range(action.shape[0]), action], axis=1)
        mcts_policy = tf.nn.softmax(Q_mcts)
        with tf.GradientTape() as tape:
            Q = tf.gather_nd(self.network([graph, trajectory]), indice)
            td_loss = tf.keras.losses.mse(Q, target)

            amortized_loss = tf.nn.softmax_cross_entropy_with_logits(
                mcts_policy, Q)
            total_loss = self.beta_q * td_loss + self.beta_a * amortized_loss
            gradient = tape.gradient(total_loss, self.trainable_variables())
            self.optimizer.apply_gradients(
                zip(gradient, self.trainable_variables()))

        # Return Loss
        return total_loss

    def network(self, inputs):
        graph = inputs[0]
        trajectory = inputs[1]
        H = self.encoder(graph)
        return self.decoder([H, trajectory])

    def network_target(self, inputs):
        graph = inputs[0]
        trajectory = inputs[1]
        H = self.encoder_target(graph)
        return self.decoder_target([H, trajectory])

    def trainable_variables(self):
        return self.encoder.trainable_variables + self.decoder.trainable_variables

    def anneal(self):
        q_step = (self.beta_q_last - self.beta_q_first) / self.annealing_step

        new_beta_q = self.beta_q + q_step
        a_step = (self.beta_a_last - self.beta_a_first) / self.annealing_step
        new_beta_a = self.beta_a + a_step

        if new_beta_q > self.beta_q_last:
            self.beta_q = new_beta_q
        if new_beta_a < self.beta_a_last:
            self.beta_a = new_beta_a

    def synchronize(self):
        self.encoder_target.set_weights(self.encoder.get_weights())
        self.decoder_target.set_weights(self.decoder.get_weights())

    def upload(self):
        self.server.upload((self.encoder.get_weights(),
                            self.decoder.get_weights()))

    # TODO implement tensorboard
    def log_metrics(self, metrics):
        print(f"train_step: {metrics['step']}  loss: {metrics['loss']}")
