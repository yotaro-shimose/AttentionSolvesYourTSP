import tensorflow as tf
import time


# value to mask before softmax operation.


def masked_cross_entropy_from_Q(Q, Q_target, mask):
    p_target = masked_softmax(Q_target, mask)
    p = masked_softmax(Q, mask)
    return -tf.math.reduce_mean(tf.keras.layers.dot([p_target, masked_log(p, mask)], axes=1))


def masked_softmax(tensor, mask):
    tensor = tensor
    exps = tf.math.exp(tensor) * (1 - tf.cast(mask, tf.float32))
    softmax = exps / tf.math.reduce_sum(exps, 1, keepdims=True)
    return softmax


def masked_log(tensor, mask):
    float_mask = tf.cast(mask, tf.float32)
    log = tf.math.log((1 - float_mask) * tensor + float_mask * 1)
    return log


# compute mask for trajectory with shape(batch_size, node_size)
def create_mask(trajectory):
    def _create_mask(trajectory):
        tf_range = tf.range(tf.size(trajectory))
        return tf.map_fn(lambda x: tf.size(tf.where(trajectory == x))
                         != 0, tf_range, fn_output_signature=tf.bool)
    return tf.map_fn(_create_mask, trajectory, fn_output_signature=tf.bool)


def masked_argmax(tensor, mask):
    min = tf.math.reduce_min(tensor)
    return tf.argmax(tf.where(mask, min, tensor), axis=1, output_type=tf.int32)


class Learner:
    def __init__(
        self,
        encoder_builder,
        decoder_builder,
        server,
        logger_builder=None,
        batch_size=512,
        gamma=0.999,
        learning_rate=1e-3,
        synchronize_interval=10,
        upload_interval=50,
        beta_q_first=1,
        beta_q_last=0.1,
        beta_a_first=0,
        beta_a_last=0.10,
        annealing_step=int(1e4),
        weight_balancer=0.12,
    ):
        self.encoder_builder = encoder_builder
        self.decoder_builder = decoder_builder
        self.server = server
        self.logger_builder = logger_builder
        self.batch_size = batch_size
        self.gamma = gamma
        learning_rate = learning_rate
        self.synchronize_interval = synchronize_interval
        self.upload_interval = upload_interval

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.huber = tf.keras.losses.Huber()

        self.weight_balancer = weight_balancer

        # annealing configuration
        self.beta_q_first = beta_q_first
        self.beta_q_last = beta_q_last
        self.beta_a_first = beta_a_first
        self.beta_a_last = beta_a_last
        self.annealing_step = annealing_step

        self.beta_q = self.beta_q_first
        self.beta_a = self.beta_a_first

    def start(self):
        self.encoder = self.encoder_builder()
        self.decoder = self.decoder_builder()
        self.encoder_target = self.encoder_builder()
        self.decoder_target = self.decoder_builder()

        self.logger = self.logger_builder() if self.logger_builder else None

        self.step = 0
        while True:
            metrics = self.train()
            if metrics:
                self.anneal()
                self.step += 1
                # lossのログ
                # TODO implement tensorboard
                if self.step % 100:
                    self.log_metrics(metrics)

                if self.step % self.synchronize_interval == 0:
                    self.synchronize()
                if self.step % self.upload_interval == 0:
                    self.upload()

    def train(self):

        # Serverからサンプル
        sample = self.server.sample(self.batch_size)
        if sample is None:
            time.sleep(1)
            return
        graph = tf.constant(sample["graph"])
        traj = tf.constant(sample["traj"])
        action = tf.constant(sample["act"])
        reward = tf.constant(sample["rew"])
        next_graph = tf.constant(sample["next_graph"])
        next_traj = tf.constant(sample["next_traj"])
        done = tf.constant(sample["done"])
        Q = tf.constant(sample["Q"])

        # Gradient Step
        td_loss, amortized_loss = self.train_on_batch(
            graph,
            traj,
            action,
            reward,
            next_graph,
            next_traj,
            done,
            Q
        )

        metrics = {"td_loss": td_loss.numpy(
        ), "amortized_loss": amortized_loss.numpy()}

        return metrics

    @tf.function
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
        next_mask = create_mask(next_trajectory)
        next_Q_list = self.network([next_graph, next_trajectory])
        next_action = masked_argmax(next_Q_list, next_mask)
        indice = tf.stack(
            [tf.range(next_action.shape[0]), next_action], axis=1)
        next_Q = tf.reshape(tf.gather_nd(self.network_target(
            [next_graph, next_trajectory]), indice), (-1, 1))
        target = reward + self.gamma * next_Q * \
            ((tf.cast(done, tf.float32) - 1) * (-1))

        # TD Loss と Amortized Loss を計算
        # Q target計算用のindice
        indice = tf.stack([tf.range(action.shape[0]),
                           tf.squeeze(action, axis=1)], axis=1)

        # Amortized Loss用のmask
        mask = create_mask(trajectory)

        with tf.GradientTape() as tape:
            # Calculate TDError
            Q_list = self.network([graph, trajectory])
            Q = tf.gather_nd(Q_list, indice)
            td_loss = self.huber(Q, target)

            # Calculate masked cross entropy
            amortized_loss = masked_cross_entropy_from_Q(Q_list, Q_mcts, mask)
            total_loss = self.beta_q * td_loss + self.beta_a * amortized_loss
            gradient = tape.gradient(total_loss, self.trainable_variables())
            self.optimizer.apply_gradients(
                zip(gradient, self.trainable_variables()))

        # Return Loss
        return td_loss, amortized_loss

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

    def log_metrics(self, metrics):
        if self.logger:
            self.logger.log(metrics, self.step)
        else:
            print(f"train_step: {self.step}  loss: {metrics['loss']}")
