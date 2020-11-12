import multiprocessing
from multiprocessing import Process

import numpy as np
from gat.environment.env import Env
from gat.modules.models.decoder import QDecoder
from gat.modules.models.encoder import Encoder
from gat.save.agents.actor import Actor
from gat.save.agents.learner import Learner
from gat.save.server.server import Server
from logger import TFLogger


class SAVE:
    def __init__(
        self,
        n_nodes,
        logdir="./logs/",
        actor_annealing_step=5000,
        learner_annealing_step=5000,
        n_actor=5,
        search_num=20,
        learning_rate=1.0e-3,
        batch_size=512,
        buffer_size=20000,
        gamma=1.0,
        actor_upload_interval=10,
        download_interval=10,
        synchronize_interval=10,
        learner_upload_interval=50,
        d_model=128,
        d_key=16,
        n_heads=8,
        depth=3,
        weight_balancer=1
    ):

        # Game Parameter
        self.n_nodes = n_nodes

        # Learning Hyper Parameters
        self.actor_annealing_step = actor_annealing_step
        self.learner_annealing_step = learner_annealing_step
        self.n_actor = n_actor
        self.search_num = search_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.actor_upload_interval = actor_upload_interval
        self.download_interval = download_interval
        self.synchronize_interval = synchronize_interval
        self.learner_upload_interval = learner_upload_interval

        self.env_builder = EnvBuilder()
        self.encoder_builder = EncoderBuilder(
            d_model=d_model,
            d_key=d_key,
            n_heads=n_heads,
            depth=depth,
            weight_balancer=weight_balancer
        )
        self.decoder_builder = DecoderBuilder(
            d_model=d_model,
            d_key=d_key,
            n_heads=n_heads,
            weight_balancer=weight_balancer
        )
        self.logger_builder = LoggerBuilder(logdir=logdir)

    def start(self):
        multiprocessing.set_start_method('spawn', force=True)

        env_dict = {
            "graph": {"shape": (14, 2), "dtype": np.float32},
            "traj": {"shape": (14, ), "dtype": np.int32},
            "act": {"dtype": np.int32},
            "rew": {"dtype": np.float32},
            "next_graph": {"shape": (14, 2), "dtype": np.float32},
            "next_traj": {"shape": (14, ), "dtype": np.int32},
            "done": {"dtype": np.bool},
            "Q": {"shape": (14, )}
        }
        server = Server(
            size=self.buffer_size, env_dict=env_dict)

        learner = Learner(
            encoder_builder=self.encoder_builder,
            decoder_builder=self.decoder_builder,
            server=server,
            logger_builder=self.logger_builder,
            batch_size=self.batch_size,
            gamma=self.gamma,
            learning_rate=self.learning_rate,
            synchronize_interval=self.synchronize_interval,
            upload_interval=self.learner_upload_interval,
            annealing_step=self.learner_annealing_step
        )
        learner_process = Process(target=learner.start)
        learner_process.start()

        actor_process_list = []
        for _ in range(self.n_actor):
            actor = Actor(
                env_builder=self.env_builder,
                encoder_builder=self.encoder_builder,
                decoder_builder=self.decoder_builder,
                server=server,
                logger_builder=self.logger_builder,
                gamma=self.gamma,
                upload_interval=self.actor_upload_interval,
                download_interval=self.download_interval,
                n_nodes=self.n_nodes,
                search_num=self.search_num,
                annealing_step=self.actor_annealing_step
            )
            p = Process(target=actor.start)
            actor_process_list.append(p)
            p.start()

        # server process must be started finally
        server.start()

        for actor_process in actor_process_list:
            actor_process.join()
        learner_process.join()

###############################################################################
# We cannot pickle tf objects, so we lazily load objects using builders       #
# in sub processes.                                                           #
# We cannot pickle local functions, so we use classes as lazy builders object #
###############################################################################


class EnvBuilder:
    def __call__(self):
        return Env()


class EncoderBuilder:
    def __init__(
        self,
        d_model=3,
        d_key=4,
        n_heads=5,
        depth=6,
        weight_balancer=7
    ):
        self.d_model = d_model
        self.d_key = d_key
        self.n_heads = n_heads
        self.depth = depth
        self.weight_balancer = weight_balancer

    def __call__(self):
        return Encoder(
            d_model=self.d_model,
            d_key=self.d_key,
            n_heads=self.n_heads,
            depth=self.depth,
            weight_balancer=self.weight_balancer
        )


class DecoderBuilder:
    def __init__(
        self,
        d_model,
        d_key,
        n_heads,
        weight_balancer
    ):
        self.d_model = d_model
        self.d_key = d_key
        self.n_heads = n_heads
        self.weight_balancer = weight_balancer

    def __call__(self):
        return QDecoder(
            d_model=self.d_model,
            d_key=self.d_key,
            n_heads=self.n_heads,
            weight_balancer=self.weight_balancer
        )


class LoggerBuilder:
    def __init__(self, logdir):
        self.logdir = logdir

    def __call__(self):
        # loggers are instantiated in each process for safety.
        logger = TFLogger(logdir=self.logdir)
        return logger
