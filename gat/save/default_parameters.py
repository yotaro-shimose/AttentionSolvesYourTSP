default_parameters = {
    # 学習収束目安
    'actor_annealing_step': 5000,
    'learner_annealing_step': 5000,

    # NNパラメータ
    'd_model': 128,
    'd_key': 16,
    'n_heads': 8,
    'depth': 3,
    'weight_balancer': 0.001,

    # 学習アルゴリズムパラメータ
    'n_actor': 2,
    'learning_rate': 1.0e-3,
    'batch_size': 512,
    'gamma': 0.99999,
    'search_num': 20,
    'actor_upload_interval': 10,
    'download_interval': 10,
    'buffer_size': 20000,

    'synchronize_interval': 10,
    'learner_upload_interval': 50,

    # 環境用パラメータ
    'n_nodes': 14,

    # TensorBoard用ログディレクトリ
    'logdir': "./logs/",

}
