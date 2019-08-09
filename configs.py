class HyperParameters(object):

    #Learning rate parameters
    max_epochs = 75
    save_epochs = 1
    batch_size = 16
    num_workers = 4
    num_classes = 8
    validation_per_class = 100
    seed = 888
    num_epochs_per_decay = 15
    num_epochs_per_eval = 1
    initial_learning_rate = 1e-2
    learning_rate_decay = 0.94
    weight_decay = 1e-4
    gamma = 0.1

    #directory parameters
    model_dir = './model/'


hyperparameters = HyperParameters()
