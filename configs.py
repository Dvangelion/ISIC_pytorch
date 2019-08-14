class HyperParameters(object):

    #Learning rate parameters
    max_epochs = 125
    num_epochs_per_decay = 25
    initial_learning_rate = 5e-4
    learning_rate_decay = 0.2
    save_epochs = 1
    batch_size = 40
    num_workers = 4
    num_classes = 8
    validation_per_class = 100
    seed = 888
    num_epochs_per_eval = 1
    weight_decay = 1e-4
    #gamma = 0.1

    #directory parameters
    model_dir = './model/'


hyperparameters = HyperParameters()
