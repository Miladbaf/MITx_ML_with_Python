# #! /usr/bin/env python

# import _pickle as cPickle, gzip
# import numpy as np
# from tqdm import tqdm
# import torch
# import torch.autograd as autograd
# import torch.nn.functional as F
# import torch.nn as nn
# import sys
# sys.path.append("..")
# import utils
# from utils import *
# from train_utils import batchify_data, run_epoch, train_model

# def main():
#     # Load the dataset
#     num_classes = 10
#     X_train, y_train, X_test, y_test = get_MNIST_data()

#     # Split into train and dev
#     dev_split_index = int(9 * len(X_train) / 10)
#     X_dev = X_train[dev_split_index:]
#     y_dev = y_train[dev_split_index:]
#     X_train = X_train[:dev_split_index]
#     y_train = y_train[:dev_split_index]

#     permutation = np.array([i for i in range(len(X_train))])
#     np.random.shuffle(permutation)
#     X_train = [X_train[i] for i in permutation]
#     y_train = [y_train[i] for i in permutation]

#     # Split dataset into batches
#     batch_size = 32
#     train_batches = batchify_data(X_train, y_train, batch_size)
#     dev_batches = batchify_data(X_dev, y_dev, batch_size)
#     test_batches = batchify_data(X_test, y_test, batch_size)

#     #################################
#     ## Model specification TODO
#     model = nn.Sequential(
#               nn.Linear(784, 10),
#               nn.ReLU(),
#               nn.Linear(10, 10),
#             )
#     lr=0.1
#     momentum=0
#     ##################################

#     train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

#     ## Evaluate the model on test data
#     loss, accuracy = run_epoch(test_batches, model.eval(), None)

#     print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))


# if __name__ == '__main__':
#     # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
#     np.random.seed(12321)  # for reproducibility
#     torch.manual_seed(12321)  # for reproducibility
#     main()

#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model

def get_model(hidden_size, activation):
    if activation == 'ReLU':
        return nn.Sequential(
            nn.Linear(784, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10)
        )
    elif activation == 'LeakyReLU':
        return nn.Sequential(
            nn.Linear(784, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size, 10)
        )

def main():
    # Load the dataset
    print("Loading MNIST data...")
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()
    print("Data loaded successfully.")

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    print("Batchifying data...")
    batch_size = 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)
    print("Data batchified successfully.")

    # Baseline configuration
    hidden_size = 10
    lr = 0.1
    momentum = 0
    activation = 'ReLU'

    # Configurations to test
    configurations = [
        {"hidden_size": 10, "activation": 'ReLU', "lr": 0.1, "momentum": 0},
        {"hidden_size": 50, "activation": 'ReLU', "lr": 0.1, "momentum": 0},
        {"hidden_size": 100, "activation": 'ReLU', "lr": 0.1, "momentum": 0},
        {"hidden_size": 10, "activation": 'LeakyReLU', "lr": 0.1, "momentum": 0},
        {"hidden_size": 10, "activation": 'ReLU', "lr": 0.01, "momentum": 0},
        {"hidden_size": 10, "activation": 'ReLU', "lr": 0.5, "momentum": 0},
        {"hidden_size": 10, "activation": 'ReLU', "lr": 0.1, "momentum": 0.9},
    ]

    best_acc = 0
    best_config = None

    for config in configurations:
        np.random.seed(12321)
        torch.manual_seed(12321)
        
        model = get_model(config["hidden_size"], config["activation"])
        
        print(f"Training with configuration: {config}")
        val_acc = train_model(train_batches, dev_batches, model, lr=config["lr"], momentum=config["momentum"], n_epochs=10)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_config = config

    print(f"Best configuration: {best_config} with validation accuracy: {best_acc}")

    ## Evaluate the best model on test data
    np.random.seed(12321)
    torch.manual_seed(12321)
    best_model = get_model(best_config["hidden_size"], best_config["activation"])
    train_model(train_batches, dev_batches, best_model, lr=best_config["lr"], momentum=best_config["momentum"], n_epochs=10)
    loss, accuracy = run_epoch(test_batches, best_model.eval(), None)

    print ("Loss on test set:"  + str(loss) + " Accuracy on test set: " + str(accuracy))


if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()