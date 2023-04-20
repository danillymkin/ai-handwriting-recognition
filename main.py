from Network import Network
from dataset import DATASET, TEST_DATASET

import numpy as np


dataset = [(np.array([DATASET['data'][i]]), DATASET['target'][i]) for i in range(len(DATASET['target']))]
test_dataset = [(np.array([TEST_DATASET['data'][i]]), TEST_DATASET['target'][i]) for i in range(len(TEST_DATASET['target']))]

if __name__ == "__main__":
    network = Network([35, 13, 4])
    network.learn(dataset)

    accuracy = network.calc_accuracy(test_dataset)
    print("Accuracy:", accuracy)
    network.test(test_dataset[0][0])

    network.show_error_graph()
