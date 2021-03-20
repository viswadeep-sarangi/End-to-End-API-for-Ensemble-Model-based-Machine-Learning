from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from svm import SVMClassifier
from decision_tree import DecisionTree
from neural_network import NeuralNetwork


def load_data_from_csv(file: str = "./heart_failure_clinical_records_dataset.csv") -> (ArrayLike, ArrayLike):
    data = pd.read_csv(file)
    target = data.pop('DEATH_EVENT').values
    print(type(data.to_numpy()))
    print(type(target))
    return data.to_numpy(), target


def train(test_size=0.33, num_iterations = 10):
    top_10_classifiers = np.array([[0, "", ""] for _ in range(10)])
    data, target = load_data_from_csv()
    print(data)
    print(target)
    for _iter in range(num_iterations):
        print("Iteration: {0} ----------".format(_iter))
        print()

        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=42)

        '''print("{0}, {1}, {2}, {3}".format(type(x_train),
                                          type(x_test),
                                          type(y_train),
                                          type(y_test)))
        print("{0}, {1}, {2}, {3}".format(np.shape(x_train),
                                          np.shape(x_test),
                                          np.shape(y_train),
                                          np.shape(y_test)))
        '''
        # SVM
        svm_models = SVMClassifier()
        top_10_classifiers = svm_models.train(x_train, x_test, y_train, y_test, top_10_classifiers=top_10_classifiers)

        # DecisionTree
        dt_models = DecisionTree()
        top_10_classifiers = dt_models.train(x_train, x_test, y_train, y_test, top_10_classifiers=top_10_classifiers)

        # Neural Network
        nn_model = NeuralNetwork()
        top_10_classifiers = nn_model.train(x_train, x_test, y_train, y_test, top_10_classifiers=top_10_classifiers)

    print("Top 10 Classifiers")
    print(top_10_classifiers)

    with open("../app/models/Top10Classifiers", 'wb') as f:
        np.save(f, top_10_classifiers)


if __name__ == "__main__":
    train()
