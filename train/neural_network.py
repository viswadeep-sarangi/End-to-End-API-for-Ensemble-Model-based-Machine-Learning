from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.typing import ArrayLike
import save_model


class NeuralNetwork:
    @staticmethod
    def get_activations():
        return ['logistic', 'tanh', 'relu']

    @staticmethod
    def get_solvers():
        return ['sgd', 'adam']

    def __init__(self):
        self.models = []
        for solver in NeuralNetwork.get_solvers():
            solver_models = []
            for activation in NeuralNetwork.get_activations():
                solver_models.append(MLPClassifier(activation=activation, solver=solver))
            self.models.append(solver_models)

    def train(self, x_train, x_test, y_train, y_test, top_10_classifiers: ArrayLike) -> ArrayLike:
        top_10_accuracies = np.array(top_10_classifiers[:, 0], dtype=float)
        for solv_index, common_solver in enumerate(self.models):
            for activ_index, clf in enumerate(common_solver):
                clf.fit(x_train, y_train)
                predicted = clf.predict(x_test)
                accuracy = accuracy_score(y_test, predicted)
                print("Model:{0}, Solver:{1}, Activation:{2}, Accuracy:{3}".format("NeuralNetwork",
                                                                                   NeuralNetwork.get_solvers()[solv_index],
                                                                                   NeuralNetwork.get_activations()[activ_index],
                                                                                   accuracy))
                if accuracy > np.amin(top_10_accuracies):
                    model_filename = save_model.save_model(clf)
                    index = np.argmin(top_10_accuracies)
                    top_10_accuracies[index] = accuracy
                    top_10_classifiers[index] = [
                        accuracy,
                        "Model:{0}, Solver:{1}, Activation:{2}, Accuracy:{3}".format("NeuralNetwork",
                                                                                     NeuralNetwork.get_solvers()[solv_index],
                                                                                     NeuralNetwork.get_activations()[activ_index],
                                                                                     accuracy),
                        model_filename
                    ]
        return top_10_classifiers
