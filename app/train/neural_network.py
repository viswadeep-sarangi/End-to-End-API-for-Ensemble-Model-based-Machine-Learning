from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from numpy.typing import ArrayLike


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

    def train(self, x_train, x_test, y_train, y_test) -> ArrayLike:
        best_model = None

        for solv_index, common_solver in enumerate(self.models):
            for activ_index, clf in enumerate(common_solver):
                clf.fit(x_train, y_train)
                predicted = clf.predict(x_test)
                accuracy = accuracy_score(y_test, predicted)
                desc = "Model:{0}, Solver:{1}, Activation:{2}, Accuracy:{3}".format("NeuralNetwork",
                                                                                   NeuralNetwork.get_solvers()[solv_index],
                                                                                   NeuralNetwork.get_activations()[activ_index],
                                                                                   accuracy)
                print(desc)

                if best_model is None:
                    best_model = [clf, accuracy, desc]
                elif accuracy > float(best_model[1]):
                    best_model = [clf, accuracy, desc]

        print("Best Model: {0}".format(best_model))
        return best_model
