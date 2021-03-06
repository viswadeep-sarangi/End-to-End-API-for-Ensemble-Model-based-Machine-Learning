from sklearn import svm
from sklearn.metrics import accuracy_score
from numpy.typing import ArrayLike


class SVMClassifier:

    @staticmethod
    def get_kernels():
        """
        Various kernel functions used for training the model
        :return: a list of strings, containing the kernel function names
        """
        return ['linear', 'rbf']

    def __init__(self):
        self.models = []
        for kernel in SVMClassifier.get_kernels():
            self.models.append(svm.SVC(kernel=kernel, gamma=2))

    def train(self, x_train, x_test, y_train, y_test) -> ArrayLike:
        """
        Trains the model based on the parameters
        :param x_train: Training dataset
        :param x_test: Testing dataset
        :param y_train: Training labels
        :param y_test: Testing labels
        :return: an array of [ sklearn model, the accuracy of the model on the training set, verbal description of the model ]
        """
        best_model = None

        for i, clf in enumerate(self.models):
            clf.fit(x_train, y_train)
            predicted = clf.predict(x_test)
            accuracy = accuracy_score(y_test, predicted)
            desc = "Model:{0}, Kernel:{1}, Accuracy:{2}".format("SVM",
                                                               SVMClassifier.get_kernels()[i],
                                                               accuracy)
            print(desc)

            if best_model is None:
                best_model = [clf, accuracy, desc]
            elif accuracy > float(best_model[1]):
                best_model = [clf, accuracy, desc]

        print("Best Model: {0}".format(best_model))
        return best_model
