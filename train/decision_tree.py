from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
from numpy.typing import ArrayLike
import save_model


class DecisionTree:

    @staticmethod
    def get_criteria():
        return ['gini', 'entropy']

    def __init__(self):
        self.models = []
        for criterion in DecisionTree.get_criteria():
            self.models.append(tree.DecisionTreeClassifier(criterion=criterion))

    def train(self, x_train, x_test, y_train, y_test, top_10_classifiers: ArrayLike) -> ArrayLike:
        top_10_accuracies = np.array(top_10_classifiers[:, 0], dtype=float)
        for i, clf in enumerate(self.models):
            clf.fit(x_train, y_train)
            predicted = clf.predict(x_test)
            accuracy = accuracy_score(y_test, predicted)
            print("Model:{0}, Criterion:{1}, Accuracy:{2}".format("DecisionTreeClassifier",
                                                                  DecisionTree.get_criteria()[i],
                                                                  accuracy))
            if accuracy > np.amin(top_10_accuracies):
                model_filename = save_model.save_model(clf)
                index = np.argmin(top_10_accuracies)
                top_10_accuracies[index] = accuracy
                top_10_classifiers[index] = [
                    accuracy,
                    "Model:{0}, Criterion:{1}, Accuracy:{2}".format("DecisionTreeClassifier",
                                                                    DecisionTree.get_criteria()[i], accuracy),
                    model_filename
                ]
        return top_10_classifiers
