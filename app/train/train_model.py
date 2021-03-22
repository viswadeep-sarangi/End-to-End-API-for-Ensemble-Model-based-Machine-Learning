import os

from sklearn.model_selection import train_test_split
import pandas as pd
from numpy.typing import ArrayLike
from .svm import SVMClassifier
from .decision_tree import DecisionTree
from .neural_network import NeuralNetwork
from .train_data_structures import ModelName
from . import save_model


def load_data_from_csv(file: str) -> (ArrayLike, ArrayLike):
    data = pd.read_csv(file)
    target = data.pop('DEATH_EVENT').values
    print(type(data.to_numpy()))
    print(type(target))
    return data.to_numpy(), target


def train_model(model_name: ModelName, file: str, test_size=0.15):
    data, target = load_data_from_csv(file=file)

    if model_name == ModelName.svm:
        model = SVMClassifier()
    elif model_name == ModelName.decisiontree:
        model = DecisionTree()
    elif model_name == ModelName.neuralnetwork:
        model = NeuralNetwork()
    else:
        raise ValueError("Incorrect model name", model_name)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=42)
    [classifier, accuracy, model_desc] = model.train(x_train, x_test, y_train, y_test)
    print("Classifier:{0}, Accuracy:{1}, Model Description:{2}".format(classifier, accuracy, model_desc))
    return [classifier, accuracy, model_desc]


def train_all_models(filename: str, test_size=0.15):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print("Current Directory: {0}".format(dir_path))
    data, target = load_data_from_csv(filename)
    print(data)
    print(target)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=42)

    # SVM
    svm_models = SVMClassifier()
    svm_best_model = svm_models.train(x_train, x_test, y_train, y_test)

    # DecisionTree
    dt_models = DecisionTree()
    dt_best_model = dt_models.train(x_train, x_test, y_train, y_test)

    # Neural Network
    nn_model = NeuralNetwork()
    nn_best_model = nn_model.train(x_train, x_test, y_train, y_test)

    # Saving all the best models
    svm_model_name = save_model.save_model_uuid(svm_best_model)
    dt_model_name = save_model.save_model_uuid(dt_best_model)
    nn_model_name = save_model.save_model_uuid(nn_best_model)

    models_file_contents = {ModelName.svm.value: svm_model_name, ModelName.decisiontree.value: dt_model_name, ModelName.neuralnetwork.value: nn_model_name}
    save_model.update_ensemble_models_file(models_file_contents)


if __name__ == "__main__":
    train_all_models()
