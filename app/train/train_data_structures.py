from enum import Enum


class ModelName(str, Enum):
    svm = "svm"
    decisiontree = "decisiontree"
    neuralnetwork = "neuralnetwork"