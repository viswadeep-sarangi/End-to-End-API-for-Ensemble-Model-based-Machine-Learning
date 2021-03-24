import csv
import string
from train.train_data_structures import ModelName
from train import train_model, save_model
from numpy.typing import ArrayLike
import prediction

base_address = "./training_data"  # the default directory for storing and retrieving training_data provided with the call to the '/train' API endpoint


def train_specific_model(data_file: str, model_name: ModelName) -> (bool, ArrayLike):
    """
    Trains a specific type of model and adds it to the ensemble of models
    :param data_file: the .csv file used for training the model
    :param model_name: name of the model, like svm, decision tree or neural network
    :return: returns the success or failure of being able to train the model, alogn with the details of the model trained
    """
    if not validate_csv(data_file):
        return False, [None, None, None]
    trained_model = train_model.train_model(model_name=model_name, file=data_file)
    model_filename = save_model.save_model_uuid(trained_model)
    save_model.update_specific_ensemble_model(model_name=model_name, model_filename=model_filename)
    prediction.load_ensemble_models()
    return True, [model_filename, *trained_model[1:]]


def validate_csv(filename: str) -> bool:
    """
    Returns a True or False based on whether the file provided is a valid .csv file or not
    :param filename: name (and path) of the file to be evaluated
    :return: True/False based on whether the file was a valid .csv file or not
    """
    # From: https://stackoverflow.com/questions/2984888/check-if-file-has-a-csv-format-with-python
    try:
        with open(filename, newline='') as csvfile:
            start = csvfile.read(4096)

            # isprintable does not allow newlines, printable does not allow umlauts...
            if not all([c in string.printable or c.isprintable() for c in start]):
                return False
            dialect = csv.Sniffer().sniff(start)
            return True
    except csv.Error:
        # Could not get a csv dialect -> probably not a csv.
        return False
    except UnicodeError:
        return False
