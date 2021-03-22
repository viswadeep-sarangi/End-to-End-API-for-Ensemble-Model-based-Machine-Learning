import csv
import string
from app.data_structures import TrainRequest, TrainResponse, ModelName
from train import train_model, save_model
from numpy.typing import ArrayLike
from app import prediction

base_address = "../training_data"


def train_specific_model(data_file: str, model_name: ModelName) -> (bool, ArrayLike):
    if not validate_csv(data_file):
        return False, [None, None, None]
    trained_model = train_model.train_model(model_name=model_name, file=data_file)
    model_filename = save_model.save_model_uuid(trained_model)
    save_model.update_specific_ensemble_model(model_name=model_name, model_filename=model_filename)
    prediction.load_ensemble_models()
    return True, [model_filename, *trained_model[1:]]


def validate_csv(filename: str) -> bool:
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
