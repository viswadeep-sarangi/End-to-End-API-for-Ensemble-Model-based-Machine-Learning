import csv
import string

from data_structures import TrainRequest, TrainResponse, ModelName
from train import train

base_address = "./training_data"


def train_specific_model(data_file: str, model_name: ModelName):
    if not validate_csv(data_file):
        return
    data, target = train.load_data_from_csv(data_file)


def validate_csv(filename: str) -> bool:
    # Refer to: https://stackoverflow.com/questions/2984888/check-if-file-has-a-csv-format-with-python
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
