import pickle
import os
import uuid
from typing import Dict
import numpy as np
from .train_data_structures import ModelName


def save_model_sequential(model, path="./models") -> str:
    """
    Saves the moel to disk with the prefix 'model' followed by the number of files currently present in the directory with a '.sav' file extension
    :param model: The model to be saved
    :param path: The directory to be saved in. Defaults to './models'
    :return: Name of the model after saving it to the disk
    """
    num_models_already_saved = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    filename = "model{0}.sav".format(num_models_already_saved)
    filepath = os.path.join(path, filename)
    pickle.dump(model, open(filepath, 'wb'))
    return filename


def save_model_uuid(model, path="./models") -> str:
    """
    Saves the model to disk by generating a unique identifier name and a .sav file extension
    :param model: The model to be saved
    :param path: The directory to be saved in. Defaults to './models'
    :return: Name of the model after saving it to the disk
    """
    filename = "{0}.sav".format(uuid.uuid4().hex)
    filepath = os.path.join(path, filename)
    pickle.dump(model, open(filepath, 'wb'))
    return filename


def check_model_exists(filename: str, path="./models") -> bool:
    """
    Checks whether a specific model exists in the path provided
    :param filename: Name of the model filename to check
    :param path: The directory where to check it in. Defaults to './models'
    :return: True or False, depending on whether the model file exists in the directory
    """
    filepath = os.path.join(path, filename)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        return True, filepath
    else:
        return False, None


def update_ensemble_models_file(models_dict: Dict[str, str], path="./models", filename='ensemble_models'):
    """
    Stores an ensemble models file in the disk, containing the details of the ensemble models; their model type and specific stored model file name
    :param models_dict: Dictionary of the type of the model (key) and path to the model file to be used during prediction (value)
    :param path: The directory of where to find the ensemble models file. Defaults to './models'
    :param filename: The name of the ensemble models file which contains information about the type of model and name of the stored model file. Defaults to 'ensemble_models'
    :return: None
    """
    filepath = os.path.join(path, filename)
    with open(filepath, 'wb') as f:
        np.save(f, models_dict)


def update_specific_ensemble_model(model_name: ModelName, model_filename: str):
    """
    Updates the stored ensemble models file in the disk, by replacing the type of model with the model filename provided in arguments
    :param model_name: The type of models in the ensemble of models to be updated
    :param model_filename: New model filename which would replace the existing filename associated with the type of model
    :return: None
    """
    existing_models_file = load_ensemble_models_file()
    existing_models_file[model_name.value] = model_filename
    update_ensemble_models_file(existing_models_file)


def load_ensemble_models_file(path="./models", filename='ensemble_models') -> Dict[str, str]:
    """
    Loads the ensemble model file and returns it as a dictionary
    :param path: The directory of where to search for the ensemble models file. Defaults to './models'
    :param filename: Name of the ensemble models file. Defaults to './ensemble_models'
    :return: Dictionary of the type of model (key) and the associated saved model file name on disk (value)
    """
    filepath = os.path.join(path, filename)
    if not os.path.isfile(filepath):
        return None
    else:
        ensemble_models_file = np.load(filepath, allow_pickle=True)
        return ensemble_models_file.item()
