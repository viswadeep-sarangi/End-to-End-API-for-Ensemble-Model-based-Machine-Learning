import pickle
import os
import uuid
from numpy.typing import ArrayLike
from typing import Dict
import numpy as np
from app.data_structures import ModelName


def save_model_sequential(model, path="../app/models") -> str:
    num_models_already_saved = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    filename = "model{0}.sav".format(num_models_already_saved + 1)
    filepath = os.path.join(path, filename)
    pickle.dump(model, open(filepath, 'wb'))
    return filename


def save_model_uuid(model, path="../app/models") -> str:
    filename = "{0}.sav".format(uuid.uuid4().hex)
    filepath = os.path.join(path, filename)
    pickle.dump(model, open(filepath, 'wb'))
    return filename


def check_model_exists(filename: str, path="../app/models") -> bool:
    filepath = os.path.join(path, filename)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        return True, filepath
    else:
        return False, None


def update_ensemble_models_file(models_dict: Dict[str, str], path="../app/models", filename='ensemble_models'):
    filepath = os.path.join(path, filename)
    with open(filepath, 'wb') as f:
        np.save(f, models_dict)


def update_specific_ensemble_model(model_name: ModelName, model_filename: str):
    existing_models_file = load_ensemble_models_file()
    existing_models_file[model_name.value] = model_filename
    update_ensemble_models_file(existing_models_file)


def load_ensemble_models_file(path="../app/models", filename='ensemble_models') -> Dict[str, str]:
    filepath = os.path.join(path, filename)
    if not os.path.isfile(filepath):
        return None
    else:
        ensemble_models_file = np.load(filepath, allow_pickle=True)
        return ensemble_models_file.item()
