import numpy as np
import os
from numpy.typing import ArrayLike
from typing import Any, List, Dict
from sklearn.base import ClassifierMixin
from app.data_structures import PredictRequest
from train import save_model, train_model

global ensemble_models

feature_names = ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']


def load_ensemble_models(base_path='./models/', filename='ensemble_models'):
    models_file = save_model.load_ensemble_models_file()

    if models_file is None:
        train_model.train_all_models('../training_data/heart_failure_clinical_records_dataset.csv')
        models_file = save_model.load_ensemble_models_file()

    print("Models:{0}".format(models_file))
    models_loaded, models = load_models(model_dict=models_file)

    global ensemble_models
    if models_loaded:
        ensemble_models = models
    else:
        ensemble_models = None


def load_models(model_dict: Dict[str, str], base_path='./models/') -> (bool, Dict[str, ClassifierMixin]):
    models = {}
    for (key, value) in model_dict.items():
        model_path = os.path.join(base_path, value)
        _model = np.load(model_path, allow_pickle=True)
        models[key] = _model

    return True, models


def predict(features: PredictRequest) -> (bool, List[Dict[str, str]]):
    output = {}
    features_converted, feature_list = convert_features_dict_to_list(features.features)

    if not features_converted:
        return False, None

    for (model_name, [model, accuracy, model_desc]) in ensemble_models.items():
        prediction = model.predict([feature_list])
        output[model_name] = prediction

    return True, output


def convert_features_dict_to_list(features:Dict[str, float]) -> (bool, List[float]):
    output = []
    for feature_name in feature_names:
        if feature_name not in features:
            return False, None
        output.append(features[feature_name])
    return True, output

load_ensemble_models()
