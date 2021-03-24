import numpy as np
import os
from typing import List, Dict
from sklearn.base import ClassifierMixin
from data_structures import PredictRequest
from train import save_model, train_model

global ensemble_models  # this contains the actual sklearn models after its loaded using the load_models(...) function

# This variable houses the sequence of keys expected in the 'features' variable of 'PredictRequest' object provided during during a call to the '/predict/ API endpoint
feature_names = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure',
                 'platelets', 'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']


def load_ensemble_models():
    """
    This function loads the data stored in the './models/ensemble_models' file and uses it to load the actual sklearn models from the same directory
    :return: Doesn't return a value
    """
    models_file = save_model.load_ensemble_models_file()

    if models_file is None:
        train_model.train_all_models('./training_data/heart_failure_clinical_records_dataset.csv')
        models_file = save_model.load_ensemble_models_file()

    print("Models:{0}".format(models_file))
    models_loaded, models = load_models(model_dict=models_file)

    global ensemble_models
    if models_loaded:
        ensemble_models = models
    else:
        ensemble_models = None


def load_models(model_dict: Dict[str, str], base_path='./models/') -> (bool, Dict[str, ClassifierMixin]):
    """
    This function loads the actual sklearn models mentioned in 'model_dict'
    :param model_dict: The dictionary of the type of the model as key and the stored sklearn model name as value
    :param base_path: (Optional) Defines where to find the stored sklearn models. Defaults to the './models' directory
    :return: Returns a bool defining whether the models were sucessfully loaded. If True, returns the models with it
    """
    models = {}
    for (key, value) in model_dict.items():
        model_path = os.path.join(base_path, value)
        _model = np.load(model_path, allow_pickle=True)
        models[key] = _model

    return True, models


def predict(features: PredictRequest) -> (bool, List[Dict[str, str]]):
    """
    Returns the predicted value based on the multiple ensemble models
    :param features: The 'PredictRequest' object provided during the API call to the '/predict' API endpoint
    :return: Whether the prediction was performed successfully (True/False), along with the prediction of the ensemnle of models based on the type of the model
    """
    output = {}
    features_converted, feature_list = convert_features_dict_to_list(features.features)

    if not features_converted:
        return False, None

    for (model_name, [model, accuracy, model_desc]) in ensemble_models.items():
        prediction = model.predict([feature_list])
        output[model_name] = prediction

    return True, output


def convert_features_dict_to_list(features: Dict[str, float]) -> (bool, List[float]):
    """
    Utility function to convert the dictionary of features into a list of values for easily consumption by the models
    :param features:
    :return:
    """
    output = []
    for feature_name in feature_names:
        if feature_name not in features:
            return False, None
        output.append(features[feature_name])
    return True, output


load_ensemble_models() # loads the models during import
