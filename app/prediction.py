import numpy as np
import os
import pickle
from typing import Any, List, Dict
from sklearn.base import ClassifierMixin
from api import PredictRequest


def load_models(base_path='./models/') -> (bool, List[str], List[ClassifierMixin]):
    classifiers_filename = os.path.join(base_path, 'Top10Classifiers1')
    if not os.path.isfile(classifiers_filename):
        return False, None, None

    top_10_classifiers = np.load(classifiers_filename)
    model_list = []
    print(top_10_classifiers)
    for (accu, model_desc, model_name) in top_10_classifiers:
        model_path = os.path.join(base_path, model_name)
        if not os.path.isfile(model_path):
            return False, None, None
        model_list.append(pickle.load(open(model_path, 'rb')))
    return True, top_10_classifiers[:, 1], model_list


(models_loaded, models_names, models) = load_models()


def predict(features: PredictRequest) -> int:
    pass
