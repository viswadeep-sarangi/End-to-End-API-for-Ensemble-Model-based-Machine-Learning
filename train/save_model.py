import pickle
import os


def save_model(model, path="../app/models") -> str:
    num_models_already_saved = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    filename = "model{0}.sav".format(num_models_already_saved + 1)
    filepath = os.path.join(path, filename)
    pickle.dump(model, open(filepath, 'wb'))
    return filename
