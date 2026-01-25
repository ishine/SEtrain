import importlib

def import_model(path, classname):
    module = importlib.import_module(path)
    model_class = getattr(module, classname)
    return model_class
