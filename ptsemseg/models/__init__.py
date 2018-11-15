import torchvision.models as models

from ptsemseg.models.bisenet import *


def get_model(name, n_classes):
    model = _get_model_instance(name)

    if name == 'bisenet':
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'bisenet': bisenet,
        }[name]
    except:
        print('Model {} not available'.format(name))
