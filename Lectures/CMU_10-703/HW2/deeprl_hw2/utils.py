"""Common functions you may find useful in your implementation."""

import semver
import tensorflow as tf
from keras.models import model_from_config, Model
import keras
import keras.layers
import keras.models
import keras.backend as K

# construct new model
def Model(input, output, **kwargs):
    if int(keras.__version__.split('.')[0]) >= 2:
        return keras.models.Model(inputs=input, outputs=output, **kwargs)
    else:
        return keras.models.Model(input=input, output=output, **kwargs)


def clone_model(model, custom_objects={}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone

def get_object_config(o):
    if o is None:
        return None
        
    config = {
        'class_name': o.__class__.__name__,
        'config': o.get_config()
    }
    return config


def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))
