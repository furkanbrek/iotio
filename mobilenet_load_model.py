import os

import tensorflow as tf

from tensorflow import keras

new_model = tf.keras.models.load_model('saved_model/my_model.keras')

new_model.summary()