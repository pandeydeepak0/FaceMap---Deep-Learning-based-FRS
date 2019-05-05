import keras
import tensorflow as tf
model_file = open('model_weights.hdf5', "rb")
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_file)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

