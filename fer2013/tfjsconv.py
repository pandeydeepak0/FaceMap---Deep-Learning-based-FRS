import keras
import tensorflow as tf
import tensorflowjs as tfjs

file_name= 'TestData/model_weights.hdf5'
model = open(file_name)
tfjs_target_dir= 'tfjs_path'
convert=tfjs.converters.save_keras_model(model, tfjs_target_dir)
converter.convert()


