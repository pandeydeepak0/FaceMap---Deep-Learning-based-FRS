import keras
from keras.utils import np_utils
from keras.preprocessing import image
from keras import regularizers
from DataGenerator import xtrain, xtest, ytrain, ytest

import numpy as np
import pandas as pd

l2_reg= 0.01

xtrain= xtrain.astype('float32')/255
xtrain= (xtrain-0.5)*2
xtest= xtest.astype('float32')/255
xtest= (xtest-0.5)*2

data_gen= image.ImageDataGenerator(
    horizontal_flip=True,
    samplewise_center=False,
    validation_split= 0.0,
    rotation_range= 20,
    featurewise_std_normalization=False,
    width_shift_range= 1,
    zca_whitening= True,
    zoom_range= 0.1,
    data_format= "channels_last"
)

#param for models
regulariser= regularizers.l2(l2_reg)
