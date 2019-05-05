import keras 

from ImagePreprocessing import data_gen, regulariser
from DataGenerator import xtrain, xtest, ytrain, ytest

from keras import layers
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, BatchNormalization
from keras.models import Sequential
from keras.models import Model
from keras.layers import MaxPooling2D, Dropout, Flatten, Input
from keras.layers import SeparableConv2D

import numpy as np
import pandas as pd
import cv2

#model parameters
image_shape= (48, 48, 1)
epochs= 10
batch_size= 128
no_classes= 7
train_sample= len(xtrain)
test_sample= len(xtest)
path= 'FER2013/'

#X_train = X[:n_samples_train].reshape(n_samples_train, -1)
#y_train = y[:n_samples_train]
#data_gen.fit(xtrain)

"""functional model API in keras

a = Input(shape=(32,))
b = Dense(32)(a)

"""
#model- mini-Xception 2017

"""mini-Xception Architecture
            ----------------
            conv2D/BatchNorm 
            ----------------
                    |
            ----------------
            conv2d/BatchNorm
            ----------------
                    |
_______________________________________________
        ---------------------                 |
        |                    |                |
    ----------------    -------------------   |
    conv2d/BatchNorm    sepconv2d/BatchNorm   |
    ----------------    -------------------   |
                            |                 |
                        ------------------- | 4X |
                        sepconv2d/BatchNorm   |
                        -------------------   |
                            |                 |
                        -------------------   |
                            MaxPooling2d      |
                        -------------------   |
______________________________________________|
                    |
            ----------------
                conv2D
            ----------------
                    |
          ---------------------
          GlobalAvg. Pooling 2d
          ---------------------
                    |
            ----------------
                Softmax
            ----------------

"""
image_shape= Input(image_shape)
model= Conv2D(8, (3,3), strides=(1, 1), activation='relu', kernel_regularizer=regulariser, use_bias=True)(image_shape)
model= BatchNormalization(axis=-1, center=True)(model)

model= Conv2D(8, (3,3), strides=(1, 1), activation='relu', kernel_regularizer=regulariser, use_bias=True)(model)
model= BatchNormalization(axis=-1, center=True)(model)

#output = keras.layers.concatenate([model], axis=1)
#model = Model(image_shape, output)
#model.summary()

#layer1
residual_model= Conv2D(16, (1,1), strides=(2, 2), padding='same') (model)
residual_model= BatchNormalization()(residual_model)

model= SeparableConv2D(16, (3, 3), padding='same', activation='relu', use_bias=False, kernel_regularizer= regulariser)(model)
model= BatchNormalization(axis=-1, center=True)(model)
model= SeparableConv2D(16, (3, 3), padding='same', activation='relu', use_bias=False, kernel_regularizer= regulariser)(model)
model= BatchNormalization(axis=-1, center=True)(model)
model= MaxPooling2D((3, 3), strides=(2, 2), padding='same')(model)

model= layers.add([model, residual_model])

#layer2
residual_model= Conv2D(32, (1,1), strides=(2, 2), padding='same') (model)
residual_model= BatchNormalization()(residual_model)

model= SeparableConv2D(32, (3, 3), padding='same', activation='relu', use_bias=False, kernel_regularizer= regulariser)(model)
model= BatchNormalization(axis=-1, center=True)(model)
model= SeparableConv2D(32, (3, 3), padding='same', activation='relu', use_bias=False, kernel_regularizer= regulariser)(model)
model= BatchNormalization(axis=-1, center=True)(model)
model= MaxPooling2D((3, 3), strides=(2, 2), padding='same')(model)

model= layers.add([model, residual_model])

#layer3
residual_model= Conv2D(64, (1,1), strides=(2, 2), padding='same') (model)
residual_model= BatchNormalization()(residual_model)

model= SeparableConv2D(64, (3, 3), padding='same', activation='relu', use_bias=False, kernel_regularizer= regulariser)(model)
model= BatchNormalization(axis=-1, center=True)(model)
model= SeparableConv2D(64, (3, 3), padding='same', activation='relu', use_bias=False, kernel_regularizer= regulariser)(model)
model= BatchNormalization(axis=-1, center=True)(model)
model= MaxPooling2D((3, 3), strides=(2, 2), padding='same')(model)

model= layers.add([model, residual_model])

#layer4
residual_model= Conv2D(128, (1,1), strides=(2, 2), padding='same') (model)
residual_model= BatchNormalization()(residual_model)

model= SeparableConv2D(128, (3, 3), padding='same', activation='relu', use_bias=False, kernel_regularizer= regulariser)(model)
model= BatchNormalization(axis=-1, center=True)(model)
model= SeparableConv2D(128, (3, 3), padding='same', activation='relu', use_bias=False, kernel_regularizer= regulariser)(model)
model= BatchNormalization(axis=-1, center=True)(model)
model= MaxPooling2D((3, 3), strides=(2, 2), padding='same')(model)

model= layers.add([model, residual_model])

#final layers
model = Conv2D(no_classes, (3, 3), padding='same')(model)
model = GlobalAveragePooling2D()(model)
output = Activation('softmax',name='predictions')(model)

model= Model(image_shape, output)
model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
#model.summary()


model_train= model.fit_generator(data_gen.flow(xtrain, ytrain, batch_size),
                        steps_per_epoch=len(xtrain) / batch_size,
                        epochs=epochs, validation_data=(xtest,ytest))


#prediction= model.predict(xtest, ytest)
#score=model.evaluate(xtest, ytest)

#saving model in hdf5
save_model = model.save(path + 'model_weights' + '.h5')


#Loss and Accuracy curves
plt.figure(figsize=[10,8])
plt.plot(model_train.history['loss'], 'g', linewidth=0.5)
plt.plot(model_train.history['val_loss'], 'r', linewidth=3.0)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])

plt.figure(figsize=[10,8])
plt.plot(model_train.history['acc'], 'g', linewidth=0.5)
plt.plot(model_train.history['val_acc'], 'r', linewidth=3.0)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
