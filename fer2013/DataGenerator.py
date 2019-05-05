import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv('Dataset/fer2013.csv')

width, height, channels = (48, 48, 1)

y = data['emotion']
X = np.zeros((len(data), width, height, channels))
for i in range(len(data)):
    X[i] = np.fromstring(data['pixels'][i], sep=' ').reshape(width, height, channels)

#data['emotion'] =["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
#X_train = X[:n_samples_train].reshape(n_samples_train, -1)
#y_train = y[:n_samples_train]
        
xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.2,shuffle=True)

print(xtrain.shape)
print(ytrain.shape)