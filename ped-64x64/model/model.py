from keras.layers import Conv2D, Flatten, Lambda, MaxPooling2D, Dropout
from keras.models import Model, Sequential
import glob
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split as trainTestSplit
import pickle
import os
from keras.callbacks import ModelCheckpoint



def cnn(inputShape=(64, 64, 3)):

    model = Sequential()
    # Center and normalize our data
    model.add(Lambda(lambda x: x / 255., input_shape=inputShape, output_shape=inputShape))
    # Block 0
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='cv0',
                     input_shape=inputShape, padding="same"))
    model.add(Dropout(0.5))

    # Block 1
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='cv1', padding="same"))
    model.add(Dropout(0.5))

    # block 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='cv2', padding="same"))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))

    # binary 'classifier'
    model.add(Conv2D(filters=1, kernel_size=(8, 8), name='fcn', activation="sigmoid"))

    return model

