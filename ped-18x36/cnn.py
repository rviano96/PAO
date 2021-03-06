
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import os

def cnn(size):
    model = Sequential()
    model.add(Convolution2D(16, 5, 5, W_regularizer=l2(5e-4), border_mode='same', input_shape=(size[0], size[1], 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 5, 5, W_regularizer=l2(5e-4), border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, W_regularizer=l2(5e-4), border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])
    #Save the model
    model_json = model.to_json()
    with open("modelcnn-18x36.json", "w") as json_file:
        print("Saving model into: ",os.getcwd() + "/modelcnn-18x36.json" )
        json_file.write(model_json)

    return model 

