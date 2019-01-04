from keras.layers import Conv2D, Flatten, Lambda, MaxPooling2D, Dropout
from keras.models import Model, Sequential
import glob
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split as trainTestSplit
import pickle
import os
from keras.callbacks import ModelCheckpoint
from model.model import poolerPico
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import pandas as pd

def main():
    
    nb_epochs=50
    nb_train_samples = 4000 + 3840 #3840 pedestrian - 4000 notPedestrian
    nb_validation_samples = 1000 + 960 #960 pedestrian/ 1000 notPedestrian
    #images_dim = 64
    images_dim = 64 # heigth = 64, width = 64
    
    sourceModel, modelName = poolerPico()
   
    
    # Adding fully-connected layer to train the 'classifier'
    x = sourceModel.output
    x = Flatten()(x)
    model = Model(inputs=sourceModel.input, outputs=x)
    print(model.summary())
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
     #Save the model
    model_json = model.to_json()
    with open("modelcnn-64x64.json", "w") as json_file:
        print("Saving model into: ",os.getcwd() + "/modelcnn-64x64.json" )
        json_file.write(model_json)
        
    #train the model    
    train(model, nb_epochs, nb_train_samples, nb_validation_samples, images_dim)
    
def data_process(size, nb_train_samples, nb_validation_samples):
    path = '/media/rodrigo/Rodrigo/PAO/PedestrianTracking-V3/data/ped-64x64/data/'
    #  Using the data Augmentation in traning data
    datagen1 = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    datagen2 = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen1.flow_from_directory(
        path + 'train',
        target_size=(size, size),
        batch_size=32,
        class_mode='binary')

    validation_generator = datagen2.flow_from_directory(
        path + 'valid',
        target_size=(size, size),
        batch_size=32,
        class_mode='binary')

    return train_generator, validation_generator


def train(model, nb_epochs, nb_train_samples, nb_validation_samples, size):
    
    train_generator, validation_generator = data_process(size, nb_train_samples, nb_validation_samples)
    
    #checkpoint
    checkpointer = ModelCheckpoint(filepath='weights-64x64.h5',
                                       monitor='val_acc', verbose=0, save_best_only=True)
    
    hist = model.fit_generator(
        train_generator, #train data
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epochs,
        validation_data=validation_generator,  #validation data
        nb_val_samples=nb_validation_samples, callbacks = [checkpointer])
        
    # Saving the loss and acc during the traning time.
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('hist-64x64.csv', encoding='utf-8', index=False)
    model.save('weights-64x64.h5')
    plotHistory(hist)
    
    evaluateOnTestSet(model,50,size)
    
def evaluateOnTestSet(model,testSteps,size):
    path = '/media/rodrigo/Rodrigo/PAO/VGG-BdBxs/data/valid/'
    datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = datagen.flow_from_directory(
        path ,
        target_size=(size, size),
        batch_size=32,
        class_mode='binary')
    accuracy = model.evaluate_generator(generator=test_generator, steps=testSteps)

    print('test accuracy: ', accuracy)
    
def plotHistory(history):
    #ACC VS VAL_ACC
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy ACC VS VAL_ACC')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    #LOSS VS VAL_LOSS
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss LOSS vs VAL_LOSS')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
 
