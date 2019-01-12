import os
import sys
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
#from keras.utils.visualize_util import plot
#from keras.utils.vis_utils import plot_model as plot
import matplotlib.pyplot as plt
import matplotlib

from cnn import cnn


def main():
    
    nb_epochs=50
    nb_train_samples = 4000 + 3840 #3840 pedestrian - 4000 notPedestrian
    nb_validation_samples = 1000 + 960 #960 pedestrian/ 1000 notPedestrian
    images_dim = 18,36 # heigths = 36, width = 18
    
    model = cnn(images_dim)
    train(model, nb_epochs, nb_train_samples, nb_validation_samples, images_dim)


def data_process(size, nb_train_samples, nb_validation_samples):
    path = '/media/rodrigo/Rodrigo/PAO/PedestrianTracking-V3/data/ped-18x36/'
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
        target_size=(size[0], size[1]),
        batch_size=32,
        class_mode='binary')

    validation_generator = datagen2.flow_from_directory(
        path + 'valid',
        target_size=(size[0], size[1]),
        batch_size=32,
        class_mode='binary')

    return train_generator, validation_generator


def train(model, nb_epochs, nb_train_samples, nb_validation_samples, size):
    
    train_generator, validation_generator = data_process(size, nb_train_samples, nb_validation_samples)
    
    # Using the early stopping technique to prevent overfitting.
    earlyStopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')

    hist = model.fit_generator(
        train_generator, #train data
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epochs,
        validation_data=validation_generator,  #validation data
        nb_val_samples=nb_validation_samples, callbacks = [earlyStopping])
    
    # Saving the loss and acc during the traning time.
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('hist-18x36v2.csv', encoding='utf-8', index=False)
    model.save('weights-18x36v2.h5')
    plotHistory(hist)

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
