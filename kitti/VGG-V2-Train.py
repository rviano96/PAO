
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import os

import matplotlib.patches as patches


# In[2]:


#########################################################################################################
#VALUES

base_path = '/media/rodrigo/Rodrigo/PAO/VGG-BdBxs/'
train_data_dir = base_path + 'data/train'
validation_data_dir = base_path  + 'data/validation'
nb_train_samples = 7481 #1779 pedestrian - 5702 notPedestrian
nb_validation_samples = 748 #180 pedestrian/ 568 notPedestrian
nb_epoch = 50

#dimensions of our images.
img = Image.open(train_data_dir + '/Pedestrian/' + "000000.png")
img_width, img_height= img.size
print(img_width, " ", img_height, " ", img.mode)
##image resizing
img_width, img_height = 150, 150


# In[3]:


#########################################################################################################

#MODEL
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#Save the model
model_json = model.to_json()
with open("modelVggV2-Kitti.json", "w") as json_file:
    print("Saving model into: ",os.getcwd() + "/modelVggV2-Kitti.json" )
    json_file.write(model_json)


# In[4]:


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

#PREPARE TRAINING DATA
train_generator = train_datagen.flow_from_directory(
        train_data_dir, #data/train
        target_size=(img_width, img_height),  #RESIZE to 150/150
        batch_size=32,
        class_mode='binary')  #since we are using binarycrosentropy need binary labels

#PREPARE VALIDATION DATA
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,  #data/validation
        target_size=(img_width, img_height), #RESIZE 150/150
        batch_size=32,
        class_mode='binary')
labels = (train_generator.class_indices)
print(labels)


# In[ ]:


#START model.fit
history =model.fit_generator(
        train_generator, #train data
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,  #validation data
        nb_val_samples=nb_validation_samples)

print("saving weights to: " , os.getcwd() +"/savedweightsvgg-Kitti.h5" )
model.save_weights('savedweightsvgg-Kitti.h5')
# list all data in history
print(history.history.keys())


# In[71]:


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

