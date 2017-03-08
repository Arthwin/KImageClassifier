import keras
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
import os.path

# step 1 collect data
img_width, img_height = 150, 150
train_data_dir = 'data/training'
validation_data_dir = 'data/validation'

#rescale pixel values from [0,255] to [0,1]
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    batch_size=16,
    class_mode='binary')

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width,img_height),
    batch_size=32,
    class_mode='binary')

# step 2 build model, either sequential or graph
model = Sequential()
model.add(Convolution2D(32,3,3,input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))##fully connected
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))#output layer 2, cats dogs
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


#step 3 train model

print('Do you want to load weights?')
load = input('y/n, n will train network')
if load == 'n':
    model.fit_generator(train_generator,
                        samples_per_epoch=2048,
                        nb_epoch=30,
                        validation_data=validation_generator,
                        nb_val_samples=832)
    model.save_weights('models/simple_CNN.h5')
else:
    model.load_weights('simple_CNN.h5')


#step 4 test model
def prediction(path):
    img = image.img_to_array(image.load_img(path,target_size=(img_width,img_height)))
    img = img.reshape( (1,) + img.shape )
    predic = model.predict(img)[0][0]
    return 'dog' if predic == 1 else 'cat'

testing = True
while testing:
        num = input('img number (x to quit): ')
        if num != 'x':
            try:
               val = int(num)
               dir = 'data/test/' + num + '.jpg'
               if os.path.isfile(dir):
                    print('I think its a ' + prediction(dir) + '!')
               else:
                    print('No file named: ' + dir)
                    print('Put img on data/test/ with a filename format <number>.jpg')
            except ValueError:
                print("That's not an int!")
        else:
            testing = False




                      