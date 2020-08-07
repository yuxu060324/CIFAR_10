from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

# CIFAR_10 is a set of 60K images 32x32 pixels on 3 channels
IMG_CHANNELS = 3    # Because the image can be divided into three colors: red, blue and green.
IMG_ROWS = 32   # The number of vertical pixels in an image
IMG_COLS = 32   # The number of horizontal pixels in an image

# constant
BATCH_SIZE = 128    # Number of data per study
NB_EPOCH = 40   # Number of times to learn
NB_CLASSES = 10     # Number of outputs (classifications)
VERBOSE = 1     # redundancy
VALIDATION_SPLIT = 0.2  # Percentage of training data used as validation data
OPTIM = RMSprop()   # Methods for optimization

# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()  # Reading CIFAR-10 data
print('X_train shape:', X_train.shape)  # The shape of the training data matrix
print(X_train.shape[0], 'train samples')    # Number of training data
print(X_test.shape[0], 'test samples')  # Number of test data

# convert to categorical
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)  # Convert target to vector
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)    # Convert target to vector

# float and normalization
X_train = X_train.astype('float32')     # Convert to float32
X_test = X_test.astype('float32')   # Convert to float32
X_train /= 255  # I made it less than 1 floating point to make it easier to calculate.
X_test /= 255   # I made it less than 1 floating point to make it easier to calculate.

# network

model = Sequential()    # Creating a Model

model.add(Conv2D(32, kernel_size=3, padding='same',
                 input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))   # Image Convolution
model.add(Activation('relu'))   # Introduction of an activation function
model.add(Conv2D(32, kernel_size=3, padding='same'))    # Image Convolution
model.add(Activation('relu'))   # Introduction of an activation function
model.add(MaxPooling2D(pool_size=(2, 2)))   # Max data pooling operation for 2D data
model.add(Dropout(0.25))    # Setting up a neuron dropout

model.add(Conv2D(64, kernel_size=3, padding='same'))    # Image Convolution
model.add(Activation('relu'))   # Introduction of an activation function
model.add(Conv2D(64, 3, 3))     # Image Convolution
model.add(Activation('relu'))   # Introduction of an activation function
model.add(MaxPooling2D(pool_size=(2, 2)))   # Max data pooling operation for 2D data
model.add(Dropout(0.25))    # Setting up a neuron dropout

model.add(Flatten())    # Flatten the input
model.add(Dense(512))   # Adding a Neural Network Layer
model.add(Activation('relu'))   # Introduction of an activation function for the middle layer
model.add(Dropout(0.5))     # Setting up a middle layer dropout
model.add(Dense(NB_CLASSES))    # Organizing data into the output layer
model.add(Activation('softmax'))    # Introduction of the activation function of the output layer

model.summary()     # Displaying a summary of the model shape

model.compile(loss='categorical_crossentropy', optimizer=OPTIM,
              metrics=['accuracy'])     # Setting up a model for learning

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)    # Expand data and generate batches

# train

history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,
                    verbose=VERBOSE)    # Training the Model

# model.fit_generator(datagen.flow(X_train, Y_train,
#                        batch_size=BATCH_SIZE),
#                        samples_per_epoch=X_train.shape[0],
#                        nb_epoch=NB_EPOCH,
#                        verbose=VERBOSE)

# server.launch(model)


print('Testing...')
score = model.evaluate(X_test, Y_test,
                       batch_size=BATCH_SIZE, verbose=VERBOSE)  # Scores on test data
print("\nTest score:", score[0])    # Display in scores in test data
print('Test accuracy:', score[1])   # Display of accuracy in test data

# save model
model_json = model.to_json()    # Saving a model
open('cifar10_architecture.json', 'w').write(model_json)    # Saving a model
model.save_weights('cifar10_weights.h5', overwrite=True)    # Storing the learned parameters

# # list all data in history
# print(history.history.keys())   #
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()