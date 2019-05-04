# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 20:41:35 2017

@author: olaf
"""

import warnings
import concurrent.futures
import numpy as np
from scipy import ndimage as scimg
import skimage.exposure

TRACK_FOLDER = "track[12]"
DATA_FOLDER = "*"
LEFT_RIGHT = True
ANGLE_ADJM = 0.2
EQUALIZE = False
FLIP_IMG = True
CROP_TOP = 60
CROP_BOTTOM = 20
EPOCHS = 3
BATCH_SIZE = 256
INITIALIZATION = "glorot_uniform"
REGULARIZATION = None #("l2", 0.001)
BATCH_NORM = False
ACTIVATION = 'elu'
DROP_RATE = 0.0
PLOT_LOSS = True

def get_image(batch_sample):
    image = scimg.imread(batch_sample[0])
    if EQUALIZE:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = skimage.exposure.equalize_adapthist(image)
    angle = batch_sample[1]
    # Data augmentation
    if FLIP_IMG:
        flip = np.random.choice(range(2))
        if flip:
            image = np.fliplr(image)
            angle = -angle
    return (image, angle)


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            futures = set()
            with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
                for batch_sample in batch_samples:
                    future = executor.submit(get_image, batch_sample)
                    futures.add(future)
                try:
                    for future in concurrent.futures.as_completed(futures):
                        err = future.exception()
                        if err is None:
                            image, angle = future.result()
                            images.append(image)
                            angles.append(angle)
                        else:
                            raise err
                except KeyboardInterrupt:
                    for future in futures:
                        future.cancel()
                    executor.shutdown()
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

            
if __name__ == "__main__":
    import os.path
    import csv
    import glob
    import matplotlib.pyplot as plt
    import keras.backend as K
    #from keras.preprocessing.image import ImageDataGenerator
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Cropping2D, Activation
    from keras.layers.convolutional import Convolution2D
    if DROP_RATE > 0.0:
        from keras.layers import Dropout
    if BATCH_NORM:
        from keras.layers.normalization import BatchNormalization
    if REGULARIZATION is None:
        regularizer = None
    else:
        from keras.regularizers import l2
        regularizer = l2(REGULARIZATION[1])
    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle

    with open("training_data.txt", 'w') as td:
        td.write("Augment with left & right camera data: {}\n".format(LEFT_RIGHT))
        td.write("Angle adjustment (absolute value): {}\n".format(ANGLE_ADJM))
        td.write("Equalize images using adapthist: {}\n".format(EQUALIZE))
        td.write("Flip images randomly: {}\n".format(FLIP_IMG))
        td.write("Number of top lines cropped: {}\n".format(CROP_TOP))
        td.write("Number of bottom lines cropped: {}\n".format(CROP_BOTTOM))
        td.write("Epochs: {}\n".format(EPOCHS))
        td.write("Batch size: {}\n".format(BATCH_SIZE))
        td.write("Weights initialization: {}\n".format(INITIALIZATION))
        td.write("Weights regularization: {}\n".format(REGULARIZATION))
        td.write("Activation function: {}\n".format(ACTIVATION))
        td.write("Batch normalization: {}\n".format(BATCH_NORM))
        td.write("Droprate: {}\n".format(DROP_RATE))

        samples = []
        logfiles = glob.glob(os.path.join(TRACK_FOLDER, DATA_FOLDER, "driving_log.csv"))
        #logfiles = glob.glob(os.path.join(DATA_FOLDER, "driving_log.csv"))
        for logfile in logfiles:
            print("Reading:", logfile)
            td.write("Reading: {}\n".format(logfile))
            basepath = os.path.dirname(logfile)
            with open(logfile) as csvfile:
                reader = csv.reader(csvfile)
                # skip header
                next(reader)
                for line in reader:
                    angle = float(line[3])
                    img_path = os.path.join(basepath, "IMG", 
                                            os.path.basename(line[0]))
                    samples.append((img_path, angle))
                    if LEFT_RIGHT:
                        img_path = os.path.join(basepath, "IMG", 
                                                os.path.basename(line[1]))
                        samples.append((img_path, angle + ANGLE_ADJM))
                        img_path = os.path.join(basepath, "IMG", 
                                                os.path.basename(line[2]))
                        samples.append((img_path, angle - ANGLE_ADJM))

        print(len(samples), "samples found")
        td.write("{} samples found\n".format(len(samples)))

    train_samples, valid_samples = train_test_split(samples, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    valid_generator = generator(valid_samples, batch_size=BATCH_SIZE)
    
    input_shape = (160, 320, 3)

    # Define model
    model = Sequential()
    # Cropping
    model.add(Cropping2D(cropping=((CROP_TOP, CROP_BOTTOM), (0, 0)), 
                         input_shape=input_shape))
    model.add(Lambda(lambda x: K.tf.image.resize_images(x, (80, 240))))
    # convert to YUV
    #model.add(Lambda(lambda x: K.dot(x, rgb2yuv_t)))
    # Preprocess incoming data, centered around zero with small standard deviation 
    # Normalization
    #model.add(Lambda(lambda x: (x - X_mean) / X_std))
    if EQUALIZE:
        model.add(Lambda(lambda x: x/0.5 - 1.))
    else:
        model.add(Lambda(lambda x: x/127.5 - 1.))
    # DL network
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', 
                            init=INITIALIZATION, W_regularizer=regularizer))
    if BATCH_NORM:
        model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))
    if DROP_RATE > 0.0:
        model.add(Dropout(DROP_RATE))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', 
                            init=INITIALIZATION, W_regularizer=regularizer))
    if BATCH_NORM:
        model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))
    if DROP_RATE > 0.0:
        model.add(Dropout(DROP_RATE))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', 
                            init=INITIALIZATION, W_regularizer=regularizer))
    if BATCH_NORM:
        model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))
    if DROP_RATE > 0.0:
        model.add(Dropout(DROP_RATE))
    model.add(Convolution2D(48, 3, 3, subsample=(1, 1), border_mode='valid', 
                            init=INITIALIZATION, W_regularizer=regularizer))
    if BATCH_NORM:
        model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))
    if DROP_RATE > 0.0:
        model.add(Dropout(DROP_RATE))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', 
                            init=INITIALIZATION, W_regularizer=regularizer))
    if BATCH_NORM:
        model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))
    if DROP_RATE > 0.0:
        model.add(Dropout(DROP_RATE))
    model.add(Flatten())
    model.add(Dense(100, init=INITIALIZATION, W_regularizer=regularizer))
    if BATCH_NORM:
        model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))
    if DROP_RATE > 0.0:
        model.add(Dropout(DROP_RATE))
    model.add(Dense(50, init=INITIALIZATION, W_regularizer=regularizer))
    if BATCH_NORM:
        model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))
    if DROP_RATE > 0.0:
        model.add(Dropout(DROP_RATE))
    model.add(Dense(10, init=INITIALIZATION, W_regularizer=regularizer))
    if BATCH_NORM:
        model.add(BatchNormalization())
    model.add(Activation(ACTIVATION))
    if DROP_RATE > 0.0:
        model.add(Dropout(DROP_RATE))
    model.add(Dense(1, init=INITIALIZATION, W_regularizer=regularizer))
    # Optimization
    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, 
                                         samples_per_epoch=len(train_samples), 
                                         validation_data=valid_generator, 
                                         nb_val_samples=len(valid_samples), 
                                         nb_epoch=EPOCHS, verbose=1)
    # Save model
    model.save("model.h5")
    
    # print the keys contained in the history object
    print(history_object.history.keys())
    
    if PLOT_LOSS:
        # plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        #plt.show()
        plt.savefig('loss.png', dpi=150)
        plt.close()
