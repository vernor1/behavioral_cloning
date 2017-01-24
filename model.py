import cv2
import numpy as np
import os.path
import pandas as pd
import re

from image_processing import get_processed_image_shape, process_image
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from tqdm import tqdm


DRIVING_LOG_FILE = 'driving_log.csv'
MODEL_FILE = 'model.json'
WEIGHTS_FILE = 'model.h5'
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-4
LAMBDA = 1e-5
KEEP_PROB = 0.5
SIDE_CAMERA_STEERING = 0.25


def mirror_image(in_img):
    mirror_img = np.zeros_like(in_img)
    mirror_img = cv2.flip(in_img, 1)
    return mirror_img

def save_image(in_img, img_name):
    out_img = cv2.cvtColor(in_img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(img_name, out_img)

def predict_steering_angle(model, img_name):
    img = cv2.imread(img_name)
    transformed_image_array = process_image(img)[None, :, :, :]
    return float(model.predict(transformed_image_array, batch_size=1))

def create_training_set():
    # Load training set from the driving log
    driving_log = pd.read_csv(DRIVING_LOG_FILE)

    # Get number of log rows
    n_rows = len(driving_log.index)
    assert(n_rows > 0)

    # Define number of images as the number of rows multiplied by 2 (mirrored) and 3 (side cameras)
    n_images = n_rows * 2 * 3

    # Get sample image to find out its final dimensions
    assert(len(driving_log.ix[0]) > 0)
    img = cv2.imread(re.search('IMG\/.+', driving_log.ix[0][0]).group())

    # Create empty training set
    X_train = np.empty(shape=((n_images,) + get_processed_image_shape(img)), dtype='uint8')
    y_train = np.empty(shape=(n_images,), dtype='float32')
    print("Training set shape %s" % (X_train.shape,))

    # Load images
    pbar_rows = tqdm(range(n_rows), unit=' rows')
    idx = 0
    for pbar_row, (log_idx, row) in zip(pbar_rows, driving_log.iterrows()):
        assert(row.shape[0] > 3)
        angle = row[3]
        # Center
        img = process_image(cv2.imread(re.search('IMG\/.+', row[0]).group()))
        X_train[idx] = img
        y_train[idx] = angle
        idx += 1
        # Center mirror
        X_train[idx] = mirror_image(img)
        y_train[idx] = -angle
        idx += 1
        # Left
        img = process_image(cv2.imread(re.search('IMG\/.+', row[1]).group()))
        X_train[idx] = img
        y_train[idx] = angle + SIDE_CAMERA_STEERING
        idx += 1
        # Left mirror
        X_train[idx] = mirror_image(img)
        y_train[idx] = -(angle + SIDE_CAMERA_STEERING)
        idx += 1
        # Right
        img = process_image(cv2.imread(re.search('IMG\/.+', row[2]).group()))
        X_train[idx] = img
        y_train[idx] = angle - SIDE_CAMERA_STEERING
        idx += 1
        # Right mirror
        X_train[idx] = mirror_image(img)
        y_train[idx] = -(angle - SIDE_CAMERA_STEERING)
        idx += 1
#        print("Center %f %f, Left %f %f, Right %f %f" % (y_train[idx-6], y_train[idx-5], y_train[idx-4], y_train[idx-3], y_train[idx-2], y_train[idx-1]))
#        print("Center %f %f, Left %f, Right %f" % (y_train[idx-4], y_train[idx-3], y_train[idx-2], y_train[idx-1]))

    return X_train, y_train

def create_model():
    # Model definition
    activation_function = 'tanh'
    model = Sequential()
    #model.add(Convolution2D(24,
    #                        5,5,
    #                        subsample=(2,2),
    #                        border_mode='valid',
    #                        input_shape=X_train.shape[1:],
    #                        activation=activation_function))
    model.add(Convolution2D(24,
                            5,5,
                            subsample=(2,2),
                            border_mode='valid',
                            input_shape=X_train.shape[1:],
                            activation=activation_function,
                            W_regularizer=l2(LAMBDA)))
    print("Input shape %s" % (model.layers[-1].input_shape,))
    print("Conv. layer 1 %s" % (model.layers[-1].output_shape,))
    #model.add(Convolution2D(36,
    #                        5,5,
    #                        subsample=(2,2),
    #                        border_mode='valid',
    #                        activation=activation_function))
    model.add(Convolution2D(36,
                            5,5,
                            subsample=(2,2),
                            border_mode='valid',
                            activation=activation_function,
                            W_regularizer=l2(LAMBDA)))
    print("Conv. layer 2 %s" % (model.layers[-1].output_shape,))
    #model.add(Convolution2D(48,
    #                        5,5,
    #                        subsample=(2,2),
    #                        border_mode='valid',
    #                        activation=activation_function))
    model.add(Convolution2D(48,
                            5,5,
                            subsample=(2,2),
                            border_mode='valid',
                            activation=activation_function,
                            W_regularizer=l2(LAMBDA)))
    print("Conv. layer 3 %s" % (model.layers[-1].output_shape,))
    #model.add(Convolution2D(64,
    #                        3,3,
    #                        border_mode='valid',
    #                        activation=activation_function))
    model.add(Convolution2D(64,
                            3,3,
                            border_mode='valid',
                            activation=activation_function,
                            W_regularizer=l2(LAMBDA)))
    print("Conv. layer 4 %s" % (model.layers[-1].output_shape,))
    #model.add(Convolution2D(64,
    #                        3,3,
    #                        border_mode='valid',
    #                        activation=activation_function))
    model.add(Convolution2D(64,
                            3,3,
                            border_mode='valid',
                            activation=activation_function,
                            W_regularizer=l2(LAMBDA)))
    print("Conv. layer 5 %s" % (model.layers[-1].output_shape,))
    model.add(Flatten())
    print("Flatten %s" % (model.layers[-1].output_shape,))
    #model.add(Dense(100,
    #                activation=activation_function))
    model.add(Dense(100,
                    activation=activation_function,
                    W_regularizer=l2(LAMBDA)))
    model.add(Dropout(KEEP_PROB))
    print("Fully-connected layer 1 %s" % (model.layers[-1].output_shape,))
    #model.add(Dense(50,
    #                activation=activation_function))
    model.add(Dense(50,
                    activation=activation_function,
                    W_regularizer=l2(LAMBDA)))
    model.add(Dropout(KEEP_PROB))
    print("Fully-connected layer 2 %s" % (model.layers[-1].output_shape,))
    #model.add(Dense(10,
    #                activation=activation_function))
    model.add(Dense(10,
                    activation=activation_function,
                    W_regularizer=l2(LAMBDA)))
    model.add(Dropout(KEEP_PROB))
    print("Fully-connected layer 3 %s" % (model.layers[-1].output_shape,))
    #model.add(Dense(1))
    model.add(Dense(1,
                    W_regularizer=l2(LAMBDA)))
    print("Output %s" % (model.layers[-1].output_shape,))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(adam, 'mse')
    return model

def sanity_check(model):
    print("Emergency steering right %f" % (predict_steering_angle(model, 'steer_right.jpg')))
    print("Emergency steering left %f" % (predict_steering_angle(model, 'steer_left.jpg')))
    test_file_names = ['center_2017_01_21_17_38_31_549_left.jpg',
                       'center_2017_01_21_17_38_54_655_straight.jpg',
                       'center_2017_01_21_17_38_16_238_right.jpg']
    for test_file_name in test_file_names:
        print("Testing %s: %f" % (test_file_name, predict_steering_angle(model, test_file_name)))

def save_model(model):
    print("Saving model and weights")
    model.save_weights(WEIGHTS_FILE)
    json_model = model.to_json()
    with open(MODEL_FILE, 'w') as f:
        f.write(json_model)


X_train, y_train = create_training_set()
save_image(X_train[1200], "test0_%.2f.jpg" % (y_train[1200]))
save_image(X_train[1201], "test1_%.2f.jpg" % (y_train[1201]))
save_image(X_train[1202], "test2_%.2f.jpg" % (y_train[1202]))
save_image(X_train[1203], "test3_%.2f.jpg" % (y_train[1203]))
save_image(X_train[1204], "test4_%.2f.jpg" % (y_train[1204]))
save_image(X_train[1205], "test5_%.2f.jpg" % (y_train[1205]))

model = create_model()

# Load weights if exist
if os.path.isfile(WEIGHTS_FILE):
    print("Loading existing weights")
    model.load_weights(WEIGHTS_FILE)

# Train model
model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS)

sanity_check(model)

save_model(model)
