import cv2
import multiprocessing
import numpy as np
import os.path
import pandas as pd
import re

from image_processing import get_processed_image_shape, crop_hood, crop_sky, resize_image, normalize_image, process_image
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from tqdm import tqdm


DRIVING_LOG_FILE = 'driving_log.csv'
MODEL_FILE = 'model.json'
WEIGHTS_FILE = 'model.h5'
BATCH_SIZE = 256
EPOCHS = 15
SAMPLES_PER_EPOCH = 102400
ACTIVATION_FUNCTION = 'relu'
LEARNING_RATE = 5e-4
LAMBDA = 1e-5
STRAIGHT_STEERING_ANGLE = 0.1          # Angle units
SIDE_CAMERA_ANGLE = 0.25               # Angle units
MAX_ANGLE_SHIFT = 0.25                 # Angle units
IMAGES_PER_FULL_STEERING_ANGLE = 0.5   # Factor of horizontal picture size. It's observed that the pictures from the left and right cameras
                                       # have and offset of about 40px from the center picture. People reported that the angle offset
                                       # of the side cameras worked good at 0.25. So the full angle of 1.0 would shift image for about
                                       # 40/0.25=160, which is 160/320=0.5 images per full steering angle.
MAX_VERTICAL_SHIFT = 0.2               # Factor of vertical picture size
BRIGHTNESS_SHIFT_LIMITS = (0.25, 1.50) # Factor of original brightness


def shift_image(in_img, angle):
    """ Randomly shifts an image in horizontal and vertical directions.

    param: in_img: the original image
    returns: the randomly changed image
    """
    angle_shift = np.random.uniform(-MAX_ANGLE_SHIFT, MAX_ANGLE_SHIFT)
    shift_x = angle_shift * IMAGES_PER_FULL_STEERING_ANGLE * in_img.shape[1]
    shift_y = np.random.randint(-MAX_VERTICAL_SHIFT, MAX_VERTICAL_SHIFT)
    transformation_matrix = np.float32([[1,0,shift_x],[0,1,shift_y]])
    out_img = cv2.warpAffine(in_img,
                             transformation_matrix,
                             (in_img.shape[1], in_img.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)
    return out_img, angle + angle_shift

def shift_brightness(in_img):
    """ Randomly shifts brightness of an image.

    param: in_img: the original image
    returns: the randomly changed image
    """
    img = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)
    brightness_multiplier = np.random.uniform(BRIGHTNESS_SHIFT_LIMITS[0], BRIGHTNESS_SHIFT_LIMITS[1])
    img[:,:,2] = img[:,:,2].clip(0, 255 // brightness_multiplier) * brightness_multiplier
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def generate_single_sample(row):
    """ Generated a single sample from a driving log entry.

    param: row: the Pandas row
    returns: the sample image and the corresponding steering angle
    """
    camera_selector = np.random.randint(3)
    angle = row[3]
    if camera_selector == 1:
        angle += SIDE_CAMERA_ANGLE
    elif camera_selector == 2:
        angle -= SIDE_CAMERA_ANGLE
    img = cv2.imread(re.search('IMG\/.+', row[camera_selector]).group())
    # Crop the car hood before shifting the image to prevent it appearing on the picture
    img = crop_hood(img)
    img, angle = shift_image(img, angle)
    # Crop the sky after shifting the image to minimize generating the sky in case of shifting the picture down
    img = crop_sky(img)
    img = shift_brightness(img)
    if np.random.randint(2) != 0:
        img = cv2.flip(img, 1)
        angle = -angle
    img = resize_image(img)
    img = normalize_image(img)
    return img, angle

def generate_samples(driving_log_file, batch_size=32, discard_straight_sample_prob=0):
    """ Generates samples from driving log.

    param: driving_log_file: the file name of the driving log
    param: batch_size: the batch size
    param: discard_straight_sample_prob: the probability of discarding a straight driving sample, if generated
    returns: the tuple of the sample images and the corresponding steering angles
    """
    driving_log = pd.read_csv(driving_log_file, header=None)
    X = np.empty(shape=((batch_size,) + get_processed_image_shape()), dtype='uint8')
    Y = np.empty(shape=(batch_size,), dtype='float32')
    while True:
        for idx in range(batch_size):
            row = driving_log.ix[np.random.randint(len(driving_log))]
            is_sample_acceptable = None
            while not is_sample_acceptable:
                x, y = generate_single_sample(row)
                if abs(y) > STRAIGHT_STEERING_ANGLE or np.random.random() > discard_straight_sample_prob:
                    is_sample_acceptable = True
            X[idx] = x
            Y[idx] = y
        yield X, Y

def create_model(input_shape):
    """ Creates a new model.

    param: input_shape: the shape of the input
    returns: the model
    """
    model = Sequential()
    model.add(Convolution2D(24,
                            5,5,
                            subsample=(2,2),
                            border_mode='valid',
                            input_shape=input_shape,
                            activation=ACTIVATION_FUNCTION,
                            W_regularizer=l2(LAMBDA)))
    model.add(Convolution2D(36,
                            5,5,
                            subsample=(2,2),
                            border_mode='valid',
                            activation=ACTIVATION_FUNCTION,
                            W_regularizer=l2(LAMBDA)))
    model.add(Convolution2D(48,
                            5,5,
                            subsample=(2,2),
                            border_mode='valid',
                            activation=ACTIVATION_FUNCTION,
                            W_regularizer=l2(LAMBDA)))
    model.add(Convolution2D(64,
                            3,3,
                            border_mode='valid',
                            activation=ACTIVATION_FUNCTION,
                            W_regularizer=l2(LAMBDA)))
    model.add(Convolution2D(64,
                            3,3,
                            border_mode='valid',
                            activation=ACTIVATION_FUNCTION,
                            W_regularizer=l2(LAMBDA)))
    model.add(Flatten())
    model.add(Dense(100,
                    activation=ACTIVATION_FUNCTION,
                    W_regularizer=l2(LAMBDA)))
    model.add(Dropout(0.5))
    model.add(Dense(50,
                    activation=ACTIVATION_FUNCTION,
                    W_regularizer=l2(LAMBDA)))
    model.add(Dropout(0.5))
    model.add(Dense(10,
                    activation=ACTIVATION_FUNCTION,
                    W_regularizer=l2(LAMBDA)))
    model.add(Dropout(0.5))
    model.add(Dense(1,
                    W_regularizer=l2(LAMBDA)))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(adam, 'mse')
    model.summary()
    return model

def train_model(model, driving_log_file):
    """ Trains a model using randomly generated jittered images of a driving log.

    param: model: the model to train
    param: driving_log_file: the file name of the driving log
    """
    validation_generator = generate_samples(driving_log_file, batch_size=BATCH_SIZE)
    for epoch in range(EPOCHS):
        discard_prob = 1-epoch/(EPOCHS-1)
        print("Epoch %d/%d, probability of discarding straight driving samples %.2f" % (epoch+1, EPOCHS, discard_prob))
        training_generator = generate_samples(DRIVING_LOG_FILE,
                                              batch_size=BATCH_SIZE,
                                              discard_straight_sample_prob=discard_prob)
        model.fit_generator(training_generator,
                            samples_per_epoch=SAMPLES_PER_EPOCH,
                            nb_epoch=1,
                            validation_data=validation_generator,
                            nb_val_samples=SAMPLES_PER_EPOCH // 5,
                            nb_worker=multiprocessing.cpu_count(),
                            pickle_safe=True)

def load_center_samples(driving_log_file):
    """ Loads and processes center images from a driving log.

    param: driving_log_file: the file name of the driving log
    returns: the tuple of the sample images and the corresponding steering angles
    """
    driving_log = pd.read_csv(driving_log_file, header=None)
    n_rows = len(driving_log)
    X = np.empty(shape=((n_rows,) + get_processed_image_shape()), dtype='uint8')
    Y = np.empty(shape=(n_rows,), dtype='float32')
    pbar_rows = tqdm(range(len(driving_log)), unit=' samples')
    for pbar_row, (idx, row) in zip(pbar_rows, driving_log.iterrows()):
        img = cv2.imread(re.search('IMG\/.+', row[0]).group())
        img = process_image(img)
        X[idx] = img
        Y[idx] = row[3]
    return X, Y

def test_model(model, driving_log_file):
    """ Tests a model using the original center images of a driving log.

    param: model: the model to test
    param: driving_log_file: the file name of the driving log
    """
    print("Loading test tamples")
    X_test, y_test = load_center_samples(driving_log_file)
    print("Testing model")
    test_score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print("Test loss %.4f" % (test_score))

def predict_steering_angle(model, img_name):
    """ Uses a model to predict the steering angle of a given image.

    param: model: the model to check
    param: img_name: the image file name
    """
    img = cv2.imread(img_name)
    transformed_image_array = process_image(img)[None, :, :, :]
    return float(model.predict(transformed_image_array, batch_size=1))

def sanity_check_model(model):
    """ Performs basic sanity checks of a model by predicting a left, center, and right images.
        The left image was captured at the steering angle of -0.25, center 0.00 and right +0.25.

    param: model: the model to check
    """
    print("Sanity checks:")
    test_file_names = ['left.jpg',
                       'straight.jpg',
                       'right.jpg']
    for test_file_name in test_file_names:
        print("    %s: %.2f" % (test_file_name, predict_steering_angle(model, test_file_name)))

def save_model(model, model_file, weights_file):
    """ Saves a model and its weights.

    param: model: the model to save
    param: model_file: the model file name
    param: weights_file: the model weights file
    """
    print("Saving model and weights")
    model.save_weights(weights_file)
    json_model = model.to_json()
    with open(model_file, 'w') as f:
        f.write(json_model)


model = create_model(get_processed_image_shape())

# Load weights if exist
if os.path.isfile(WEIGHTS_FILE):
    print("ATTENTION: loading existing weights")
    model.load_weights(WEIGHTS_FILE)

train_model(model, DRIVING_LOG_FILE)

sanity_check_model(model)

test_model(model, DRIVING_LOG_FILE)

save_model(model, MODEL_FILE, WEIGHTS_FILE)
