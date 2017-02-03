import cv2
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os.path
import pandas as pd
import re
import sys
import urllib.request
import zipfile

from image_processing import get_processed_image_shape, crop_hood, crop_sky, resize_image, normalize_image, process_image
from keras.layers import Dense, Dropout, Flatten, Input, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

# Global constants
SAMPLE_DATA_URL = 'https://s3.amazonaws.com/vernor-carnd/behavioral_cloning_data.zip'
DRIVING_LOG_FILE = 'driving_log.csv'
MODEL_FILE = 'model.json'
WEIGHTS_FILE = 'model.h5'
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1024
EPOCHS = 20
SAMPLES_PER_EPOCH = 12800
ACTIVATION_FUNCTION = 'relu'
LEARNING_RATE = 1e-4
LAMBDA = 1e-3
DISCARD_STRAIGHT_SAMPLE_PROB = 0.8    # Probability of discarding straight driving samples taked from the center camera
STRAIGHT_STEERING_ANGLE = 0.1         # Angle units
SIDE_CAMERA_ANGLE = 0.25               # Angle units
MAX_ANGLE_SHIFT = 0.1                  # Angle units
IMAGES_PER_FULL_STEERING_ANGLE = 0.5   # Factor of horizontal picture size. It's observed that the pictures from the left and right cameras
                                       # have and offset of about 40px from the center picture. People reported that the angle offset
                                       # of the side cameras worked good at 0.25. So the full angle of 1.0 would shift image for about
                                       # 40/0.25=160, which is 160/320=0.5 images per full steering angle.
MAX_VERTICAL_SHIFT = 0.2               # Factor of vertical picture size
BRIGHTNESS_SHIFT_LIMITS = (0.25, 1.25) # Factor of original brightness
DEBUG_IMAGES_ENABLED = True            # If the boolean constant is set to True, the supplementary debug images are dumped

# Global variables
debug_images_dumped = True


def report_download_progress(block_nr, block_size, size):
    """ Displays download progress.

    param: block_nr: the block number
    param: block_size: the block size
    param: size: the content size
    """
    progress = block_nr * block_size
    sys.stdout.write("\r%.2f" % (100.0 * progress/size))

def download_sample_data(sample_data_url):
    """ Downloads sample data and unpacks it.

    param: sample_data_url: the sample data URL
    """
    file_name = sample_data_url.split('/')[-1]
    if not os.path.isfile(file_name):
        print("Downloading sample data file %s:" % (file_name))
        urllib.request.urlretrieve(sample_data_url, file_name, report_download_progress)
    print("\nUnpacking sample data")
    with zipfile.ZipFile(file_name, "r") as zip_handle:
        zip_handle.extractall(".")

def save_histogram(angles, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Steering Angle')
    ax.set_ylabel('# of Samples')
    n, bins, patches = plt.hist(angles, bins=40, range=(-1., 1.))
    fig.savefig(file_name)

def prepare_training_set(driving_log_file):
    driving_log = pd.read_csv(driving_log_file, header=None)
    if DEBUG_IMAGES_ENABLED:
        save_histogram(driving_log[3], "original_distribution.svg")
    X_train = []
    y_train = []

    pbar_rows = tqdm(range(len(driving_log)), desc='Processing driving log', unit=' samples')
    for pbar_row, (idx, row) in zip(pbar_rows, driving_log.iterrows()):
        center_img_path = re.search('IMG\/.+', row[0]).group()
        left_img_path = re.search('IMG\/.+', row[1]).group()
        right_img_path = re.search('IMG\/.+', row[2]).group()
        angle = row[3]
        if abs(angle) < STRAIGHT_STEERING_ANGLE:
            if np.random.random() > DISCARD_STRAIGHT_SAMPLE_PROB:
                # If driving straight, use only a fraction of center camera images for training
                X_train.append(center_img_path)
                y_train.append(angle)
                if angle < 0:
                    # Left steering: use right camera, adjust steering angle to the left
                    X_train.append(right_img_path)
                    y_train.append(angle - SIDE_CAMERA_ANGLE)
                elif angle > 0:
                    # Right steering: use left camera, ajust steering to the right
                    X_train.append(left_img_path)
                    y_train.append(angle + SIDE_CAMERA_ANGLE)
        else:
            # If the sample is a steering one, always use the center camera image for training
            # as well as the image from the camera facing the outer shoulder
            X_train.append(center_img_path)
            y_train.append(angle)
            if angle < 0:
                # Left steering: use right camera, adjust steering angle to the left
                X_train.append(right_img_path)
                y_train.append(angle - SIDE_CAMERA_ANGLE)
            else:
                # Right steering: use left camera, ajust steering to the right
                X_train.append(left_img_path)
                y_train.append(angle + SIDE_CAMERA_ANGLE)

    pbar_samples = tqdm(range(len(y_train)), desc='Generating flips and distortions', unit=' samples')
    for idx in pbar_samples:
        angle_shift = np.random.uniform(-MAX_ANGLE_SHIFT, MAX_ANGLE_SHIFT)
        is_flipped = np.random.randint(2)
        if not is_flipped:
            y_train[idx] = y_train[idx]+angle_shift
        else:
            y_train[idx] = -(y_train[idx]+angle_shift)
        # Transform X_train by replacing the path values with tuples (path, is_flipped, angle_shift)
        X_train[idx] = (X_train[idx], is_flipped, angle_shift)

    if DEBUG_IMAGES_ENABLED:
        save_histogram(y_train, "balanced_distribution.svg")

    return X_train, y_train

def shift_image(in_img, angle_shift):
    """ Randomly shifts an image in horizontal and vertical directions.

    param: in_img: the original image
    returns: the randomly changed image
    """
    shift_x = angle_shift * IMAGES_PER_FULL_STEERING_ANGLE * in_img.shape[1]
    shift_y = np.random.randint(-MAX_VERTICAL_SHIFT, MAX_VERTICAL_SHIFT)
    transformation_matrix = np.float32([[1,0,shift_x],[0,1,shift_y]])
    out_img = cv2.warpAffine(in_img,
                             transformation_matrix,
                             (in_img.shape[1], in_img.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)
    return out_img

def shift_brightness(in_img):
    """ Randomly shifts brightness of an image.

    param: in_img: the original image
    returns: the randomly changed image
    """
    img = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)
    brightness_multiplier = np.random.uniform(BRIGHTNESS_SHIFT_LIMITS[0], BRIGHTNESS_SHIFT_LIMITS[1])
    img[:,:,2] = img[:,:,2].clip(0, 255 // brightness_multiplier) * brightness_multiplier
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def generate_samples(X, y, batch_size=32):
    """ Generates samples from driving log.

    param: driving_log_file: the file name of the driving log
    param: batch_size: the batch size
    param: discard_straight_sample_prob: the probability of discarding a straight driving sample, if generated
    returns: the tuple of the sample images and the corresponding steering angles
    """
    assert(len(X) == len(y))
    X_batch = np.empty(shape=((batch_size,) + get_processed_image_shape()), dtype='uint8')
    y_batch = np.empty(shape=(batch_size,), dtype='float32')
    global debug_images_dumped
    while True:
        for batch_idx in range(batch_size):
            source_idx = np.random.randint(len(y))
            # X[][0] provides the image path
            img = cv2.imread(X[source_idx][0])
            if DEBUG_IMAGES_ENABLED and not debug_images_dumped:
                cv2.imwrite("original.jpg", img)
            # Crop the car hood before shifting the image to prevent it appearing on the picture
            img = crop_hood(img)
            if DEBUG_IMAGES_ENABLED and not debug_images_dumped:
                cv2.imwrite("cropped_hood.jpg", img)
            # X[][2] provides the angle shift (before possible flipping)
            img = shift_image(img, X[source_idx][2])
            if DEBUG_IMAGES_ENABLED and not debug_images_dumped:
                cv2.imwrite("shifted_image_%.2f.jpg" % (X[source_idx][2]), img)
            # Crop the sky after shifting the image to minimize generating the sky in case of shifting the picture down
            img = crop_sky(img)
            if DEBUG_IMAGES_ENABLED and not debug_images_dumped:
                cv2.imwrite("cropped_sky.jpg", img)
            img = shift_brightness(img)
            if DEBUG_IMAGES_ENABLED and not debug_images_dumped:
                cv2.imwrite("shifted_brightness.jpg", img)
            # X[][2] indicates the flip
            if X[source_idx][1]:
                img = cv2.flip(img, 1)
                if DEBUG_IMAGES_ENABLED and not debug_images_dumped:
                    cv2.imwrite("flipped.jpg", img)
            img = resize_image(img)
            if DEBUG_IMAGES_ENABLED and not debug_images_dumped:
                cv2.imwrite("resized.jpg", img)
            debug_images_dumped = True
            X_batch[batch_idx], y_batch[batch_idx] = img, y[source_idx]
        yield X_batch, y_batch

def create_model(input_shape):
    """ Creates a new model.

    param: input_shape: the shape of the input
    returns: the model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/255.-0.5, input_shape=input_shape))
#    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Convolution2D(24,
                            5,5,
                            subsample=(2,2),
                            border_mode='valid',
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
    print("Preparing training set")
    X_train, y_train = prepare_training_set(DRIVING_LOG_FILE)
    X_train, y_train = shuffle(X_train, y_train)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.1, random_state=0)
    training_generator = generate_samples(X_train, y_train, batch_size=BATCH_SIZE)
    validation_generator = generate_samples(X_validation, y_validation, batch_size=BATCH_SIZE)
    print("Training model")
    model.fit_generator(training_generator,
                        samples_per_epoch=SAMPLES_PER_EPOCH,
                        nb_epoch=EPOCHS,
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
    test_score = model.evaluate(X_test, y_test, batch_size=TEST_BATCH_SIZE)
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

if __name__ == '__main__':
    model = create_model(get_processed_image_shape())

    # Load weights if exist
    if os.path.isfile(WEIGHTS_FILE):
        print("ATTENTION: loading existing weights")
        model.load_weights(WEIGHTS_FILE)

    # Download sample data if needed
    if not os.path.isfile(DRIVING_LOG_FILE):
        download_sample_data(SAMPLE_DATA_URL)

    train_model(model, DRIVING_LOG_FILE)

    sanity_check_model(model)

    test_model(model, DRIVING_LOG_FILE)

    save_model(model, MODEL_FILE, WEIGHTS_FILE)
