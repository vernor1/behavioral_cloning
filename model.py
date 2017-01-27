import cv2
import numpy as np
import multiprocessing
import os.path
import pandas as pd
import re

from image_processing import get_processed_image_shape, crop_hood, crop_sky, resize_image, normalize_image, process_image
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from scipy import ndimage
from tqdm import tqdm


DRIVING_LOG_FILE = 'driving_log.csv'
MODEL_FILE = 'model.json'
WEIGHTS_FILE = 'model.h5'
BATCH_SIZE = 256
EPOCHS = 5
SAMPLES_PER_EPOCH = 102400
ACTIVATION_FUNCTION = 'relu'
LEARNING_RATE = 5e-4
LAMBDA = 1e-5
DROPOUT_KEEP_PROB = 0.5
SIDE_CAMERA_ANGLE = 0.25               # Angle units
STRAIGHT_STEERING_ANGLE = 0.1          # Angle units
MAX_ANGLE_SHIFT = 0.25                 # Angle units
IMAGES_PER_FULL_STEERING_ANGLE = 0.5   # Factor of horizontal picture size. It's observed that the pictures from the left and right cameras
                                       # have and offset of about 40px from the center picture. People reported that the angle offset
                                       # of the side cameras worked good at 0.25. So the full angle of 1.0 would shift image for about
                                       # 40/0.25=160, which is 160/320=0.5 images per full steering angle.
MAX_VERTICAL_SHIFT = 0.2               # Factor of vertical picture size
BRIGHTNESS_SHIFT_LIMITS = (0.25, 1.50) # Factor of original brightness


def save_image(in_img, file_name):
#    out_img = cv2.cvtColor(in_img, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(file_name, in_img)

def shift_image(in_img, angle):
    angle_shift = np.random.uniform(-MAX_ANGLE_SHIFT, MAX_ANGLE_SHIFT)
#    print("Angle shift %.2f" % (angle_shift))
    shift_x = angle_shift * IMAGES_PER_FULL_STEERING_ANGLE * in_img.shape[1]
    shift_y = np.random.randint(-MAX_VERTICAL_SHIFT, MAX_VERTICAL_SHIFT)
#    print("Shift x, y = %d, %d" % (shift_x, shift_y))
#    out_img = ndimage.shift(in_img, shift=(shift_y, shift_x, 0), mode='nearest')
    transformation_matrix = np.float32([[1,0,shift_x],[0,1,shift_y]])
    out_img = cv2.warpAffine(in_img,
                             transformation_matrix,
                             (in_img.shape[1], in_img.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)
    return out_img, angle + angle_shift

def shift_brightness(in_img):
    img = cv2.cvtColor(in_img, cv2.COLOR_BGR2HSV)
    brightness_multiplier = np.random.uniform(BRIGHTNESS_SHIFT_LIMITS[0], BRIGHTNESS_SHIFT_LIMITS[1])
#    print("Brightness factor %.2f" % (brightness_multiplier))
    img[:,:,2] = img[:,:,2].clip(0, 255 // brightness_multiplier) * brightness_multiplier
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

def generate_sample_from_log_entry(row):
    camera_selector = np.random.randint(3)
    angle = row[3]
    if camera_selector == 1:
        angle += SIDE_CAMERA_ANGLE
    elif camera_selector == 2:
        angle -= SIDE_CAMERA_ANGLE
#    print("Original angle %.2f" % (angle))
    img = cv2.imread(re.search('IMG\/.+', row[camera_selector]).group())
#    if not os.path.isfile('original.jpg'):
#        save_image(img, 'original.jpg')
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
#    save_image(img, 'generated.jpg')
    img = normalize_image(img)
#    if not os.path.isfile('processed.jpg'):
#        save_image(img, 'processed.jpg')
#    print("Final angle %.2f" % (angle))
    return img, angle

def generate_samples_from_driving_log(driving_log_file, batch_size=32, discard_straight_sample_prob=0):
    driving_log = pd.read_csv(driving_log_file, header=None)
    X = np.empty(shape=((batch_size,) + get_processed_image_shape()), dtype='uint8')
    Y = np.empty(shape=(batch_size,), dtype='float32')
    while True:
        for idx in range(batch_size):
            row = driving_log.ix[np.random.randint(len(driving_log))]
            is_sample_acceptable = None
            while not is_sample_acceptable:
                x, y = generate_sample_from_log_entry(row)
                if abs(y) > STRAIGHT_STEERING_ANGLE or np.random.random() > discard_straight_sample_prob:
                    is_sample_acceptable = True
            X[idx] = x
            Y[idx] = y
        yield X, Y

def predict_steering_angle(model, img_name):
    img = cv2.imread(img_name)
    transformed_image_array = process_image(img)[None, :, :, :]
    return float(model.predict(transformed_image_array, batch_size=1))

def create_model(input_shape):
    # Model definition
    model = Sequential()
    model.add(Convolution2D(24,
                            5,5,
                            subsample=(2,2),
                            border_mode='valid',
                            input_shape=input_shape,
                            activation=ACTIVATION_FUNCTION,
                            W_regularizer=l2(LAMBDA)))
    print("Input shape %s" % (model.layers[-1].input_shape,))
    print("Conv. layer 1 %s" % (model.layers[-1].output_shape,))
    model.add(Convolution2D(36,
                            5,5,
                            subsample=(2,2),
                            border_mode='valid',
                            activation=ACTIVATION_FUNCTION,
                            W_regularizer=l2(LAMBDA)))
    print("Conv. layer 2 %s" % (model.layers[-1].output_shape,))
    model.add(Convolution2D(48,
                            5,5,
                            subsample=(2,2),
                            border_mode='valid',
                            activation=ACTIVATION_FUNCTION,
                            W_regularizer=l2(LAMBDA)))
    print("Conv. layer 3 %s" % (model.layers[-1].output_shape,))
    model.add(Convolution2D(64,
                            3,3,
                            border_mode='valid',
                            activation=ACTIVATION_FUNCTION,
                            W_regularizer=l2(LAMBDA)))
    print("Conv. layer 4 %s" % (model.layers[-1].output_shape,))
    model.add(Convolution2D(64,
                            3,3,
                            border_mode='valid',
                            activation=ACTIVATION_FUNCTION,
                            W_regularizer=l2(LAMBDA)))
    print("Conv. layer 5 %s" % (model.layers[-1].output_shape,))
    model.add(Flatten())
    print("Flatten %s" % (model.layers[-1].output_shape,))
    model.add(Dense(100,
                    activation=ACTIVATION_FUNCTION,
                    W_regularizer=l2(LAMBDA)))
    model.add(Dropout(DROPOUT_KEEP_PROB))
    print("Fully-connected layer 1 %s" % (model.layers[-1].output_shape,))
    model.add(Dense(50,
                    activation=ACTIVATION_FUNCTION,
                    W_regularizer=l2(LAMBDA)))
    model.add(Dropout(DROPOUT_KEEP_PROB))
    print("Fully-connected layer 2 %s" % (model.layers[-1].output_shape,))
    model.add(Dense(10,
                    activation=ACTIVATION_FUNCTION,
                    W_regularizer=l2(LAMBDA)))
    model.add(Dropout(DROPOUT_KEEP_PROB))
    print("Fully-connected layer 3 %s" % (model.layers[-1].output_shape,))
    model.add(Dense(1,
                    W_regularizer=l2(LAMBDA)))
    print("Output %s" % (model.layers[-1].output_shape,))

    adam = Adam(lr=LEARNING_RATE)
    model.compile(adam, 'mse')
    return model

def train_model(model):
    validation_generator = generate_samples_from_driving_log(DRIVING_LOG_FILE, batch_size=BATCH_SIZE)
    for epoch in range(EPOCHS):
        discard_prob = 1-epoch/(EPOCHS-1)
        print("Epoch %d/%d, probability of discarding straight driving samples %.2f" % (epoch+1, EPOCHS, discard_prob))
        training_generator = generate_samples_from_driving_log(DRIVING_LOG_FILE,
                                                               batch_size=BATCH_SIZE,
                                                               discard_straight_sample_prob=discard_prob)
        model.fit_generator(training_generator,
                            samples_per_epoch=SAMPLES_PER_EPOCH,
                            nb_epoch=1,
                            validation_data=validation_generator,
                            nb_val_samples=SAMPLES_PER_EPOCH // 5,
                            nb_worker=multiprocessing.cpu_count(),
                            pickle_safe=True)


def sanity_check_model(model):
    print("Emergency steering right %f" % (predict_steering_angle(model, 'emergency_right.jpg')))
    print("Emergency steering left %f" % (predict_steering_angle(model, 'emergency_left.jpg')))
    test_file_names = ['left.jpg',
                       'straight.jpg',
                       'right.jpg']
    for test_file_name in test_file_names:
        print("Testing %s: %f" % (test_file_name, predict_steering_angle(model, test_file_name)))

def save_model(model):
    print("Saving model and weights")
    model.save_weights(WEIGHTS_FILE)
    json_model = model.to_json()
    with open(MODEL_FILE, 'w') as f:
        f.write(json_model)

model = create_model(get_processed_image_shape())

# Load weights if exist
if os.path.isfile(WEIGHTS_FILE):
    print("ATTENTION: loading existing weights")
    model.load_weights(WEIGHTS_FILE)

train_model(model)

sanity_check_model(model)

save_model(model)
