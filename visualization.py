import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from scipy.misc import imsave
import numpy as np
import time
import cv2
from keras.applications import vgg16
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
import h5py
from keras import __version__ as keras_version

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    K.set_learning_phase(1)
    model = load_model(args.model)
    # dimensions of the generated pictures for each filter.
    img_width = 160
    img_height = 320
    layer_name = "conv2d_3"

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128
    input_img_data = np.array([cv2.imread('input.jpg')])

    intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[8].output])
    intermediate_tensor = intermediate_tensor_function([input_img_data])[0]
    print intermediate_tensor.shape
    visImg = deprocess_image(intermediate_tensor)
    cv2.imshow('vis', visImg[0,:,:,0])
    cv2.waitKey(0)
