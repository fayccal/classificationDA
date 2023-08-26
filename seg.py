import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import tensorflow.keras.backend as K

import numpy as np
from sklearn.utils import shuffle
import cv2

import os

IMAGE_SIZE = 224
modelPath = 'segmodel.h5'


def iou_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

model = keras.models.load_model(modelPath, custom_objects={'iou_coef': iou_coef})

def open_images(path):

    image = load_img(path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = np.mean(image, axis=-1)/255.0
    return np.array(image)

def predict(images):
    pred = model.predict(images)
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    return pred



# récupère les indices le plus haut et bas
def find_segmentation_range_y(segmentation):
    non_zero_indices = np.nonzero(np.any(segmentation, axis=1))[0]

    if len(non_zero_indices) > 0:
        # Find the lowest and highest Y-axis values
        lowest_y = non_zero_indices[0]
        highest_y = non_zero_indices[-1]
    else:
        lowest_y = 0
        highest_y = segmentation.shape[0]

    return lowest_y, highest_y


def treat_all(input_folder, output_folder):
    already = os.listdir(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpeg") and filename not in already:

            input_path = os.path.join(input_folder, filename)

            output_path = os.path.join(output_folder, filename)

            image = open_images(input_path).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
            pred = predict(image)
            min_y, max_y = find_segmentation_range_y(pred[0])
            newimage = image[0]
            # garde seulement comme partie visible à partir des indices
            newimage[:min_y, :] = 0
            newimage[max_y:, :] = 0

            segmentation_image = (newimage * 255).astype(np.uint8)
            segmentation_image_rgb = cv2.cvtColor(segmentation_image, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(output_path, segmentation_image_rgb)