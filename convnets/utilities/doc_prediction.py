#%%

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image as KerasImage
import argparse

img_width, img_height = 224, 224


def predict_dir(dir, type):
    error_counter = 0
    directory = dir
    images = os.listdir(dir)
    filenames = []
    imgs = []
    for name in images:
        path = dir + '/' + name
        img_full = KerasImage.load_img(path)
        img = KerasImage.load_img(path, target_size=(img_width, img_height))
        img = np.expand_dims(img, axis=0)
        img = img / 255
        classes = model.predict(img)
        #     print(classes)
        if type == 'ndfl' or type == 'dover':
            classes = 1 - classes
        if classes < 0.5:
            error_counter = error_counter + 1
            filenames.append(name)
            imgs.append(img_full)
            info = str(error_counter) + "/" + str(len(images)) + " name: " + name + " value:" + str(
                float(classes[0]))
            print(info)

    dir_split = directory.split("/")
    original_label = dir_split[len(dir_split) - 2]
    for i in range(len(filenames)):
        title = 'Original label:{}, Prediction :{}, filename:{}'.format(
            original_label,
            "non-" + original_label,
            filenames[i]
        )
        image = imgs[i]
        plt.figure(figsize=[7, 7])
        plt.axis('off')
        plt.title(title)
        plt.imshow(image)
        plt.show()


def predict_one(weights_path, img_path, type):
    model = load_model_from_path(weights_path)
    img = KerasImage.load_img(img_path, target_size=(img_width, img_height))
    if img is None:
        print('Could not open image for path:', weights_path)
        exit(0)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    prediction = model.predict(img)
    if type == 'passport' or type == 'snils' or type == 'sved':
        if prediction > 0.5:
            return True
        else:
            return False
    else:
        if prediction < 0.5:
            return True
        else:
            return False


def load_model_from_path(weights_path):
    model = load_model(weights_path)
    if model is None:
        print('Could not load model for path:', weights_path)
        exit(0)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code for hand prediction.')
    parser.add_argument('--doctype', help='Type of the document. Variants: passport, snils, dover, ndfl, sved', required=True)
    parser.add_argument('--dir', help='Path to dir with images', required=False)
    parser.add_argument('--image', help='Path to one image', required=False)
    args = parser.parse_args()
    type = args.doctype
    weights_path = '../../data/weights/' + type + '_model_mcp.h5'
    model = load_model(weights_path)
    if model is None:
        print('Could not load model for path:', weights_path)
        exit(0)
    dir = args.dir
    if dir is not None:
        predict_dir(dir, type)
    else:
        img = args.image
        if img is not None:
            prediction = predict_one(img, weights_path, type)
            print(prediction)
