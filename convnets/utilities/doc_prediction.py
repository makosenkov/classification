#%%

import os
import matplotlib.pyplot as plt
import numpy as np
import time
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image as KerasImage
import argparse
from statistics import mean

img_width, img_height = 224, 224


def predict_dir(dir, type):
    error_counter = 0
    directory = dir
    images = os.listdir(dir)
    filenames = []
    imgs = []
    times = []
    start_full_loop = time.process_time()
    for name in images:
        path = dir + '/' + name
        img_full = KerasImage.load_img(path)
        img = KerasImage.load_img(path, target_size=(img_width, img_height))
        img = np.expand_dims(img, axis=0)
        img = img / 255
        start = time.process_time()
        classes = model.predict(img)
        #     print(classes)
        times.append(time.process_time() - start)
        if type == 'ndfl' or type == 'dover':
            classes = 1 - classes
        if classes < 0.5:
            error_counter = error_counter + 1
            filenames.append(name)
            imgs.append(img_full)
            info = str(error_counter) + "/" + str(len(images)) + " name: " + name + " value:" + str(
                float(classes[0]))
            print(info)
    print("Full loop: " + str(time.process_time() - start_full_loop))
    sum = 0
    for elem in times:
        sum += elem
    res = sum / len(times)
    print("Average for 1 image: " + str(res))


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


def predict_one(model, img_path, type):
    start = time.process_time()
    img = KerasImage.load_img(img_path, target_size=(img_width, img_height))
    if img is None:
        print('Could not open image for path:', weights_path)
        exit(0)
    img = np.expand_dims(img, axis=0)
    img = img / 255
    prediction = model.predict(img)
    print(time.process_time() - start)
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
    weights_path = '../../data/weights/simple/conv3_' + type + '.h5'
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
            prediction = predict_one(model, img, type)
            print(prediction)
