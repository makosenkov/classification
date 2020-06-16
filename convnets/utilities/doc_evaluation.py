#%%

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

img_width, img_height = 224, 224


def evaluate(model, evaluation_data_dir):
    global generator
    datagen = ImageDataGenerator(rescale=1. / 255)
    generator = datagen.flow_from_directory(
        evaluation_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='binary',
        shuffle=False)
    model.evaluate(generator)


def build_confusion_matrix():
    generator.reset()
    Y_pred = model.predict(generator, 490)
    y_pred = np.where(Y_pred < 0.5, 0, 1)
    # print(Y_pred)
    # print(y_pred)
    print('Confusion Matrix')
    print(confusion_matrix(generator.classes, y_pred))
    print('Classification Report')
    target_names = ['non-' + type, type]
    print(classification_report(generator.classes, y_pred, target_names=target_names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Code for evaluating models.')
    parser.add_argument('--doctype', help='Type of the document. Variants: passport, snils, dover, ndfl, sved', default='passport')
    args = parser.parse_args()
    global type
    type = args.doctype
    weights_path = '../../data/weights/' + type + '_model_mcp.h5'
    model = load_model(weights_path)
    if model is None:
        print('Could not load model for path:', weights_path)
        exit(0)
    evaluation_dir = '../../data/evaluation/' + type
    evaluate(model, evaluation_dir)
    build_confusion_matrix()
