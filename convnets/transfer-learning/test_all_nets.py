from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.models import *
from convnets.utilities import utils_nets as utils
import matplotlib.pyplot as plt

img_width, img_height = 224, 224

def evaluate(model, evaluation_data_dir, type):
    global generator
    datagen = ImageDataGenerator(rescale=1. / 255)
    print("\n====================" + str(type) + " testing======================\n")
    generator = datagen.flow_from_directory(
        evaluation_data_dir,
        target_size=(img_width, img_height),
        batch_size=1,
        class_mode='binary',
        shuffle=False)
    model.evaluate(generator)


def build_confusion_matrix(model, type):
    generator.reset()
    Y_pred = model.predict(generator, 680)
    y_pred = np.where(Y_pred < 0.5, 0, 1)
    # print(Y_pred)
    # print(y_pred)
    print('Confusion Matrix')
    print(confusion_matrix(generator.classes, y_pred))
    print('Classification Report')
    target_names = ['non-' + type, type]
    print(classification_report(generator.classes, y_pred, target_names=target_names))


def eval(weights_path, type):
    model = load_model(weights_path)
    if model is None:
        print('Could not load model for path:', weights_path)
        exit(0)

    evaluation_dir = "../../data/" + str(type) + "_2class_divided_clean/evaluation/"
    evaluate(model, evaluation_dir, type)
    build_confusion_matrix(model, type)


if __name__ == '__main__':
    # print("\n====================DENSENET TESTING======================\n")
    # eval('../../data/weights/densenet/densenet_passport_fine_tuned.h5')
    eval('../../data/weights/inception/inception_passport.h5', 'passport')
    eval('../../data/weights/inception/inception_snils.h5', 'snils')
    eval('../../data/weights/inception/inception_dover.h5', 'dover')
    eval('../../data/weights/inception/inception_ndfl.h5', 'ndfl')
    eval('../../data/weights/inception/inception_sved.h5', 'sved')
    # print("\n====================MOBILENETv2 TESTING======================\n")
    # eval('../../data/weights/mobilenet/fine_tuned_mobilenet_passport.h5')
    # print("\n====================VGG16 TESTING======================\n")
    # eval('../../data/weights/passport_model_mcp.h5')
    # print("\n====================CONV3BLOCKS TESTING======================\n")
    # eval('../../data/weights/simple/conv3_passport.h5')
    # print("\n====================CONV4BLOCKS TESTING======================\n")
    # eval('../../data/weights/simple/conv4_passport.h5')