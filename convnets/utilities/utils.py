from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import *


def init_generators(train_data_dir, validation_data_dir, img_width, img_height, batch_size_train, batch_size_val):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size_train,
        color_mode='rgb',
        class_mode='binary',
        shuffle=True)

    val_generator = val_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size_val,
        color_mode='rgb',
        class_mode='binary',
        shuffle=True)
    return train_generator, val_generator


def plot(history):
    # Plot training & validation accuracy values
    print(history.history.keys())
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def save_model_structure_to_json(path, model):
    json_file = open(path, "w")
    json_file.write(model.to_json())
    json_file.close()


def load_model_structure_from_json(path):
    json_file = open(path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    return model_from_json(loaded_model_json)


def freeze(model):
    for layer in model.layers:
        layer.trainable = False

        if isinstance(layer, Model):
            freeze(layer)


def unfreeze(model):
    for layer in model.layers:
        layer.trainable = True

        if isinstance(layer, Model):
            unfreeze(layer)
