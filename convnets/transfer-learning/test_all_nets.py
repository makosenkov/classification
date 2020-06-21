# %%
from tensorflow.keras.applications import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from convnets.utilities import utils_nets as utils
import matplotlib.pyplot as plt

# %%
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = '../../data/passport_2class_divided_clean/train'
validation_data_dir = '../../data/passport_2class_divided_clean/validation'
nb_train_samples = 480
nb_validation_samples = 130
epochs = 20
batch_size = 3

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
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=True)

val_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='binary',
    shuffle=True)


# %%
def plot_all_nets(histories, titles):
    rows = 2
    cols = 5
    axes = []
    fig = plt.figure(figsize=(19.2, 10.8))

    for i in range(rows * cols):
        axes.append(fig.add_subplot(rows, cols, i + 1))
        subplot_title = (titles[i])
        axes[-1].set_title(subplot_title)
        plt.plot(histories[i].history['accuracy'])
        plt.plot(histories[i].history['val_accuracy'])
        plt.axis('off')
        plt.show()
    fig.tight_layout()
    plt.show()


def train_classifier_by_model_name(model_name):
    model = Sequential()
    if model_name == 'resnet':
        model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
    elif model_name == 'inception':
        model.add(InceptionV3(include_top=False, weights='imagenet'))
    elif model_name == 'densenet':
        model.add(DenseNet121(include_top=False, weights='imagenet'))
    elif model_name == 'mobilenet':
        model.add(MobileNetV2(input_shape=(img_width, img_height, 3), alpha=1.0, include_top=False, weights='imagenet'))
    elif model_name == 'vgg16':
        model.add(VGG16(include_top=False, weights='imagenet', input_shape=(img_width, img_height, 3)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.layers[0].trainable = False

    opt = Adam()
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=nb_validation_samples // batch_size
    )
    return history


def run_train():
    history_resnet = train_classifier_by_model_name('resnet')
    history_inception = train_classifier_by_model_name('inception')
    history_densenet = train_classifier_by_model_name('densenet')
    history_mobilenet = train_classifier_by_model_name('mobilenet')
    history_vgg16 = train_classifier_by_model_name('vgg16')
    histories = [history_resnet, history_inception, history_densenet, history_mobilenet, history_vgg16]
    titles = ['Resnet50', 'InceptionV3', 'Densenet121', 'MobilenetV2', 'VGG16']
    plot_all_nets(histories, titles)


# %%
history_resnet = train_classifier_by_model_name('resnet')
# %%
history_inception = train_classifier_by_model_name('inception')
# %%
history_densenet = train_classifier_by_model_name('densenet')
# %%
history_mobilenet = train_classifier_by_model_name('mobilenet')
# %%
history_vgg16 = train_classifier_by_model_name('vgg16')
# %%
histories = [history_resnet, history_inception, history_densenet, history_mobilenet, history_vgg16]
titles = ['Resnet50', 'InceptionV3', 'Densenet121', 'MobilenetV2', 'VGG16']
plot_all_nets(histories, titles)
