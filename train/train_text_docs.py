from __future__ import print_function

import os
import sys

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error, RotNetDataGenerator, binarize_images
from data.text_docs import get_filenames as get_text_docs_data

data_path = os.path.join('data', 'text_docs')

X_train, X_test = get_text_docs_data(data_path)

model_name = 'rotnet_text_docs'

# number of convolutional filters to use
nb_filters = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# number of classes
nb_classes = 360
# nb_classes = 4                                    # for 4 classes

# input image shape
nb_train_samples, img_rows, img_cols = X_train.shape
img_channels = 1
# img_channels = 3
input_shape = (img_rows, img_cols, img_channels)
nb_test_samples = X_test.shape[0]

print('Input shape:', input_shape)
print(nb_train_samples, 'train samples')
print(nb_test_samples, 'test samples')

# model definition 
# ---------------------------------------------------------------
input = Input(shape=input_shape)
x = Conv2D(nb_filters, kernel_size, activation='relu')(input)
x = Conv2D(nb_filters, kernel_size, activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.4)(x)                                     # 0.25 0.4
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)                                     # 0.25 0.4
x = Dense(nb_classes, activation='softmax')(x)

model = Model(inputs=input, outputs=x)
# ----------------------------------------------------------------

# # load base model
# base_model = ResNet50(weights='imagenet', include_top=False,
#                       input_shape=input_shape)

# # append classification layer
# x = base_model.output
# x = Flatten()(x)
# final_output = Dense(nb_classes, activation='softmax', name='fc360')(x)

# # create the new model
# model = Model(inputs=base_model.input, outputs=final_output)

# ----------------------------------------------------------------

model.summary()

# model compilation
model.compile(loss='categorical_crossentropy',
            #   optimizer='adam',
              optimizer=SGD(lr=0.01, momentum=0.9),         # lr=0.01, momentum=0.9
              metrics=[angle_error, 'accuracy'])

# training parameters
batch_size = 8          # 128
nb_epoch = 50           # 20

output_folder = 'models'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# callbacks
monitor = 'val_angle_error'
checkpointer = ModelCheckpoint(
    filepath=os.path.join(output_folder, model_name + '.hdf5'),
    monitor=monitor,
    save_best_only=True
)
reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)      #3
early_stopping = EarlyStopping(monitor=monitor, patience=5)     #5
tensorboard = TensorBoard()

# training loop
model.fit_generator(
    RotNetDataGenerator(
        X_train,
        batch_size=batch_size,
        preprocess_func=binarize_images,
        shuffle=True
    ),
    steps_per_epoch=nb_train_samples / batch_size,
    epochs=nb_epoch,
    validation_data=RotNetDataGenerator(
        X_test,
        batch_size=batch_size,
        preprocess_func=binarize_images,
    ),
    validation_steps=nb_test_samples / batch_size,
    verbose=1,
    callbacks=[checkpointer, reduce_lr, early_stopping, tensorboard],
    workers=10
)
