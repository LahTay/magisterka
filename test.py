from tensorflow.keras.layers import (Layer, Input, GlobalAveragePooling2D, Conv2D, Dense, BatchNormalization, multiply)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.backend import epsilon
from tensorflow.keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

from datasets.cut_dataset.load_cut_files import load_cut_files
from datasets.data_loaders.data_splitter import split_data
import tensorflow_model_optimization as tfmot
ConstantSparsity = tfmot.sparsity.keras.ConstantSparsity
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
import tensorflow as tf
import os
import time
import datetime
from generator import AudioGenerator
import pathlib

import tensorflow_model_optimization as tfmot
strip_pruning = tfmot.sparsity.keras.strip_pruning

from model.metrics import *
from sklearn.preprocessing import MultiLabelBinarizer


class Dataset:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

class TimeFreqAttentionLayer(Layer):
    def __init__(self, filters, **kwargs):
        super(TimeFreqAttentionLayer, self).__init__(**kwargs)
        self.filters = filters
        self.conv1 = Conv2D(filters=filters, kernel_size=(1, 1), padding='same', activation='relu')
        self.conv2 = Conv2D(filters=filters, kernel_size=(1, 5), padding='same', activation='sigmoid')
        self.conv3 = Conv2D(filters=filters, kernel_size=(5, 1), padding='same', activation='sigmoid')

    def call(self, inputs, **kwargs):
        feature_maps = self.conv1(inputs)
        attention_time = self.conv2(inputs)
        attention_freq = self.conv3(inputs)
        attended_time = multiply([feature_maps, attention_time])
        attended = multiply([attended_time, attention_freq])
        return attended

    def get_config(self):
        config = super(TimeFreqAttentionLayer, self).get_config()
        config.update({
            'filters': self.filters
        })
        return config

def create_tfq_prune(input_shape, num_classes, target_sparsity=0.5, begin_step=0, frequency=100, n=2, m=4):
    pruning_params_1_by_3 = {'sparsity_m_by_n': (n, m)}
    pruning_params_sparsity = {'pruning_schedule': ConstantSparsity(target_sparsity=target_sparsity,
                                                                    begin_step=begin_step, frequency=frequency)}

    input_layer = Input(shape=input_shape)

    x = (prune_low_magnitude(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'), **pruning_params_sparsity)
         (input_layer))
    x = BatchNormalization()(x)
    x = prune_low_magnitude(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'), **pruning_params_1_by_3)(x)
    x = BatchNormalization()(x)
    x = prune_low_magnitude(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'), **pruning_params_1_by_3)(x)
    x = BatchNormalization()(x)

    x = TimeFreqAttentionLayer(filters=64)(x)

    x = GlobalAveragePooling2D()(x)
    x = prune_low_magnitude(Dense(128, activation='relu'), **pruning_params_1_by_3)(x)
    x = prune_low_magnitude(Dense(64, activation='relu'), **pruning_params_sparsity)(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def create_tfq_no_prune(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    x = BatchNormalization()(x)

    x = TimeFreqAttentionLayer(filters=64)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def create_attentionMIC_TFQ_model(input_shape, num_classes, prune=False,
                                  target_sparsity=0.5, begin_step=0, frequency=100, n=2, m=4):
    if prune:
        model = create_tfq_prune(input_shape, num_classes, target_sparsity, begin_step, frequency, n, m)
    else:
        model = create_tfq_no_prune(input_shape, num_classes)
    return model


def train(self, epochs=30):

    if not self._load_model():
        input_shape = (*self.generator.data_shape, 1)
        self.model = create_attentionMIC_TFQ_model(input_shape, self.num_outputs, self.prune)

        metrics = [MultiLabelAccuracy(self.num_outputs, self.classes_names),
                   MultiLabelPrecision(self.num_outputs, self.classes_names),
                   MultiLabelRecall(self.num_outputs, self.classes_names),
                   MultiLabelF1Score(self.num_outputs, self.classes_names),
                   MultiLabelInformedness(self.num_outputs, self.classes_names),
                   MultiLabelMarkedness(self.num_outputs, self.classes_names),
                   MultiLabelMCC(self.num_outputs, self.classes_names),
                   MultiLabelCohenKappa(self.num_outputs, self.classes_names)]

        if self.pretrained_model is not None:
            self._transfer_weights()

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                           loss="binary_crossentropy",
                           metrics=metrics)
    if self.verbose:
        self.model.summary()
        if self.pretrained_model is not None:
            self._verify_weights_transfer()

    model_checkpoint_callback = self._set_checkpoint_callback()
    tensorboard_callback = self._set_tensorboard_callback(histogram_freq=1, update_freq="epoch")
    early_stopping = EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience,
                                   restore_best_weights=True, mode="min")

    wandb_callback = WandbMetricsLogger()
    wandb_checkpoint = WandbModelCheckpoint("models")
    pruning = tfmot.sparsity.keras.UpdatePruningStep()

    callbacks = [model_checkpoint_callback, tensorboard_callback, wandb_callback,
                 early_stopping, pruning]

    #if self.prune:
    #    callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

    start_time = time.time()
    self.model.fit(self.generator,
                   epochs=epochs,
                   steps_per_epoch=len(self.generator),
                   verbose=1,
                   use_multiprocessing=False,
                   workers=4,
                   callbacks=callbacks,
                   validation_data=self.val_generator,
                   validation_steps=len(self.val_generator))
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds for {(self.generator.batch_size - 1) * len(self.generator)} samples")

    if self.prune and self.load_model_path == "":
        self._save_and_convert_pruned_model()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                           loss="binary_crossentropy",
                           metrics=metrics)
        return

    self.model.save('model_no_pruning.h5')
    file_size = os.path.getsize('model_no_pruning.h5')
    print(f"Rozmiar modelu na dysku: {file_size} bajtów")

def get_model(self):
    return self.model


def _set_checkpoint_callback(self):
    model_checkpoint_callback = ModelCheckpoint(
        filepath=self.checkpoint_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '_{epoch:02d}.h5',
        save_weights_only=False,
        save_best_only=True,
        verbose=1)
    if self.verbose:
        print(f"Checkpoint callback set to filepath: {os.path.abspath(self.checkpoint_dir)}")

    return model_checkpoint_callback

def _set_tensorboard_callback(self, histogram_freq=1, update_freq=50):
    tensorboard_callback = TensorBoard(log_dir=self.log_dir, histogram_freq=histogram_freq, update_freq=update_freq)
    if self.verbose:
        print(f"Tensorboard callback set to filepath: {os.path.abspath(self.log_dir)}")
    return tensorboard_callback

def file_loader(files_path, is_string=False):
    filenames_train, labels_train = load_cut_files(pathlib.Path(files_path).absolute(), is_string)

    return filenames_train, labels_train


def save_and_convert_pruned_model(model, mode, folder):
    # Strip pruning before saving the model
    model = strip_pruning(model)

    pruned_model_path = os.path.join(folder, 'my_pruned_model_' + mode)
    model.save(pruned_model_path, save_format='tf')

    pruned_model_h5_path = os.path.join(folder, 'my_pruned_model_' + mode + '.h5')
    model.save(pruned_model_h5_path)  # Saves as a .h5 file

    converter = tf.lite.TFLiteConverter.from_saved_model(pruned_model_path)
    converter.experimental_enable_resource_variables = True
    converter.experimental_new_converter = True
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    tflite_model_path = os.path.join(folder, 'my_model' + mode + '.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    # Print the size of the saved pruned model
    file_size = os.path.getsize(pruned_model_h5_path)
    print(f"Rozmiar pruned modelu na dysku: {file_size} bajtów")

    # Print the size of the TFLite model
    tflite_file_size = os.path.getsize(tflite_model_path)
    print(f"Rozmiar TFLite modelu na dysku: {tflite_file_size} bajtów")



def return_models(dt, batch_size, mlb, c):
    generator = AudioGenerator(dt.x_train, dt.y_train, batch_size, use_mfcc=True, multi_label=False,
                               classes_names=mlb.classes_, testing=False)

    input_shape = (40, 216, 1)

    num_outputs = len(mlb.classes_)
    model = create_attentionMIC_TFQ_model(input_shape, num_outputs, prune=True, n=c[0], m=c[1], target_sparsity=c[2])
    model_no_prune = create_attentionMIC_TFQ_model(input_shape, num_outputs, prune=False)

    model.summary()
    model_no_prune.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                       loss="binary_crossentropy")

    model_no_prune.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss="binary_crossentropy")



    model.fit(generator,
                   epochs=1,
                   steps_per_epoch=len(generator),
                   verbose=1,
                   use_multiprocessing=False,
                   workers=4,
                   callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
              )

    model_no_prune.fit(generator,
                       epochs=1,
                       steps_per_epoch=len(generator),
                       verbose=1,
                       use_multiprocessing=False,
                       workers=4
                       )



    return model, model_no_prune


files_path = f"./datasets/datasets/musicnet/{2.5}s_len.{0}s_overlap"

file_folder = 'test'

filenames, labels = file_loader(files_path, False)
mlb = MultiLabelBinarizer()
start_time = time.time()
x_train, y_train, x_val, y_val, x_test, y_test = split_data(filenames, labels, mlb, 0.1, 0.1, False)

end_time = time.time()
print(f"Split_data time for audio_length {2.5}s: {end_time-start_time}s")
dt = Dataset(x_train, y_train, x_val, y_val, x_test, y_test)

config = [(1, 9, 0.2), (8, 9, 0.8)]
for c in config:
    model, no_prune = return_models(dt, 42, mlb, c)
    no_prune_filepath = os.path.join(file_folder, 'model_no_pruning.h5')
    prune_filepath = os.path.join(file_folder, 'model_prune.h5')
    prune_filepath_stripped = os.path.join(file_folder, 'model_prune_stripped.h5')


    #save_and_convert_pruned_model(model, f'{c[0]}_{c[1]}', file_folder)
    no_prune.save(no_prune_filepath)
    model.save(prune_filepath)
    model = strip_pruning(model)
    model.summary()
    model.save(prune_filepath_stripped)
    file_size = os.path.getsize(no_prune_filepath)
    file_size2 = os.path.getsize(prune_filepath)
    file_size3 = os.path.getsize(prune_filepath_stripped)
    print(f"Rozmiar modelu na dysku prune: {file_size} bajtów")
    print(f"Rozmiar modelu na dysku no prune: {file_size2} bajtów")
    print(f"Rozmiar modelu na dysku no prune with stripping: {file_size3} bajtów")


