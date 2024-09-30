import os
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from model.metrics.MultiLabelAccuracy import MultiLabelAccuracy
from model.metrics.MultiLabelPrecision import MultiLabelPrecision
from model.metrics.MultiLabelF1Score import MultiLabelF1Score
from model.custom_callbacks.CSVLogger import CSVLoggerCallback
from model.metrics.F1Score import F1Score

from model.conv_model.conv_model import create_conv_model_from_paper

import tensorflow as tf



class TrainConvModel:
    def __init__(self, generator, validation_generator, log_dir, checkpoint_dir, classes_names, load_model_path="",
                 verbose=1, class_weights=None, learning_rate=0.001, pretrain_model=None):
        self.generator = generator
        self.val_generator = validation_generator
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.load_model_path = load_model_path
        self.model = None
        self.num_outputs = self.generator.get_label_num()
        self.classes_names = classes_names
        self.verbose = verbose
        self.class_weights = class_weights
        self.lr = learning_rate
        self.pretrained_model = pretrain_model

    def __call__(self, epochs, *args, **kwargs):
        self.train(epochs)
        return self.get_model()

    def train(self, epochs=30):

        if not self._load_model():
            self.model = create_conv_model_from_paper((*self.generator.data_shape, 1),
                                                      self.num_outputs, self.classes_names)

            losses = {}
            metrics = {}
            for name in self.classes_names:
                name = name.replace(" ", "_")
                output_name = f"{name}"
                losses[output_name] = 'binary_crossentropy'
                metrics[output_name] = [
                    tf.keras.metrics.BinaryAccuracy(name=f'accuracy'),
                    tf.keras.metrics.Precision(name=f'precision'),
                    tf.keras.metrics.Recall(name=f'recall'),
                    F1Score(name=f'f1_score')
                ]

            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                               loss=losses,
                               metrics=metrics)
        if self.verbose:
            self.model.summary()

        model_checkpoint_callback = self._set_checkpoint_callback()
        tensorboard_callback = self._set_tensorboard_callback(histogram_freq=1, update_freq="epoch")
        csv_logger_callback = CSVLoggerCallback(self.log_dir)

        self.model.fit(self.generator,
                       epochs=epochs,
                       steps_per_epoch=len(self.generator),
                       verbose=1,
                       callbacks=[model_checkpoint_callback, tensorboard_callback, csv_logger_callback],
                       validation_data=self.val_generator,
                       validation_steps=len(self.val_generator)
                       )

    def get_model(self):
        return self.model

    def _load_model(self):
        if self.load_model_path != "":
            self.model = load_model(self.load_model_path,
                                    custom_objects={'F1Score': F1Score()})
            if self.verbose:
                print(f"Model loaded from filepath: {os.path.abspath(self.load_model_path)}")
            return True
        return False

    def _set_checkpoint_callback(self):
        model_checkpoint_callback = ModelCheckpoint(
            filepath=self.checkpoint_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '{epoch:02d}.h5',
            save_weights_only=False,
            save_best_only=False,
            verbose=1)
        if self.verbose:
            print(f"Checkpoint callback set to filepath: {os.path.abspath(self.checkpoint_dir)}")

        return model_checkpoint_callback

    def _set_tensorboard_callback(self, histogram_freq=1, update_freq=50):
        tensorboard_callback = TensorBoard(log_dir=self.log_dir, histogram_freq=histogram_freq, update_freq=update_freq)
        if self.verbose:
            print(f"Tensorboard callback set to filepath: {os.path.abspath(self.log_dir)}")
        return tensorboard_callback
