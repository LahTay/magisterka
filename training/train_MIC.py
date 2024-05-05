import os
import datetime
from enum import Enum
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from model.metrics.MultiLabelAccuracy import MultiLabelAccuracy
from model.metrics.MultiLabelPrecision import MultiLabelPrecision
from model.metrics.MultiLabelF1Score import MultiLabelF1Score
from model.custom_callbacks.CSVLogger import CSVLoggerCallback
from model.metrics.F1Score import F1Score
from model.metrics.HammingLoss import HammingLoss

from model.attentionMIC.spatial_groupwise_enhance_attentionMIC import create_attentionMIC_SGE_model
from model.attentionMIC.efficient_channel_attentionMIC import create_attentionMIC_ECA_model
from model.attentionMIC.time_frequency_attentionMIC import create_attentionMIC_TFQ_model
from model.attentionMIC.attentionMIC import create_attentionMIC_model


from model.conv_model.conv_model import create_conv_model_from_paper

import tensorflow as tf


class MICType(Enum):
    NONE = 0
    ECA = 1
    SGE = 2
    TFQ = 3


class TrainMICModel:
    def __init__(self, MIC_type: MICType, generator, validation_generator, log_dir, checkpoint_dir, classes_names,
                 load_model_path="", verbose=1):
        self.MIC_type = MIC_type
        self.generator = generator
        self.val_generator = validation_generator
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.load_model_path = load_model_path
        self.model = None
        self.num_outputs = self.generator.get_label_num()
        self.classes_names = classes_names
        self.verbose = verbose

    def __call__(self, epochs, *args, **kwargs):
        self.train(epochs)
        return self.get_model()

    def train(self, epochs=30):

        if not self._load_model():
            input_shape = (*self.generator.data_shape, 1)
            match self.MIC_type:
                case MICType.ECA:
                    self.model = create_attentionMIC_ECA_model(input_shape, self.num_outputs)
                case MICType.SGE:
                    num_groups = 1
                    # TODO: NOT FINISHED
                    self.model = create_attentionMIC_SGE_model(input_shape, self.num_outputs, num_groups)
                case MICType.TFQ:
                    self.model = create_attentionMIC_TFQ_model(input_shape, self.num_outputs)
                case _:
                    self.model = create_attentionMIC_model(input_shape, self.num_outputs)

            metrics = [tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score(), HammingLoss()]
            self.model.compile(optimizer='adam',
                               loss='binary_crossentropy',
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
                       use_multiprocessing=False,
                       workers=4,
                       callbacks=[model_checkpoint_callback, tensorboard_callback, csv_logger_callback],
                       validation_data=self.val_generator,
                       validation_steps=len(self.val_generator))

    def get_model(self):
        return self.model

    def _load_model(self):
        if self.load_model_path != "":
            self.model = load_model(self.load_model_path,
                                    custom_objects={'F1Score': F1Score(),
                                                    'HammingLoss': HammingLoss()})
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

    def _set_tensorboard_callback(self, histogram_freq=1, update_freq: (int | str) = 50):
        tensorboard_callback = TensorBoard(log_dir=self.log_dir, histogram_freq=histogram_freq, update_freq=update_freq)
        if self.verbose:
            print(f"Tensorboard callback set to filepath: {os.path.abspath(self.log_dir)}")
        return tensorboard_callback
