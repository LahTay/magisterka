import os
import datetime
from enum import Enum
import time

from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.backend import epsilon
from tensorflow.keras.utils import plot_model
from model.custom_callbacks.CSVLogger import CSVLoggerCallback
from model.metrics.F1Score import F1Score
from model.metrics.HammingLoss import HammingLoss
from tensorflow.keras.metrics import AUC
import tensorflow_model_optimization as tfmot
strip_pruning = tfmot.sparsity.keras.strip_pruning

from model.attentionMIC.spatial_groupwise_enhance_attentionMIC import create_attentionMIC_SGE_model
from model.attentionMIC.efficient_channel_attentionMIC import create_attentionMIC_ECA_model
from model.attentionMIC.time_frequency_attentionMIC import (create_attentionMIC_TFQ_model, TimeFreqAttentionLayer,
                                                            TimeFreqAttentionQuantizeConfig)
from model.attentionMIC.attentionMIC import create_attentionMIC_model, create_attentionMIC_multihead_model
from model.attentionMIC.no_attention import create_no_attentionMIC_model
from model.attentionMIC.ECA_TFQ_hybrid import create_attentionMIC_ECA_TFQ_hybrid_model
from model.attentionMIC.new_models import *

from model.metrics import MultiLabelAccuracy
from model.metrics import MultiLabelPrecision
from model.metrics import MultiLabelRecall
from model.metrics import MultiLabelF1Score
from model.metrics import MultiLabelInformedness
from model.metrics import MultiLabelMarkedness
from model.metrics import MultiLabelMCC
from model.metrics import MultiLabelCohenKappa

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint


class MICType(Enum):
    NONE = 0
    ECA = 1
    SGE = 2
    TFQ = 3
    NONE_NO_ATTENTION = 4
    TFQ_ECA_HYBRID = 5
    NONE_MULTIHEAD = 6
    RESNET = 7
    COMBINATION_1D_2D = 8
    CRNN = 9
    CRNN_BIDIRECTIONAL = 10
    DILATED_CONV = 11
    INCEPTION = 12
    MULTI_SCALE = 13
    SQUEEZE_EXCITATION = 14
    TEMPORAL_CONV = 15
    DEPTHWISE_SEPARABLE = 16
    RESNET_MOBILENET = 17


class TrainMICModel:
    def __init__(self, MIC_type: MICType, generator, validation_generator, log_dir, checkpoint_dir, classes_names,
                 load_model_path="", verbose=1, class_weights=None, audio_bits=None, quantization_weights=None,
                 learning_rate=0.001, pretrain_model=None, early_stopping_patience=10, prune=False,
                 quantize=False, flexible_filters=None, other_param=None):
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
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = tf.constant([class_weights[i] for i in range(len(class_weights))], dtype=tf.float32)
        self.lr = learning_rate
        self.pretrained_model = pretrain_model
        self.early_stopping_patience = early_stopping_patience
        self.prune = prune
        self.quantize = quantize
        self.quantized_model = None
        self.flexible_filters = flexible_filters
        self.other_param = other_param

    def __call__(self, epochs, *args, **kwargs):
        self.train(epochs)
        return self.get_model(), self.get_quantized_model()

    def _weighted_binary_crossentropy(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon(), 1 - epsilon())
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * self.class_weights + (1 - y_true) * (1 - self.class_weights)
        weighted_bce = weight_vector * bce
        print(tf.reduce_mean(weighted_bce))
        return tf.reduce_mean(weighted_bce)

    def _transfer_weights(self):
        pretrain_layers = [layer for layer in self.pretrained_model.layers if
                           isinstance(layer, (Conv2D, BatchNormalization))]
        main_layers = [layer for layer in self.model.layers if isinstance(layer, (Conv2D, BatchNormalization))]

        # Transfer weights
        for pre_layer, main_layer in zip(pretrain_layers, main_layers):
            main_layer.set_weights(pre_layer.get_weights())

    def _verify_weights_transfer(self):
        pretrain_layers = [layer for layer in self.pretrained_model.layers if
                           isinstance(layer, (Conv2D, BatchNormalization))]
        main_layers = [layer for layer in self.model.layers if isinstance(layer, (Conv2D, BatchNormalization))]

        if len(pretrain_layers) != len(main_layers):
            raise ValueError("Mismatch in the number of Conv2D/BatchNormalization layers between the models.")

        for i, (pre_layer, main_layer) in enumerate(zip(pretrain_layers, main_layers)):
            pre_weights = pre_layer.get_weights()
            main_weights = main_layer.get_weights()

            print(f"Layer {i} - {pre_layer.name} weights:")
            print("Pretrained weights:")
            for weight in pre_weights:
                print(weight)

            print("Main model weights after transfer:")
            for weight in main_weights:
                print(weight)

            # Check if weights match
            for pre_w, main_w in zip(pre_weights, main_weights):
                if not (pre_w == main_w).all():
                    print(f"Warning: Weights do not match for layer {i} - {pre_layer.name}")
                else:
                    print(f"Weights match for layer {i} - {pre_layer.name}")


    def train(self, epochs=30):

        metrics = [MultiLabelAccuracy(self.num_outputs, self.classes_names),
                   MultiLabelPrecision(self.num_outputs, self.classes_names),
                   MultiLabelRecall(self.num_outputs, self.classes_names),
                   MultiLabelF1Score(self.num_outputs, self.classes_names),
                   MultiLabelInformedness(self.num_outputs, self.classes_names),
                   MultiLabelMarkedness(self.num_outputs, self.classes_names),
                   MultiLabelMCC(self.num_outputs, self.classes_names),
                   MultiLabelCohenKappa(self.num_outputs, self.classes_names),
                   AUC(curve="PR", multi_label=True, name="AUC_PR"),
                   AUC(curve="ROC", multi_label=True, name="AUC_ROC")]

        if not self._load_model():
            input_shape = (*self.generator.data_shape, 1)
            t = self.MIC_type
            if self.flexible_filters is None:
                self.flexible_filters = 64

            if t == MICType.ECA:
                self.model = create_attentionMIC_ECA_model(input_shape, self.num_outputs,
                                                           filters=self.flexible_filters)
            elif t == MICType.SGE:
                """
                There's 64 filters so num_groups can be:
                1, 2, 4, 8, 16, 32, 64
                """
                num_groups = 8
                self.model = create_attentionMIC_SGE_model(input_shape, self.num_outputs, num_groups,
                                                           filters=self.flexible_filters)
            elif t == MICType.TFQ:
                self.model = create_attentionMIC_TFQ_model(input_shape, self.num_outputs, self.prune,
                                                           quantize=self.quantize, filters=self.flexible_filters)
            elif t == MICType.NONE_NO_ATTENTION:
                self.model = create_no_attentionMIC_model(input_shape, self.num_outputs, filters=self.flexible_filters)
            elif t == MICType.TFQ_ECA_HYBRID:
                self.model = create_attentionMIC_ECA_TFQ_hybrid_model(input_shape, self.num_outputs,
                                                                      filters=self.flexible_filters)
            elif t == MICType.NONE_MULTIHEAD:
                self.model = create_attentionMIC_multihead_model(input_shape, self.num_outputs, filters=self.flexible_filters,
                                                                 head_size=32, num_heads=8)
            elif t == MICType.RESNET:
                self.model = build_resnet_model(input_shape, self.num_outputs, filters=self.flexible_filters)
            elif t == MICType.COMBINATION_1D_2D:
                self.model = build_combined_conv_model(input_shape, self.num_outputs, filters1d=self.flexible_filters,
                                                       filters2d=self.other_param)
            elif t == MICType.CRNN:
                self.model = build_crnn_model(input_shape, self.num_outputs, filters=self.flexible_filters,
                                              lstm_units=self.other_param)
            elif t == MICType.CRNN_BIDIRECTIONAL:
                self.model = build_crnn_bidirectional_model(input_shape, self.num_outputs,
                                                            filters=self.flexible_filters, lstm_units=self.other_param)
            elif t == MICType.DEPTHWISE_SEPARABLE:
                self.model = build_mobilenet_style_model(input_shape, self.num_outputs, filters=self.flexible_filters)
            elif t == MICType.DILATED_CONV:
                self.model = build_dilated_conv_model(input_shape, self.num_outputs, filters=self.flexible_filters)
            elif t == MICType.INCEPTION:
                self.model = build_inception_model(input_shape, self.num_outputs, filters1=self.flexible_filters,
                                                   filters2=self.other_param)
            elif t == MICType.MULTI_SCALE:
                self.model = build_multi_scale_cnn_model(input_shape, self.num_outputs, filters=self.flexible_filters)
            elif t == MICType.SQUEEZE_EXCITATION:
                self.model = build_se_model(input_shape, self.num_outputs, filters=self.flexible_filters)
            elif t == MICType.TEMPORAL_CONV:
                self.model = build_tcn_model(input_shape, self.num_outputs, filters=self.flexible_filters)
            elif t == MICType.RESNET_MOBILENET:
                self.model = build_mobilenet_resnet_style_model(input_shape, self.num_outputs,
                                                                filters=self.flexible_filters)
            elif t == MICType.NONE:
                self.model = create_attentionMIC_model(input_shape, self.num_outputs,
                                                       filters=self.flexible_filters)
            else:
                raise Exception(f"Something went wrong MICType is wrong, MICType is: {t}")

            if self.pretrained_model is not None:
                self._transfer_weights()

            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                               loss="binary_crossentropy",
                               metrics=metrics)

        mic_type_str = self.MIC_type.name
        if self.verbose:
            self.model.summary()
            plot_model(self.model, to_file=f'./model_architecture_picture/model_{mic_type_str}_architecture_vertical.png')

            plot_model(self.model, to_file=f'./model_architecture_picture/model_{mic_type_str}_architecture_horizontal.png',
                       rankdir='LR')

            if self.pretrained_model is not None:
                self._verify_weights_transfer()

        model_checkpoint_callback = self._set_checkpoint_callback()
        tensorboard_callback = self._set_tensorboard_callback(histogram_freq=1, update_freq="epoch")
        csv_logger_callback = CSVLoggerCallback(self.log_dir)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.early_stopping_patience,
                                       restore_best_weights=True, mode="min")

        wandb_callback = WandbMetricsLogger()
        wandb_checkpoint = WandbModelCheckpoint("models")
        pruning = tfmot.sparsity.keras.UpdatePruningStep()

        callbacks = [model_checkpoint_callback, tensorboard_callback, csv_logger_callback, wandb_callback,
                     early_stopping]

        if self.prune:
            callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

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
            if not self.quantize:
                return

        if self.prune and self.quantize:
            quantize_scope = tfmot.quantization.keras.quantize_scope
            custom_objects = {"TimeFreqAttentionLayer": TimeFreqAttentionLayer,
                              "TimeFreqAttentionQuantizeConfig": TimeFreqAttentionQuantizeConfig}
            with quantize_scope(custom_objects):
                quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
                    self.model)
                q_aware_model = tfmot.quantization.keras.quantize_apply(
                    quant_aware_annotate_model,
                    tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme())
            q_aware_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                                  loss="binary_crossentropy",
                                  metrics=metrics)

        elif self.quantize and not self.prune:
            quantize_model = tfmot.quantization.keras.quantize_model
            quantize_scope = tfmot.quantization.keras.quantize_scope
            custom_objects = {"TimeFreqAttentionLayer": TimeFreqAttentionLayer,
                              "TimeFreqAttentionQuantizeConfig": TimeFreqAttentionQuantizeConfig}
            with quantize_scope(custom_objects):
                q_aware_model = quantize_model(self.model)
            q_aware_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                                  loss="binary_crossentropy",
                                  metrics=metrics)

        if self.quantize:
            callbacks = []
            if self.prune:
                callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

            q_aware_model.summary()
            q_aware_model.fit(self.generator,
                              epochs=1,
                              steps_per_epoch=len(self.generator),
                              verbose=1,
                              use_multiprocessing=False,
                              workers=4,
                              callbacks=callbacks,
                              validation_data=self.val_generator,
                              validation_steps=len(self.val_generator))
            converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            self.quantized_model = converter.convert()

            self.model = q_aware_model
            if self.prune and self.load_model_path == "":
                return



        # self.model.save('model_no_pruning.h5')
        # file_size = os.path.getsize('model_no_pruning.h5')
        # print(f"Rozmiar modelu na dysku: {file_size} bajtów")

    def get_model(self):
        return self.model

    def get_quantized_model(self):
        return self.quantized_model

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

    def _save_and_convert_pruned_model(self):
        # Strip pruning before saving the model
        self.model = strip_pruning(self.model)
        mode = 'test2'

        pruned_model_path = 'my_pruned_model_' + mode
        self.model.save(pruned_model_path, save_format='tf')

        pruned_model_h5_path = 'my_pruned_model_' + mode + '.h5'
        self.model.save(pruned_model_h5_path)  # Saves as a .h5 file

        converter = tf.lite.TFLiteConverter.from_saved_model(pruned_model_path)
        converter.experimental_enable_resource_variables = True
        converter.experimental_new_converter = True
        tflite_model = converter.convert()

        # Save the TFLite model to a file
        tflite_model_path = 'my_model' + mode + '.tflite'
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)

        # Print the size of the saved pruned model
        file_size = os.path.getsize(pruned_model_h5_path)
        print(f"Rozmiar pruned modelu na dysku: {file_size} bajtów")

        # Print the size of the TFLite model
        tflite_file_size = os.path.getsize(tflite_model_path)
        print(f"Rozmiar TFLite modelu na dysku: {tflite_file_size} bajtów")
