import numpy as np
import os
import json
import time

import pathlib
from generator import AudioGenerator
from sklearn.preprocessing import MultiLabelBinarizer

from datasets.cut_dataset.load_cut_files import load_cut_files
from datasets.data_loaders.data_splitter import split_data
from model.metrics import *
from model.attentionMIC.time_frequency_attentionMIC import TimeFreqAttentionLayer

from training.train_conv import TrainConvModel
from training.train_MIC import TrainMICModel, MICType
from training.pretrain_model import PretrainMICModel
from testing.test_conv import TestConvModel
from testing.test_MIC import TestMICModel
from testing.test_MIC_quantized import TestMICQuantizedModel

from tensorflow.keras.models import load_model



import itertools
import datetime
from matplotlib import pyplot as plt

import wandb

import tensorflow as tf
import librosa


"""
TODO:
MusicNet is not 100% automated. If you download the data and unpack the tar.gz file using the class you need to merge
the train and test folders together by hand.


Wskaźniki:

    Micro-averaging – sklearn.metrics.f1_score(average=’micro’),
    Macro-averaging – sklearn.metrics.f1_score(average=’macro’),
    Hamming-Loss – sklearn.metrics.hamming_loss
    Jaccard similarity coefficient – sklearn.metrics.jaccard_similarity_score
    AUC – sklearn.metrics.roc_auc_score
    Exact Match Ratio – sklearn.metrics.accuracy_score



Musicnet:
PCM-encoded




PLANNED CHANGES FOR OSION:

Main goal - make the model as most online as possible - eliminate the time necessary for inference.
Additionally remove as much memory requisite as possible.

1. Adaptation of models to a new genre: i.e. instead of violin, viola etc. use stringed instruments
2. Change some blocks for DWConv
3. Change the amounts of bits necessary for weights (test decreasing before training and after training)
4. Change the Sampling rate of the original signal to decrease the mfcc size
5. Pruning

"""


def file_loader(files_path, is_string=False):
    filenames_train, labels_train = load_cut_files(pathlib.Path(files_path).absolute(), is_string)

    return filenames_train, labels_train


def return_dirs(log_base, model_base, extension, current_time):
    log_dir = os.path.join(log_base, extension, current_time)
    os.makedirs(log_dir, exist_ok=True)
    model_base = os.path.join(model_base, extension, "_")
    return log_dir, model_base


def save_data_to_json(json_path, dictionary):
    if os.path.exists(json_path):
        with open(json_path, "r") as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = {}

    existing_data.update(dictionary)

    with open(json_path, "w") as json_file:
        json.dump(existing_data, json_file, indent=4)

    print(f"Global mean and standard deviation updated in {json_path}")


def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model


def save_tflite_model(tflite_model, filename="model.tflite"):
    """
    Saves a TensorFlow Lite model to a file.

    Args:
    tflite_model: The converted TFLite model in bytes.
    filename: The name of the file to save the model to.
    """
    with open(filename, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved as {filename}")


def start_training(*, model_type, extension, epoch_num, dt, mlb, model_path="", class_weights=None, batch_size=32,
                   pretraining=False, pretraining_dt=None, learning_rate=0.001, augment=False, pretrain_lr=0.01,
                   pretraining_mlb=None, epoch_pretrain=30, early_stopping_patience=10, prune=False,
                   quantize=False, feature_type="mfcc", pooling_size=None, filters=None, n_chroma=12):
    # Training set up---------------------------------------------------------------------------------------------------
    if not isinstance(batch_size, int):
        raise Exception(f"Batch size is not an int, batch_size: {batch_size}")

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir, model_dir = return_dirs("./model/logs/", "./model/saved_model/", extension, current_time)
    multi_label = False
    if model_type == "CONV":
        multi_label = True

    pooling = False
    if pooling_size is not None:
        pooling = True

    generator = AudioGenerator(dt.x_train, dt.y_train, batch_size, feature_type=feature_type, multi_label=multi_label,
                               classes_names=mlb.classes_, testing=False, pool_spectrogram=pooling,
                               pool_size=pooling_size, n_chroma=n_chroma)

    print(f"Batch size is: {batch_size}")

    global_mean = generator.train_mean
    global_std = generator.train_std

    config_data = {
        "global_mean": global_mean.tolist() if hasattr(global_mean, 'tolist') else global_mean,
        "global_std": global_std.tolist() if hasattr(global_std, 'tolist') else global_std
    }

    save_data_to_json('json_file.json', config_data)

    val_generator = AudioGenerator(dt.x_val, dt.y_val, batch_size, feature_type=feature_type, multi_label=multi_label,
                                   classes_names=mlb.classes_, testing=True, train_mean=global_mean,
                                   train_std=global_std, pool_spectrogram=pooling, pool_size=pooling_size,
                                   n_chroma=n_chroma)
    testing_generator = AudioGenerator(dt.x_test, dt.y_test, batch_size, feature_type=feature_type,
                                       multi_label=multi_label, classes_names=mlb.classes_, testing=True,
                                       train_mean=global_mean, train_std=global_std, pool_spectrogram=pooling,
                                       pool_size=pooling_size, n_chroma=n_chroma)

    pretrain_model = None
    if pretraining:
        pretraining_generator = AudioGenerator(pretraining_dt.x_train, pretraining_dt.y_train, batch_size, use_mfcc=True,
                                               multi_label=multi_label, classes_names=pretraining_mlb.classes_,
                                               testing=False)

        pretrain_model = PretrainMICModel(pretraining_generator, verbose=1, learning_rate=pretrain_lr,
                                          augment=augment, classes_names=pretraining_mlb.classes_)(epoch_pretrain)

    if isinstance(filters, tuple):
        filters, other_param = filters
    else:
        other_param = None

    if model_type == "ECA":
        model, quantized = (TrainMICModel(MICType.ECA, generator, val_generator, log_dir, model_dir, mlb.classes_, model_path,
                                          class_weights=class_weights, learning_rate=learning_rate,
                                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "SGE":
        model, quantized = (TrainMICModel(MICType.SGE, generator, val_generator, log_dir, model_dir, mlb.classes_,model_path,
                               class_weights=class_weights, learning_rate=learning_rate,
                               pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "TFQ":
        model, quantized = (TrainMICModel(MICType.TFQ, generator, val_generator, log_dir, model_dir, mlb.classes_, model_path,
                               class_weights=class_weights, learning_rate=learning_rate,
                               pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                               prune=prune, quantize=quantize,
                                          flexible_filters=filters)(epoch_num))

        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir, on_cpu=True)()
        if quantized:
            TestMICQuantizedModel(quantized, testing_generator, mlb.classes_, log_dir=log_dir)()

        model_converted = convert_to_tflite(model)
        save_tflite_model(model_converted, f'model_normal{"_pruned" if prune else ""}.tflite')
        if quantized:
            save_tflite_model(quantized, f'model_quantized{"_pruned" if prune else ""}.tflite')
            quantized_size = os.path.getsize(f'model_quantized{"_pruned" if prune else ""}.tflite')
            print(f"Size of quantized model: {quantized_size}")
        normal_size = os.path.getsize(f'model_normal{"_pruned" if prune else ""}.tflite')
        print(f"Size of normal model: {normal_size}")

    elif model_type == "NONE":
        model, quantized = (TrainMICModel(MICType.NONE, generator, val_generator, log_dir, model_dir, mlb.classes_, model_path,
                               class_weights=class_weights, learning_rate=learning_rate,
                               pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "NONE_NO_ATTENTION":
        model, quantized = (TrainMICModel(MICType.NONE_NO_ATTENTION, generator, val_generator, log_dir, model_dir, mlb.classes_, model_path,
                               class_weights=class_weights, learning_rate=learning_rate,
                               pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "TFQ_ECA_HYBRID":
        model, quantized = (TrainMICModel(MICType.TFQ_ECA_HYBRID, generator, val_generator, log_dir, model_dir, mlb.classes_,
                               model_path,
                               class_weights=class_weights, learning_rate=learning_rate,
                               pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "CONV":
        model = (TrainConvModel(generator, val_generator, log_dir, model_dir, mlb.classes_, model_path,
                                class_weights=class_weights, learning_rate=learning_rate,
                                pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                                flexible_filters=filters)(epoch_num))
        TestConvModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "NONE_MULTIHEAD":
        model, quantized = (
            TrainMICModel(MICType.NONE_MULTIHEAD, generator, val_generator, log_dir, model_dir, mlb.classes_, model_path,
                          class_weights=class_weights, learning_rate=learning_rate,
                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "RESNET":
        model, quantized = (
            TrainMICModel(MICType.RESNET, generator, val_generator, log_dir, model_dir, mlb.classes_,
                          model_path,
                          class_weights=class_weights, learning_rate=learning_rate,
                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()

    elif model_type == "COMBINATION_1D_2D":
        model, quantized = (
            TrainMICModel(MICType.COMBINATION_1D_2D, generator, val_generator, log_dir, model_dir, mlb.classes_,
                          model_path,
                          class_weights=class_weights, learning_rate=learning_rate,
                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                          flexible_filters=filters, other_param=other_param)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "CRNN":
        model, quantized = (
            TrainMICModel(MICType.CRNN, generator, val_generator, log_dir, model_dir, mlb.classes_,
                          model_path,
                          class_weights=class_weights, learning_rate=learning_rate,
                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                          flexible_filters=filters, other_param=other_param)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "CRNN_BIDIRECTIONAL":
        model, quantized = (
            TrainMICModel(MICType.CRNN_BIDIRECTIONAL, generator, val_generator, log_dir, model_dir, mlb.classes_,
                          model_path,
                          class_weights=class_weights, learning_rate=learning_rate,
                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                          flexible_filters=filters, other_param=other_param)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "DEPTHWISE_SEPARABLE":
        model, quantized = (
            TrainMICModel(MICType.DEPTHWISE_SEPARABLE, generator, val_generator, log_dir, model_dir, mlb.classes_,
                          model_path,
                          class_weights=class_weights, learning_rate=learning_rate,
                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "DILATED_CONV":
        model, quantized = (
            TrainMICModel(MICType.DILATED_CONV, generator, val_generator, log_dir, model_dir, mlb.classes_,
                          model_path,
                          class_weights=class_weights, learning_rate=learning_rate,
                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "INCEPTION":
        model, quantized = (
            TrainMICModel(MICType.INCEPTION, generator, val_generator, log_dir, model_dir, mlb.classes_,
                          model_path,
                          class_weights=class_weights, learning_rate=learning_rate,
                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                          flexible_filters=filters, other_param=other_param)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "MULTI_SCALE":
        model, quantized = (
            TrainMICModel(MICType.MULTI_SCALE, generator, val_generator, log_dir, model_dir, mlb.classes_,
                          model_path,
                          class_weights=class_weights, learning_rate=learning_rate,
                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "SQUEEZE_EXCITATION":
        model, quantized = (
            TrainMICModel(MICType.SQUEEZE_EXCITATION, generator, val_generator, log_dir, model_dir, mlb.classes_,
                          model_path,
                          class_weights=class_weights, learning_rate=learning_rate,
                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == "TEMPORAL_CONV":
        model, quantized = (
            TrainMICModel(MICType.TEMPORAL_CONV, generator, val_generator, log_dir, model_dir, mlb.classes_,
                          model_path,
                          class_weights=class_weights, learning_rate=learning_rate,
                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif model_type == 'RESNET_MOBILENET':
        model, quantized = (
            TrainMICModel(MICType.RESNET_MOBILENET, generator, val_generator, log_dir, model_dir, mlb.classes_,
                          model_path,
                          class_weights=class_weights, learning_rate=learning_rate,
                          pretrain_model=pretrain_model, early_stopping_patience=early_stopping_patience,
                          flexible_filters=filters)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    else:
        raise Exception("Wrong mode")

    return model, generator


def just_test(model_path, test_x, test_y, batches, config_json_path, mlb):

    with open(config_json_path, "r") as json_file:
        config_data = json.load(json_file)

    global_mean = np.array(config_data.get("global_mean", None))  # Replace None with a default value if the key is missing
    global_std = np.array(config_data.get("global_std", None))
    if global_mean is None or global_std is None:
        raise Exception("There is no correct generator config given")

    batch_size = int(test_x.shape[0] / batches)

    multi_label = False  # True if CONV

    generator = AudioGenerator(test_x, test_y, batch_size, use_mfcc=True, multi_label=multi_label,
                                       classes_names=mlb.classes_, testing=True,
                                       train_mean=global_mean, train_std=global_std)


    classes_num = len(mlb.classes_)
    model = load_model(model_path,
                       custom_objects={'MultiLabelAccuracy': MultiLabelAccuracy(classes_num, mlb.classes_),
                                       'MultiLabelPrecision': MultiLabelPrecision(classes_num, mlb.classes_),
                                       'MultiLabelRecall': MultiLabelRecall(classes_num, mlb.classes_),
                                       'MultiLabelF1Score': MultiLabelF1Score(classes_num, mlb.classes_),
                                       'MultiLabelInformedness': MultiLabelInformedness(classes_num, mlb.classes_),
                                       'MultiLabelMarkedness': MultiLabelMarkedness(classes_num, mlb.classes_),
                                       'MultiLabelMCC': MultiLabelMCC(classes_num, mlb.classes_),
                                       'MultiLabelCohenKappa': MultiLabelCohenKappa(classes_num, mlb.classes_),
                                       'TimeFreqAttentionLayer': TimeFreqAttentionLayer(filters=64)})
    model.summary()

    TestMICModel(model, generator, mlb.classes_)()
    print(f"Model loaded from filepath: {os.path.abspath(model_path)}")


def calculate_class_weights(y):
    class_weights = {}
    # Calculate total number of samples
    total_samples = y.shape[0]

    # Calculate weights for each class
    for i in range(y.shape[1]):  # assuming y is shape (samples, classes)
        # Number of positive examples for the class
        class_sum = np.sum(y[:, i])
        # Weight for class i
        class_weights[i] = (total_samples / (y.shape[1] * class_sum))

    return class_weights


def representative_data_gen(generator, num_samples=100):
    for _ in range(num_samples):
        x, _ = next(generator)
        for i in range(x.shape[0]):
            yield [x[i:i+1, ...]]


def calculate_normalized_class_weights(dt):
    class_totals = np.sum(dt.y_train, axis=0)  # How many samples is there per each class
    total_samples = dt.y_train.shape[0]  # Total number of samples
    class_percentages = (class_totals / total_samples) * 100  # How many samples is there per each class in percent
    # Calculate the weights for each class (the more samples the less the weight)
    class_weights = calculate_class_weights(dt.y_train)
    max_weight = max(class_weights.values())  # Maximum weight
    min_allowed_weight = 0.2
    # Normalize weights from 0 to 1 taking the minimum weight into consideration
    normalized_class_weights = {k: max(v / max_weight, min_allowed_weight) for k, v in class_weights.items()}
    return normalized_class_weights



class Dataset:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test


class Config:
    def __init__(self):
        # Model configurations
        """
        Model types
        ECA, SGE, TFQ, NONE, NONE_NO_ATTENTION, CONV, TFQ_ECA_HYBRID, NONE_MULTIHEAD, RESNET

        New Models
        RESNET, COMBINATION_1D_2D, CRNN, DEPTHWISE_SEPARABLE, DILATED_CONV, INCEPTION, MULTI_SCALE, SQUEEZE_EXCITATION,
        TEMPORAL_CONV

        'ECA', 'SGE', 'TFQ', 'NONE', 'NONE_NO_ATTENTION', 'TFQ_ECA_HYBRID', 'NONE_MULTIHEAD', 'COMBINATION_1D_2D',
        'RESNET', 'CRNN', 'CRNN_BIDIRECTIONAL', 'DEPTHWISE_SEPARABLE', 'DILATED_CONV', 'INCEPTION', 'MULTI_SCALE',
        'SQUEEZE_EXCITATION', 'TEMPORAL_CONV'
        """

        #TODO In here didnt do COMBINATION_1D_2D yet
        self.model_types = ['NONE_MULTIHEAD',
                            'COMBINATION_1D_2D', 'RESNET', 'CRNN', 'CRNN_BIDIRECTIONAL', 'DEPTHWISE_SEPARABLE',
                            'DILATED_CONV', 'INCEPTION', 'MULTI_SCALE', 'SQUEEZE_EXCITATION', 'TEMPORAL_CONV', 'TFQ',
                            'TFQ_ECA_HYBRID']

        self.current_model = self.model_types[0]
        self.use_scheduler = True  # False for testing the model with 1 epoch
        self.just_test = False

        # Hyperparameters
        self.extension = os.path.join("NEW_TEST", f"{self.model_types[0]}_testing")
        self.batch_sizes = [16, 32, 48, 64, 80]
        self.audio_bits = [8, 16, 32]
        self.quantization_weights = [8, 16, 32]
        self.learning_rates = [0.001]
        self.audio_length = [2.5]
        self.overlap = [0]
        # Selected hyperparameters
        self.learning_rate = 0.001
        self.audio_bit = 16
        self.batch_size = 32
        self.batch_size_adaptable = True
        self.early_stopping_patience = 10
        self.prune = False  # Works just for TFQ
        self.quantize = False  # Works just for TFQ
        self.feature_type = ["mfcc"]
        self.pooling_size = None

        self.flexible_filters = [True]
        #TODO: CHANGED TO MAX FOR NOW
        #todo: IDEA USE THE NORMAL BASE BUT CHANGE CONV TYPE TO LIKE THE ONE THAT WORKS BY LAYER OR OTHERS
        self.flexible_filters_type = ['min']

        self.epoch_num = 200

        # Scheduler parameters
        # Best parameters - scheduler (Cosine Decay): 0.01 - alpha, 0.01 - initial learning rate, 0.98 - decay percent
        self.decay_percent = 0.15
        self.decay_percent_for_config = self.decay_percent
        self.initial_learning_rate = [0.001]
        self.initial_learning_rate_for_config = None
        self.alpha = [0.01]
        # Below are updated using self.update_scheduler_params(dt)
        self.lr_limit = 0.0
        self.steps_per_epoch = 0
        self.decay_epochs = 0
        self.decay_steps = 0

        # Generator configs
        self.n_mfcc = 40
        self.n_fft = 512
        self.n_chroma = [12, 24, 36]

        # Pretraining
        self.pretraining_flags = [False]
        self.pretraining_flag_for_config = False
        self.augment = False
        self.epoch_pretrain = 40
        self.use_genres = True
        self.pretrain_lr = 0.01

        # Dataset paths
        self.files_path = f"./datasets/datasets/musicnet/{self.audio_length[0]}s_len.{self.overlap[0]}s_overlap"
        self.pretraining_files_path = "./datasets/datasets/musan/4s_len.0s_overlap"

        # WandB configurations
        self.project_name = "magisterka-instrument-detection"
        self.entity_name = "magisterka-instrument-detection"
        self.wandb_group = "Final with tuned hiperparameters"
        self.wandb_job_type = "train"

        # Model checkpointing
        self.load_from_checkpoint = False
        self.model_load_path = "my_pruned_model_1_by_9.h5"
        if self.load_from_checkpoint is False:
            self.model_load_path = ""

    def update_scheduler_params(self, dt, initial_learning_rate, alpha):
        """Update scheduler-related parameters based on the current dataset and batch size."""
        self.steps_per_epoch = dt.x_train.shape[0] // self.batch_size
        self.decay_epochs = int(self.epoch_num * self.decay_percent)
        self.decay_steps = self.steps_per_epoch * self.decay_epochs
        self.lr_limit = alpha * initial_learning_rate

    def wandb_config(self):
        """Return a dictionary of parameters for WandB initialization."""
        config = {
            "epoch_num": self.epoch_num,
            "model": self.current_model,
            "audio_bits": self.audio_bit,
            "batch_size": self.batch_size,
            "pretraining_flag": self.pretraining_flag_for_config,
            "learning_rate": self.learning_rate,
            "augment": False
        }

        if self.use_scheduler:
            config.update({
                "decay_percent": self.decay_percent_for_config,
                "initial_learning_rate": self.initial_learning_rate_for_config,
                "lr_limit": self.lr_limit,
            })



        return config

    def get_all_hyperparameters_combinations(self, *args):
        for arg in args:
            if not hasattr(self, arg):
                raise ValueError(f"{arg} is not a valid attribute of Config class")

        # Retrieve the attributes and their names from the Config instance
        attributes = {arg: getattr(self, arg) for arg in args}

        # Generate the Cartesian product of the specified attributes
        combinations = itertools.product(*attributes.values())

        # Convert each combination to a dictionary
        for combination in combinations:
            yield dict(zip(attributes.keys(), combination))

    def set_file_path(self, audio_length, overlap):
        self.files_path = f"./datasets/datasets/musicnet/{audio_length}s_len.{overlap}s_overlap"

    def update_batch_size(self, x_data_size: int, second_shape_size: int = None):
        baseline_batch_size_1 = 771
        baseline_batch_size_2 = 40

        batch_size_1 = round(x_data_size / baseline_batch_size_1)
        # Adjust to the nearest lower multiple of 4
        batch_size_1 -= (batch_size_1 % 4)

        if second_shape_size is not None:
            scaling_factor = baseline_batch_size_2 / second_shape_size
            batch_size_2 = round(batch_size_1 * scaling_factor)

            # Ensure batch_size_2 is a multiple of 4
            batch_size_2 -= (batch_size_2 % 4)

            # Ensure the batch size remains positive and reasonable
            batch_size_2 = max(batch_size_2, 4)

            # Use the adjusted batch size
            batch_size = batch_size_2
        else:
            batch_size = batch_size_1

        self.batch_size = batch_size

    def calc_new_filters(self, model_type):

        if model_type == "ECA":
            filters = 52
        elif model_type == "SGE":
            filters = 61 - 61 % 8
        elif model_type == "TFQ":
            filters = 50
        elif model_type == "NONE":
            filters = 61
        elif model_type == "NONE_NO_ATTENTION":
            filters = 64
        elif model_type == "TFQ_ECA_HYBRID":
            filters = 42
        elif model_type == 'NONE_MULTIHEAD':
            filters = 42
        elif model_type == "RESNET":
            filters = 38
        elif model_type == "COMBINATION_1D_2D":
            filters = (32, 20)
        elif model_type == "CRNN_BIDIRECTIONAL":
            filters = (24, 10)
        elif model_type == "CRNN":
            filters = (32, 14)
        elif model_type == "DEPTHWISE_SEPARABLE":
            filters = 112
        elif model_type == "DILATED_CONV":
            filters = 64
        elif model_type == "INCEPTION":
            filters = (35, 25)
        elif model_type == "MULTI_SCALE":
            filters = 27
        elif model_type == "SQUEEZE_EXCITATION":
            filters = 64
        elif model_type == "TEMPORAL_CONV":
            filters = 54
        elif model_type == "RESNET_MOBILENET":
            filters = 104
        else:
            raise Exception("Model doesn't support custom filter sizes")
        return filters

    def calc_new_filters_max(self, model_type):
        if model_type == "ECA":
            filters = 73
        elif model_type == "SGE":
            filters = 84 - 84 % 8 + 8
        elif model_type == "TFQ":
            filters = 74
        elif model_type == "NONE":
            filters = 86
        elif model_type == "NONE_NO_ATTENTION":
            filters = 90
        elif model_type == "TFQ_ECA_HYBRID":
            filters = 64
        elif model_type == 'NONE_MULTIHEAD':
            filters = 68
        elif model_type == "RESNET":
            filters = 52
        elif model_type == "COMBINATION_1D_2D":
            filters = (42, 28)
        elif model_type == "CRNN_BIDIRECTIONAL":
            filters = (32, 14)
        elif model_type == "CRNN":
            filters = (46, 18)
        elif model_type == "DEPTHWISE_SEPARABLE":
            filters = 160
        elif model_type == "DILATED_CONV":
            filters = 90
        elif model_type == "INCEPTION":
            filters = (52, 32)
        elif model_type == "MULTI_SCALE":
            filters = 36
        elif model_type == "SQUEEZE_EXCITATION":
            filters = 88
        elif model_type == "TEMPORAL_CONV":
            filters = 74
        elif model_type == "RESNET_MOBILENET":
            filters = 146
        else:
            raise Exception("Model doesn't support custom filter sizes")
        return filters


def main():
    # Uncomment if you want to run model on cpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    config = Config()
    combinations = config.get_all_hyperparameters_combinations("model_types", "learning_rates",
                                                               "initial_learning_rate", "alpha", "pretraining_flags",
                                                               "audio_length", "overlap", "feature_type",
                                                               "flexible_filters", "flexible_filters_type")

    start_time = time.time()
    filenames, labels = file_loader(config.files_path, False)
    mlb = MultiLabelBinarizer()

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(filenames, labels, mlb, 0.1, 0.1, False)



    end_time = time.time()
    print(f"Split_data time: {end_time - start_time}s")

    for combination in combinations:
        model_type = combination.get('model_types', config.model_types[0])
        learning_rate = combination.get('learning_rates', config.learning_rates[0])
        config.learning_rate = learning_rate
        pretraining_flag = combination.get('pretraining_flags', config.pretraining_flags[0])
        config.pretraining_flag_for_config = pretraining_flag

        initial_learning_rate = combination.get('initial_learning_rate', config.initial_learning_rate[0])
        config.initial_learning_rate_for_config = initial_learning_rate
        alpha = combination.get('alpha', config.alpha[0])  # Default to 0.01 or any sensible value
        audio_length = combination.get('audio_length', config.audio_length[0])
        overlap = combination.get('overlap', config.overlap[0])
        config.set_file_path(audio_length, overlap)

        feature_type = combination.get('feature_type', config.feature_type[0])
        flexible_filters = combination.get('flexible_filters', config.flexible_filters[0])
        flexible_filters_type = combination.get('flexible_filters_type', config.flexible_filters_type[0])
        n_chroma = combination.get('n_chroma', config.n_chroma[0])

        # Temporary disable some trainings
        # if flexible_filters is False and feature_type == "mfcc":
        #     continue

        # if flexible_filters is True:
        #     config.wandb_group = "Changing filters"
        # else:
        #     config.wandb_group = "Training chromagram"

        # if n_chroma != config.n_chroma[0] and feature_type == "mfcc":
        #     continue


        "HYPERPARAMETERS ---------------------------------------------------------------------------------------------"
        config.extension = os.path.join(f"NEW_TEST", f"{model_type}_testing")
        config.current_model = model_type
        if config.use_scheduler:
            print(f"-------------- NOW TRAINING MODEL: {model_type} WITH LR: Scheduler ----------------------\n")
        else:
            print(f"-------------- NOW TRAINING MODEL: {model_type} WITH LR: {learning_rate} -------------------\n")

        print(f"Config - flexible_filters: {flexible_filters}, feature_type: {feature_type},"
              f" wandb_name: {config.wandb_group}, n_chroma: {n_chroma}")


        dt = Dataset(x_train, y_train, x_val, y_val, x_test, y_test)

        if config.batch_size_adaptable is True:
            if feature_type == "mfcc":
                config.update_batch_size(dt.x_train.shape[0], config.n_mfcc)
            elif feature_type == "spectrogram":
                config.update_batch_size(dt.x_train.shape[0], config.n_fft+1)
            elif feature_type == "chromagram":
                config.update_batch_size(dt.x_train.shape[0], n_chroma)
            else:
                raise Exception(f"Wrong feature type {feature_type}")

        one_batch_x = x_test[:config.batch_size]
        one_batch_y = y_test[:config.batch_size]

        if flexible_filters:
            if flexible_filters_type == 'min':
                filters = config.calc_new_filters(model_type)
            elif flexible_filters_type == 'max':
                filters = config.calc_new_filters_max(model_type)
            else:
                raise Exception("Wrong filter change mode")
        else:
            filters = 64



        if config.just_test:
            just_test('my_model.h5', one_batch_x, one_batch_y, 1, "json_file.json", mlb)
            return

        if config.use_scheduler:
            config.update_scheduler_params(dt, initial_learning_rate, alpha)
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=config.decay_steps,
                alpha=alpha
            )

        if pretraining_flag:
            pretraining_filenames, pretraining_labels = file_loader(config.pretraining_files_path, True)
            pretraining_mlb = MultiLabelBinarizer()
            x_pre_train, y_pre_train, x_pre_val, y_pre_val, x_pre_test, y_pre_test = split_data(
                pretraining_filenames, pretraining_labels, pretraining_mlb, 0.0, 0.0, True)
            pretraining_dt = Dataset(x_pre_train, y_pre_train, x_pre_val, y_pre_val, x_pre_test, y_pre_test)
        else:
            pretraining_dt = None
            pretraining_mlb = None

        normalized_class_weights = calculate_normalized_class_weights(dt)

        wandb_lr_text = learning_rate if not config.use_scheduler else 'scheduler'
        if config.use_scheduler:
            name = (f"Model: {model_type}, Filters: {filters}")

            #name = f"testing_quantization"
        else:
            name = (f"Pretraining: {pretraining_flag}, Lr: {wandb_lr_text}, Model: {model_type}, "
                    f"Preprocessing type {feature_type}")

        wandb_config = config.wandb_config()
        wandb_config.update({'filters': filters})
        wandb.init(
            # set the wandb project where this run will be logged
            project=config.project_name,
            entity=config.entity_name,
            name=name,
            group=config.wandb_group,
            job_type=config.wandb_job_type,
            # track hyperparameters and run metadata
            config=wandb_config
        )

        try:
            model, generator = start_training(model_type=model_type, extension=config.extension,
                                              epoch_num=config.epoch_num, dt=dt, mlb=mlb,
                                              model_path=config.model_load_path, batch_size=config.batch_size,
                                              pretraining=pretraining_flag, pretraining_dt=pretraining_dt,
                                              pretraining_mlb=pretraining_mlb, learning_rate=learning_rate,
                                              augment=config.augment, pretrain_lr=config.pretrain_lr,
                                              epoch_pretrain=config.epoch_pretrain,
                                              early_stopping_patience=config.early_stopping_patience,
                                              prune=config.prune, quantize=config.quantize,
                                              feature_type=feature_type, filters=filters,
                                              n_chroma=n_chroma)
        except Exception as e:
            # Log the exception to wandb and continue with the next iteration
            wandb.log({"error": str(e)})
            print(f"An error occurred with batch size {config.batch_size} and audio bit {config.audio_bit}: {e}")
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
