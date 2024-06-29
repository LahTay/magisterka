import numpy as np
import os

import pathlib
from generator import AudioGenerator
from misc.instrument_number_map import instruments_map
from sklearn.preprocessing import MultiLabelBinarizer

from datasets.cut_dataset.load_cut_files import load_cut_files
from datasets.cut_dataset.cut_audio_file import main_audio_cutter

from model.conv_model.model_predict import predict_model
from training.train_conv import TrainConvModel
from training.train_MIC import TrainMICModel, MICType
from testing.test_conv import TestConvModel
from testing.test_MIC import TestMICModel

import datetime
from matplotlib import pyplot as plt

import wandb

import tensorflow_model_optimization as tfmot
import tensorflow as tf


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

"""


def extract_instruments(data, return_dict=False):
    """
    Extracts instruments from given data.

    Args:
        data (tuple): A tuple of two lists. The first list contains filenames, and the second list contains intervaltrees corresponding to each filename.
        return_dict (bool): If return_dict is True returns a dictionary of values where id of audio is its key,
                            if False returns a list in the same order as filenames list.

    Returns:
        dict: A dictionary with filenames as keys and sets of instruments as values.
    """
    if return_dict:
        instrument_dict = {}
    else:
        instrument_dict = []
    filenames, trees = data

    # Iterate through each file and its corresponding intervaltree
    for filepath in filenames:
        # Extract the filename without path and .wav suffix to match keys in `trees`
        filename = int(os.path.splitext(os.path.basename(filepath))[0])

        # Initialize an empty set for the instruments
        instruments = set()

        # Check if the filename exists in trees and extract instruments
        if filename in trees:
            for interval in trees[filename]:
                instruments.add(interval.data)  # Assuming interval.data holds the instrument

        if return_dict:
            instrument_dict[filename] = instruments
        else:
            instrument_dict.append(instruments)

    return instrument_dict


def file_cutter(root_path):
    segment_length = 4
    overlap_length = 0
    cut_files_folder_name = f"{segment_length}s_len.{overlap_length}s_overlap"
    main_audio_cutter(root_path, cut_files_folder_name, segment_length, overlap_length, None)


def file_loader(root_path, cut_files_folder_name):
    filenames_train, labels_train = load_cut_files(pathlib.Path(root_path).absolute() / cut_files_folder_name)

    return filenames_train, labels_train


def transform_labels_to_names(labels, instruments_map):
    """Transforms list of lists of numeric labels to list of lists of string labels."""
    transformed_labels = []
    for label_set in labels:
        transformed_labels.append([instruments_map[label] for label in label_set if label in instruments_map])
    return transformed_labels


from datasets.data_loaders.data_splitter import split_data


def return_dirs(log_base, model_base, extension, current_time):
    log_dir = os.path.join(log_base, extension, current_time)
    os.makedirs(log_dir, exist_ok=True)
    model_base = os.path.join(model_base, extension, "_")
    return log_dir, model_base


def start_training(type, extension, epoch_num, x_train, y_train, x_val, y_val, x_test, y_test, mlb, model_path="",
                   class_weights=None, audio_bits=None, quantization_weights=None, batch_size=32):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir, model_dir = return_dirs("./model/logs/", "./model/saved_model/", extension, current_time)
    multi_label = False
    if type == "CONV":
        multi_label = True

    generator = AudioGenerator(x_train, y_train, batch_size, use_mfcc=True, multi_label=multi_label,
                               classes_names=mlb.classes_, testing=False)
    global_mean = generator.train_mean
    global_std = generator.train_std
    val_generator = AudioGenerator(x_val, y_val, batch_size, use_mfcc=True, multi_label=multi_label,
                                   classes_names=mlb.classes_, testing=True,
                                   train_mean=global_mean, train_std=global_std)
    testing_generator = AudioGenerator(x_test, y_test, batch_size, use_mfcc=True, multi_label=multi_label,
                                       classes_names=mlb.classes_, testing=True,
                                       train_mean=global_mean, train_std=global_std)

    if type == "ECA":
        model = (TrainMICModel(MICType.ECA, generator, val_generator, log_dir, model_dir, mlb.classes_, model_path,
                               class_weights=class_weights)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif type == "SGE":
        model = (TrainMICModel(MICType.SGE, generator, val_generator, log_dir, model_dir, mlb.classes_,model_path,
                               class_weights=class_weights)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif type == "TFQ":
        model = (TrainMICModel(MICType.TFQ, generator, val_generator, log_dir, model_dir, mlb.classes_, model_path,
                               class_weights=class_weights, audio_bits=None, quantization_weights=None)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif type == "NONE":
        model = (TrainMICModel(MICType.NONE, generator, val_generator, log_dir, model_dir, mlb.classes_, model_path,
                               class_weights=class_weights)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif type == "NONE_NO_ATTENTION":
        model = (TrainMICModel(MICType.NONE_NO_ATTENTION, generator, val_generator, log_dir, model_dir, mlb.classes_, model_path,
                               class_weights=class_weights)(epoch_num))
        TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    elif type == "CONV":
        model = (TrainConvModel(generator, val_generator, log_dir, model_dir, mlb.classes_, model_path,
                                class_weights=class_weights)(epoch_num))
        TestConvModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()
    else:
        raise Exception("Wrong mode")

    return model, generator


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

def quantize_model(model, quantization_type, generator):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization_type == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization_type == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = lambda: representative_data_gen(generator)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
    elif quantization_type == 'int16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    else:
        raise ValueError(f'Unsupported quantization type: {quantization_type}')

    tflite_model = converter.convert()
    return tflite_model

import time
import psutil
def evaluate_model(tflite_model_path, generator, num_batches=100):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Measure inference time
    start_time = time.time()
    for _ in range(num_batches):
        x, _ = next(generator)
        for i in range(x.shape[0]):
            interpreter.set_tensor(input_details[0]['index'], x[i:i+1, ...])
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.time()

    # Measure memory usage
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    inference_time = (end_time - start_time) / (num_batches * x.shape[0])
    memory_usage = memory_info.rss / 1024 ** 2  # Convert to MB

    return inference_time, memory_usage


def main():
    # Uncomment if you want to run model on cpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    cutting_files = False

    root_path = "./datasets/datasets/musicnet"
    cut_files_folder_name = "4s_len.0s_overlap"

    if cutting_files:
        file_cutter(root_path)
        return
    filenames, labels = file_loader(root_path, cut_files_folder_name)
    mlb = MultiLabelBinarizer()

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(filenames, labels, mlb, 0.1, 0.1)
    class_totals = np.sum(y_train, axis=0)
    total_samples = y_train.shape[0]
    class_percentages = (class_totals / total_samples) * 100
    class_weights = calculate_class_weights(y_train)
    max_weight = max(class_weights.values())
    min_allowed_weight = 0.2
    normalized_class_weights = {k: max(v / max_weight, min_allowed_weight) for k, v in class_weights.items()}

    load_from_checkpoint = False
    epoch_num = 100
    """
    Model types
    ECA, SGE, TFQ, NONE, NONE_NO_ATTENTION, CONV
    """
    model_type = "TFQ"
    extension = "TFQ_now_correct_with_sigmoid"


    batch_sizes = [16, 32, 48, 64, 80]
    audio_bits = [8, 16, 32]
    quantization_weights = [8, 16, 32]




    wandb.init(
        # set the wandb project where this run will be logged
        project="magisterka-instrument-detection",
        entity="magisterka-instrument-detection",
        # track hyperparameters and run metadata
        config={
            "epoch_num": epoch_num,
            "model": model_type,
            "audio_bits": audio_bit,
            "batch_size": batch_size
        }
    )

    try:
        model, generator = start_training(model_type, extension, epoch_num, x_train, y_train, x_val, y_val, x_test, y_test, mlb, "",
                               normalized_class_weights, audio_bit, quantization_weights, batch_size)
    except Exception as e:
        # Log the exception to wandb and continue with the next iteration
        wandb.log({"error": str(e)})
        print(f"An error occurred with batch size {batch_size} and audio bit {audio_bit}: {e}")
    finally:
        # Ensure wandb run is ended correctly
        wandb.finish()


if __name__ == "__main__":
    main()
