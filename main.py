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


def main():
    # Uncomment if you want to run model on cpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    cutting_files = False

    root_path = "./datasets/datasets/musicnet"
    cut_files_folder_name = "4s_len.0s_overlap"

    if cutting_files:
        file_cutter(root_path)
        return
    dataset_path = os.path.join(root_path, cut_files_folder_name)
    filenames, labels = file_loader(root_path, cut_files_folder_name)
    mlb = MultiLabelBinarizer()

    x_train, y_train, x_val, y_val, x_test, y_test = split_data(filenames, labels, mlb, 0.1, 0.1)

    load_from_checkpoint = False

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    def return_dirs(log_base, model_base, extension):
        log_dir = os.path.join(log_base, extension, current_time)
        os.makedirs(log_dir, exist_ok=True)
        model_base = os.path.join(model_base, extension, "_")
        return log_dir, model_base
    model_path = ""
    # log_dir_ECA, model_dir_ECA = return_dirs("./model/logs/", "./model/saved_model/", "ECA")
    # log_dir_SGE, model_dir_SGE = return_dirs("./model/logs/", "./model/saved_model/", "SGE")
    # log_dir_TFQ, model_dir_TFQ = return_dirs("./model/logs/", "./model/saved_model/", "TFQ")
    # log_dir_None, model_dir_None = return_dirs("./model/logs/", "./model/saved_model/", "None")
    log_dir_Conv, model_dir_Conv = return_dirs("./model/logs/", "./model/saved_model/", "Conv")

    epoch_num = 200
    multi_label = False

    # generator = AudioGenerator(x_train, y_train, 32, use_mfcc=True, multi_label=multi_label,
    #                            classes_names=mlb.classes_, testing=False)
    # val_generator = AudioGenerator(x_val, y_val, 32, use_mfcc=True, multi_label=multi_label,
    #                                classes_names=mlb.classes_, testing=True)
    # testing_generator = AudioGenerator(x_test, y_test, 32, use_mfcc=True, multi_label=multi_label,
    #                                    classes_names=mlb.classes_, testing=True)
    # model = TrainMICModel(MICType.SGE, generator, val_generator, log_dir_SGE, model_dir_SGE, mlb.classes_, model_path)(
    #     epoch_num)
    # TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir_SGE)()

    # generator = AudioGenerator(x_train, y_train, 32, use_mfcc=True, multi_label=multi_label,
    #                            classes_names=mlb.classes_, testing=False)
    # val_generator = AudioGenerator(x_val, y_val, 32, use_mfcc=True, multi_label=multi_label,
    #                                classes_names=mlb.classes_, testing=True)
    # testing_generator = AudioGenerator(x_test, y_test, 32, use_mfcc=True, multi_label=multi_label,
    #                                    classes_names=mlb.classes_, testing=True)
    #
    #
    # model = TrainMICModel(MICType.ECA, generator, val_generator, log_dir_ECA, model_dir_ECA, mlb.classes_, model_path)(
    #     epoch_num)
    # TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir_ECA)()
    #
    #
    # generator = AudioGenerator(x_train, y_train, 32, use_mfcc=True, multi_label=multi_label,
    #                            classes_names=mlb.classes_, testing=False)
    # val_generator = AudioGenerator(x_val, y_val, 32, use_mfcc=True, multi_label=multi_label,
    #                                classes_names=mlb.classes_, testing=True)
    # testing_generator = AudioGenerator(x_test, y_test, 32, use_mfcc=True, multi_label=multi_label,
    #                                    classes_names=mlb.classes_, testing=True)
    #
    # model = TrainMICModel(MICType.SGE, generator, val_generator, log_dir_SGE, model_dir_SGE, mlb.classes_, model_path)(
    #    epoch_num)
    # TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir_SGE)()
    #
    # generator = AudioGenerator(x_train, y_train, 32, use_mfcc=True, multi_label=multi_label,
    #                            classes_names=mlb.classes_, testing=False)
    # val_generator = AudioGenerator(x_val, y_val, 32, use_mfcc=True, multi_label=multi_label,
    #                                classes_names=mlb.classes_, testing=True)
    # testing_generator = AudioGenerator(x_test, y_test, 32, use_mfcc=True, multi_label=multi_label,
    #                                    classes_names=mlb.classes_, testing=True)
    #
    # model = TrainMICModel(MICType.TFQ, generator, val_generator, log_dir_TFQ, model_dir_TFQ, mlb.classes_, model_path)(
    #     epoch_num)
    # TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir_TFQ)()
    #
    # generator = AudioGenerator(x_train, y_train, 32, use_mfcc=True, multi_label=multi_label,
    #                            classes_names=mlb.classes_, testing=False)
    # val_generator = AudioGenerator(x_val, y_val, 32, use_mfcc=True, multi_label=multi_label,
    #                                classes_names=mlb.classes_, testing=True)
    # testing_generator = AudioGenerator(x_test, y_test, 32, use_mfcc=True, multi_label=multi_label,
    #                                    classes_names=mlb.classes_, testing=True)
    #
    # model = TrainMICModel(MICType.NONE, generator, val_generator, log_dir_None, model_dir_None, mlb.classes_, model_path)(
    #     epoch_num)
    # TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir_None)()
    #
    #
    #
    multi_label = True
    generator = AudioGenerator(x_train, y_train, 32, use_mfcc=True, multi_label=multi_label,
                               classes_names=mlb.classes_, testing=False)
    val_generator = AudioGenerator(x_val, y_val, 32, use_mfcc=True, multi_label=multi_label,
                                   classes_names=mlb.classes_, testing=True)
    testing_generator = AudioGenerator(x_test, y_test, 32, use_mfcc=True, multi_label=multi_label,
                                       classes_names=mlb.classes_, testing=True)

    model = TrainConvModel(generator, val_generator, log_dir_Conv, model_dir_Conv, mlb.classes_, model_path)(epoch_num)
    TestConvModel(model, testing_generator, mlb.classes_, log_dir=log_dir_Conv)()


if __name__ == "__main__":
    main()
