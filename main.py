import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import pathlib
from generator import AudioGenerator
from misc.instrument_number_map import instruments_map
from sklearn.preprocessing import MultiLabelBinarizer

from cut_dataset.load_cut_files import load_cut_files
from cut_dataset.cut_audio_file import main_audio_cutter
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from model.create_models.conv_model import create_conv_model_from_paper, MultiOutputAccuracy

import datetime
from matplotlib import pyplot as plt

"""
What to have before presentation 13.02

1. Learn how to work with musicnet dataset.
a) Know how to extract data from the files
b) Create functions that will do that
c) Have a train and test datasets saved in shape of:
- x: raw audio file
- y: list of instruments used in that file
d) The same as above but y is a list of instruments used and timestamps of times when they are used
2. Create a simple network with the context of understanding instruments.
2.5 Add tensorboard
3. Copy the cnn network that was created in Maciek's paper.
4. Explore other datasets, change them to x and y and incorporate into the networks, check which ones work better alone.
5. After checking just audio and labels check if any other metadata could be used.


So the model has 4 phases of progress:
a) Be able to tell whether the following instrument exists in the audio.
b) List what instruments are in the audio based on the labels it knows.
c) Return a list of timestamps where the given instrument is playing.
d) Return a list of timestamps where all the instruments found are playing.

"""


"""
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


def create_example_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    return model

class BatchLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_every_n_batches=100, logdir="./logs/batch"):
        super(BatchLogger, self).__init__()
        self.log_every_n_batches = log_every_n_batches
        self.logdir = logdir
        self.writer = tf.summary.create_file_writer(logdir)

    def on_batch_end(self, batch, logs=None):
        if batch % self.log_every_n_batches == 0:
            with self.writer.as_default():
                for name, value in logs.items():
                    tf.summary.scalar(name, value, step=batch)
                self.writer.flush()


def file_cutter(root_path):
    segment_length = 4
    overlap_length = 0
    cut_files_folder_name = f"{segment_length}s_len.{overlap_length}s_overlap"
    main_audio_cutter(root_path, cut_files_folder_name, segment_length, overlap_length, None)


def file_loader(root_path, cut_files_folder_name):
    train_data_path = pathlib.Path(root_path).absolute() / "train_data"
    test_data_path = pathlib.Path(root_path).absolute() / "test_data"

    filenames_train, labels_train = load_cut_files(train_data_path / cut_files_folder_name)
    filenames_test, labels_test = load_cut_files(test_data_path / cut_files_folder_name)

    return filenames_train, labels_train, filenames_test, labels_test


from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay

def predict_model(model, testing_generator, true_labels, label_binarizer):
    # Predict probabilities on the test set
    predicted_probs = model.predict(testing_generator, steps=len(testing_generator))

    # Convert probabilities to binary predictions using a threshold
    predicted_labels = (predicted_probs > 0.5).astype(int)

    class_names = [str(label) for label in label_binarizer.classes_]
    # Print a classification report
    print(classification_report(true_labels, predicted_labels, target_names=class_names))

    # Calculate ROC-AUC for each class

    # roc_auc_scores = roc_auc_score(true_labels, predicted_probs, average=None)  # average=None for per-class scores
    #
    # # Print ROC-AUC scores
    # for i, class_name in enumerate(label_binarizer.classes_):
    #     print(f"ROC-AUC for class {class_name}: {roc_auc_scores[i]}")

    # Confusion matrices for each class
    confusion_matrices = multilabel_confusion_matrix(true_labels, predicted_labels)
    for i, class_name in enumerate(label_binarizer.classes_):
        print(f"Confusion matrix for class {class_name}:")
        print(confusion_matrices[i])

    num_classes = confusion_matrices.shape[0]

    # Setting up the figure, subplots
    fig, axes = plt.subplots(nrows=int(np.ceil(num_classes / 3)), ncols=3, figsize=(15, num_classes * 2))
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    # Loop through all classes and plot the confusion matrix for each
    for i in range(num_classes):
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrices[i], display_labels=["Absent", "Present"])
        disp.plot(ax=axes[i], cmap='Blues', values_format='d', colorbar=False)
        axes[i].title.set_text(f'Class: {label_binarizer.classes_[i]}')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def transform_labels_to_names(labels, instruments_map):
    """Transforms list of lists of numeric labels to list of lists of string labels."""
    transformed_labels = []
    for label_set in labels:
        transformed_labels.append([instruments_map[label] for label in label_set if label in instruments_map])
    return transformed_labels


def main():
    # Uncomment if you want to run model on cpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    cutting_files = False

    root_path = "./datasets/musicnet"
    cut_files_folder_name = "4s_len.0s_overlap"

    if cutting_files:
        file_cutter(root_path)
        return

    filenames, labels, filenames_test, labels_test = file_loader(root_path, cut_files_folder_name)

    labels_named = transform_labels_to_names(labels, instruments_map)
    labels_named_test = transform_labels_to_names(labels_test, instruments_map)

    # filenames, labels = load_cut_files(train_data_path / cut_files_folder_name)
    # filenames_test, labels_test = load_cut_files(test_data_path / cut_files_folder_name)

    mlb = MultiLabelBinarizer()
    instruments_transformed = mlb.fit_transform(labels_named)
    num_outputs = instruments_transformed.shape[1]

    generator = AudioGenerator(np.array(filenames), instruments_transformed, 32, use_mfcc=True)

    load_from_checkpoint = False
    if load_from_checkpoint:
        model_path = "model/saved_model/20240424-05313650.h5"
        model = load_model(model_path,
                           custom_objects={'MultiOutputAccuracy': MultiOutputAccuracy(num_outputs=num_outputs)})
    else:

        model = create_conv_model_from_paper((*generator.data_shape_mfcc, 1), num_outputs, mlb.classes_)
        model.summary()

        model_checkpoint_callback = ModelCheckpoint(
            filepath='./model/saved_model/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '{epoch:02d}.h5',
            save_weights_only=False,
            save_best_only=False,
            verbose=1)

        log_dir = "./model/logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq=50)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=MultiOutputAccuracy(num_outputs=num_outputs))

        model.fit(generator,
                  epochs=30,
                  steps_per_epoch=len(generator),
                  verbose=1,
                  use_multiprocessing=True,
                  workers=6,
                  callbacks=[model_checkpoint_callback, tensorboard_callback])

    instruments_transformed_test = mlb.transform(labels_named_test)
    testing_generator = AudioGenerator(np.array(filenames_test), instruments_transformed_test, 1, use_mfcc=True)
    test_metrics = model.evaluate(testing_generator, steps=len(testing_generator))
    test_loss = test_metrics[0]
    test_average_acc = test_metrics[-1]
    test_accuracies = test_metrics[1:-1]
    print(f"Test Loss: {test_loss}")
    print(f"Test Average Accuracy: {test_average_acc}")
    print(f"Test Accuracies: {test_accuracies}")

    predict_model(model, testing_generator, instruments_transformed_test, mlb)


if __name__ == "__main__":
    main()
