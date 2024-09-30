import contextlib
import os
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, multilabel_confusion_matrix, ConfusionMatrixDisplay

from generator import AudioGenerator
from model.metrics import MultiLabelAccuracy
from model.metrics import MultiLabelCohenKappa
from model.metrics import MultiLabelF1Score
from model.metrics import MultiLabelInformedness
from model.metrics import MultiLabelMCC
from model.metrics import MultiLabelMarkedness
from model.metrics import MultiLabelPrecision
from model.metrics import MultiLabelRecall


def calc_metrics(y_true, y_pred, num_classes):
    metric_objects = [MultiLabelAccuracy(num_classes), MultiLabelPrecision(num_classes), MultiLabelRecall(num_classes),
                      MultiLabelF1Score(num_classes), MultiLabelInformedness(num_classes),
                      MultiLabelMarkedness(num_classes), MultiLabelMCC(num_classes),
                      MultiLabelCohenKappa(num_classes)]

    for metric in metric_objects:
        metric.update_state(y_true, y_pred)

        # Collect and print the results
    for metric in metric_objects:
        results = metric.result()
        print(f"\nResults for {metric.name}:")
        if isinstance(results, dict):
            for name, value in results.items():
                print(f"{name}: {value.numpy():.4f}")
        else:
            print(f"Metric result: {results.numpy():.4f}")







# Define a context manager to allow printing to both terminal and file
@contextlib.contextmanager
def multi_print(*files):
    original_stdout = sys.stdout
    sys.stdout = open(files[0], 'w') if len(files) == 1 else MultiStream(*files)
    try:
        yield
    finally:
        sys.stdout.close() if len(files) == 1 else None
        sys.stdout = original_stdout

class MultiStream:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)

    def flush(self):
        for f in self.files:
            f.flush()

def predict_model(model, testing_generator: AudioGenerator, true_labels, label_names,  file=None, log_dir=""):
    """
    TODO: generator has true labels in the generator.y, use them.
    Args:
        model: Model instance
        testing_generator: Generator with audio data and true labels for prediction
        true_labels: True labels
        label_names: Label names, gotten from the mlb instance

    Returns: None
    """
    # Predict probabilities on the test set
    start_time = time.time()
    predicted_probs = model.predict(testing_generator, steps=len(testing_generator))
    end_time = time.time()
    #print(f"Inference time: {end_time - start_time} seconds for {(testing_generator.batch_size - 1) * len(testing_generator)} samples")

    #metrics = calc_metrics(predicted_probs)


    data_shape = testing_generator._return_data_shape()
    mock_data = np.random.rand(1, *data_shape)
    start_time = time.time()

    model(mock_data, training=False)
    end_time = time.time()
    print(
        f"Inference time: {end_time - start_time} seconds")


    # Convert probabilities to binary predictions using a threshold
    predicted_labels = np.hstack([np.where(probs > 0.5, 1, 0) for probs in predicted_probs])
    true_labels = np.stack([testing_generator.y[:, i] for i in range(len(label_names))], axis=1)

    with multi_print(file, sys.stdout):
        print("\nClassification report:\n")
        print(classification_report(true_labels, predicted_labels, target_names=label_names))

    # Confusion matrices for each class
    confusion_matrices = multilabel_confusion_matrix(true_labels, predicted_labels)
    for i, class_name in enumerate(label_names):
        with multi_print(file, sys.stdout):
            print(f"Confusion matrix for class {class_name}:")
            print(confusion_matrices[i])

    num_classes = len(label_names)
    num_cols = 4
    num_rows = int(np.ceil(num_classes / num_cols))

    # Setting up the figure, subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_classes * 2, 15))
    axes = axes.flatten()

    # Loop through all classes and plot the confusion matrix for each
    for i in range(num_classes):
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrices[i], display_labels=["Absent", "Present"])
        disp.plot(ax=axes[i], cmap='Blues', values_format='d', colorbar=False)
        axes[i].title.set_text(f'Class: {label_names[i]}')
    for i in range(num_classes, len(axes)):
        axes[i].axis('off')

    # Adjust layout
    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plot_file_path = os.path.join(log_dir, "confusion_matrices.png")
    plt.savefig(plot_file_path)
    plt.close()
    file.close()
    plt.show()
