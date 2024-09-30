from pathlib import Path
from generator import AudioGenerator

from model.metrics import MultiLabelAccuracy
from model.metrics import MultiLabelPrecision
from model.metrics import MultiLabelRecall
from model.metrics import MultiLabelF1Score
from model.metrics import MultiLabelInformedness
from model.metrics import MultiLabelMarkedness
from model.metrics import MultiLabelMCC
from model.metrics import MultiLabelCohenKappa

import time
import sys

from misc.multi_print import multi_print
from misc.MultiLabelConfusionMatrix import ConfusionMatrix

import pandas as pd
import wandb
import tensorflow as tf

def calc_metrics(y_true, y_pred, num_classes, classes_names):
    metric_objects = [MultiLabelAccuracy(num_classes, classes_names),
                      MultiLabelPrecision(num_classes, classes_names),
                      MultiLabelRecall(num_classes, classes_names),
                      MultiLabelF1Score(num_classes, classes_names),
                      MultiLabelInformedness(num_classes, classes_names),
                      MultiLabelMarkedness(num_classes, classes_names),
                      MultiLabelMCC(num_classes, classes_names),
                      MultiLabelCohenKappa(num_classes, classes_names)
                      ]

    total_results = {}

    for metric in metric_objects:
        metric.update_state(y_true, y_pred)

    for metric in metric_objects:
        results = metric.result()
        total_results[metric.name] = results

    return total_results


def classification_report(metrics, label_names, use_wandb=True, quantized=False):
    """
       Generate a detailed classification report using the results from calc_metrics.

       Args:
           metrics (dict): A dict returned by calc_metrics containing total results.
           label_names (list): List of label names.

       Returns:
           str: Formatted classification report.
       """

    total_results = metrics

    # Start with the header for each metric
    header = f"{'Label':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Informedness':<15} {'Markedness':<15} {'MCC':<10} {'CohenKappa':<15}"
    formatted_report = [header]
    formatted_report.append("-" * len(header))

    columns = ["Label", "Accuracy", "Precision", "Recall", "F1-Score", "Informedness", "Markedness", "MCC",
               "CohenKappa"]
    wandb_table = wandb.Table(columns=columns)

    # Loop through each label and format the metrics for each
    for i, label_name in enumerate(label_names):
        accuracy = total_results['multi_output_accuracy'][f'accuracy_{label_name}'].numpy()
        precision = total_results['multi_output_precision'][f'precision_{label_name}'].numpy()
        recall = total_results['multi_output_recall'][f'recall_{label_name}'].numpy()
        f1_score = total_results['multi_output_f1_score'][f'f1_score_{label_name}'].numpy()
        informedness = total_results['multi_output_informedness'][f'informedness_{label_name}'].numpy()
        markedness = total_results['multi_output_markedness'][f'markedness_{label_name}'].numpy()
        mcc = total_results['multi_output_mcc'][f'mcc_{label_name}'].numpy()
        cohen_kappa = total_results['multi_output_cohen_kappa'][f'cohen_kappa_{label_name}'].numpy()

        wandb_table.add_data(label_name, accuracy, precision, recall, f1_score, informedness, markedness, mcc,
                             cohen_kappa)

        formatted_report.append(
            f"{label_name:<20} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1_score:<10.4f} {informedness:<15.4f} {markedness:<15.4f} {mcc:<10.4f} {cohen_kappa:<15.4f}"
        )

    # Add a line for average metrics
    avg_accuracy = total_results['multi_output_accuracy']['Average_accuracy'].numpy()
    avg_precision = total_results['multi_output_precision']['Average_precision'].numpy()
    avg_recall = total_results['multi_output_recall']['Average_recall'].numpy()
    avg_f1_score = total_results['multi_output_f1_score']['Average_f1_score'].numpy()
    avg_informedness = total_results['multi_output_informedness']['Average_informedness'].numpy()
    avg_markedness = total_results['multi_output_markedness']['Average_markedness'].numpy()
    avg_mcc = total_results['multi_output_mcc']['Average_mcc'].numpy()
    avg_cohen_kappa = total_results['multi_output_cohen_kappa']['Average_cohen_kappa'].numpy()

    wandb_table.add_data("Average", avg_accuracy, avg_precision, avg_recall, avg_f1_score, avg_informedness,
                         avg_markedness, avg_mcc, avg_cohen_kappa)
    if quantized:
        text = '_TFLite'
    else:
        text = ''

    if use_wandb:
        # Log the table to WandB
        wandb.log({f"classification_report_table{text}": wandb_table})

    formatted_report.append("-" * len(header))
    formatted_report.append(
        f"{'Average':<20} {avg_accuracy:<10.4f} {avg_precision:<10.4f} {avg_recall:<10.4f} {avg_f1_score:<10.4f} {avg_informedness:<15.4f} {avg_markedness:<15.4f} {avg_mcc:<10.4f} {avg_cohen_kappa:<15.4f}"
    )

    # Join the report lines and return the final report string
    return "\n".join(formatted_report)


def predict_model(model, testing_generator: AudioGenerator, true_labels, label_names, file=None, log_dir="",
                  on_cpu=False):
    """
    TODO: generator has true labels in the generator.y, use them.
    Args:
        model: Model instance
        testing_generator: Generator with audio data and true labels for prediction
        true_labels: True labels
        label_names: Label names, gotten from the mlb instance

    Returns: None
    """
    if on_cpu:
        device = '/CPU:0'
    else:
        device = '/GPU:0'
    with tf.device(device):
        start_time = time.time()
        predicted_probs = model.predict(testing_generator, steps=len(testing_generator))
        end_time = time.time()
    with multi_print(file, sys.stdout):
        print(f"Inference time: {end_time - start_time} seconds {'on cpu' if on_cpu else 'on gpu'}")

    # Calculate metrics using the provided metrics functions
    num_classes = len(label_names)
    metrics = calc_metrics(true_labels, predicted_probs, num_classes, label_names)

    classification_string = classification_report(metrics, label_names)

    with multi_print(file, sys.stdout):
        print("\nClassification report:\n")
        print(classification_string)


    confusion_matrix = ConfusionMatrix(y_true=true_labels, y_pred=predicted_probs,
                                       label_names=label_names, threshold=0.5)

    with multi_print(file, sys.stdout):
        print(confusion_matrix)

    confusion_matrix_plot_path = Path(log_dir) / "confusion_matrices.png"

    confusion_matrix.plot(file_name=confusion_matrix_plot_path)

    wandb.log({"confusion_matrix": wandb.Image(str(confusion_matrix_plot_path))})
    predicted_labels = (predicted_probs > 0.5).astype(int)

    rows = []

    # Iterate over each label
    for i, label_name in enumerate(label_names):
        true_label = true_labels[:, i]
        predicted_label = predicted_labels[:, i]
        predicted_prob = predicted_probs[:, i]

        # Iterate over each sample
        for j in range(len(true_label)):
            # Append a dictionary representing each row to the list
            rows.append({
                'Class': label_name,
                'True Label': 'Yes' if true_label[j] == 1 else 'No',
                'Predicted Label': 'Yes' if predicted_label[j] == 1 else 'No',
                'Probability': predicted_prob[j]
            })

    # Convert the list of rows into a DataFrame
    true_pred_table = pd.DataFrame(rows)

    # Log the table to WandB
    wandb_true_pred_table = wandb.Table(dataframe=true_pred_table)
    wandb.log({"true_vs_predicted": wandb_true_pred_table})

