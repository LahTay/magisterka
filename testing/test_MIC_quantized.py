from model.attentionMIC.model_predict import predict_model
from prettytable import PrettyTable
import os
import sys
import time
import numpy as np
from pathlib import Path
import tensorflow as tf
import pandas as pd
import wandb
from tqdm import tqdm

from misc.multi_print import multi_print
from misc.MultiLabelConfusionMatrix import ConfusionMatrix
from model.attentionMIC.model_predict import calc_metrics, classification_report


class TestMICQuantizedModel:
    def __init__(self, model, generator, classes_names, log_dir="", verbose=1):
        self.model = model
        self.interpreter = tf.lite.Interpreter(model_content=self.model, num_threads=16)
        self.generator = generator
        self.true_labels = generator.y
        self.classes_names = classes_names
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.log_filepath = Path(log_dir) / "test_output.txt"

    def __call__(self, *args, **kwargs):
        self.test()

    def tflite_predict(self, file=None):
        input_index = self.interpreter.get_input_details()[0]["index"]
        output_index = self.interpreter.get_output_details()[0]["index"]
        input_details = self.interpreter.get_input_details()
        input_shape = input_details[0]['shape']

        new_shape = (self.generator.batch_size, input_shape[1], input_shape[2], input_shape[3])
        self.interpreter.resize_tensor_input(input_details[0]['index'], new_shape)
        self.interpreter.allocate_tensors()

        predicted_probs = []
        start_time = time.time()

        # Function to run inference on a single sample
        def run_inference(single):
            single = np.expand_dims(single, -1)
            self.interpreter.set_tensor(input_index, single)
            self.interpreter.invoke()
            output = np.copy(self.interpreter.get_tensor(output_index))
            return output

        # Run predictions using TFLite interpreter
        for i in tqdm(range(len(self.generator)), desc="Processing test samples"):
            test_sample = self.generator[i]
            test_data = np.array(test_sample[0], dtype=np.float32)
            if test_data.shape[0] < self.generator.batch_size:
                pad_size = self.generator.batch_size - test_data.shape[0]
                mock_data = np.zeros((pad_size, input_shape[1], input_shape[2]), dtype=np.float32)

                # Pad the batch with mock data
                padded_test_data = np.concatenate((test_data, mock_data), axis=0)

                # Run inference on the padded data
                output = run_inference(padded_test_data)

                # Only keep the predictions for the real data (ignore the padded part)
                output = output[:test_data.shape[0]]
            else:
                # For normal batches, run inference as usual
                output = run_inference(test_data)

            predicted_probs.extend(np.copy(output))
            # for single in tqdm(test_data, desc="Running single batch"):
            #     output = run_inference(single)
            #     predicted_probs.append(output)


        end_time = time.time()
        with multi_print(file, sys.stdout):
            print(f"Inference time: {end_time - start_time} seconds")

        predicted_probs = np.vstack(predicted_probs)  # Stack all predictions
        return predicted_probs

    def _evaluate_model(self, file):

        test_metrics = self.model.evaluate(self.generator, steps=len(self.generator), return_dict=True)

        # Print the metrics
        for key, value in test_metrics.items():
            with multi_print(file, sys.stdout):
                print(f"{key}: {value:.4f}")

        return test_metrics


    def predict_model(self, predicted_probs, file):
        num_classes = len(self.classes_names)
        metrics = calc_metrics(self.true_labels, predicted_probs, num_classes, self.classes_names)

        classification_string = classification_report(metrics, self.classes_names, quantized=True)

        with multi_print(file, sys.stdout):
            print("\nClassification report:\n")
            print(classification_string)

        confusion_matrix = ConfusionMatrix(y_true=self.true_labels, y_pred=predicted_probs,
                                           label_names=self.classes_names, threshold=0.5)

        with multi_print(file, sys.stdout):
            print(confusion_matrix)

        confusion_matrix_plot_path = Path(self.log_dir) / "confusion_matrices.png"

        confusion_matrix.plot(file_name=confusion_matrix_plot_path)

        wandb.log({"confusion_matrix TFLog": wandb.Image(str(confusion_matrix_plot_path))})
        predicted_labels = (predicted_probs > 0.5).astype(int)

        rows = []

        # Iterate over each label
        for i, label_name in enumerate(self.classes_names):
            true_label = self.true_labels[:, i]
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
        wandb.log({"true_vs_predicted TFLite": wandb_true_pred_table})


    def test(self):
        with open(self.log_filepath, 'w') as file:
            with multi_print(file, sys.stdout):
                if self.verbose:
                    print("Running model evaluate on the testing data")

            predicted_probs = self.tflite_predict(file=file)

            self.predict_model(np.array(predicted_probs), file)



