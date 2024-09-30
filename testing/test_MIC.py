from model.attentionMIC.model_predict import predict_model
from prettytable import PrettyTable
import os
import sys
from pathlib import Path

from misc.multi_print import multi_print


class TestMICModel:
    def __init__(self, model, generator, classes_names, log_dir="", verbose=1, on_cpu=False):
        self.model = model
        self.generator = generator
        self.true_labels = generator.y
        self.classes_names = classes_names
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.log_filepath = Path(log_dir) / "test_output.txt"
        self.on_cpu = on_cpu

    def __call__(self, *args, **kwargs):
        self.test()

    def _evaluate_model(self, file):

        test_metrics = self.model.evaluate(self.generator, steps=len(self.generator), return_dict=True)

        # Print the metrics
        for key, value in test_metrics.items():
            with multi_print(file, sys.stdout):
                print(f"{key}: {value:.4f}")

        return test_metrics

    def test(self):
        with open(self.log_filepath, 'w') as file:
            with multi_print(file, sys.stdout):
                if self.verbose:
                    print("Running model evaluate on the testing data")

            metrics = self._evaluate_model(file)

            predict_model(self.model, self.generator, self.true_labels, self.classes_names,
                          file=file, log_dir=self.log_dir, on_cpu=self.on_cpu)
