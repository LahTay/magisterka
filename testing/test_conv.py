from model.conv_model.model_predict import predict_model
from prettytable import PrettyTable
import os
import sys
from pathlib import Path
from misc.multi_print import multi_print



class TestConvModel:
    def __init__(self, model, generator, classes_names, log_dir="", verbose=1):
        self.model = model
        self.generator = generator
        self.true_labels = generator.y
        self.classes_names = classes_names
        self.log_dir = Path(log_dir)
        self.verbose = verbose
        self.log_filepath = self.log_dir / "test_output.txt"

    def __call__(self, *args, **kwargs):
        self.test()


    def _evaluate_model(self, file):
        test_metrics = self.model.evaluate(self.generator, steps=len(self.generator), return_dict=True)

        table = PrettyTable()
        table.field_names = ["Instrument", "Loss", "Accuracy", "Precision", "Recall", "F1Score"]

        for instrument in self.classes_names:
            instrument = instrument.replace(" ", "_")
            loss = test_metrics[f"{instrument}_loss"]
            accuracy = test_metrics[f"{instrument}_accuracy"]
            precision = test_metrics[f"{instrument}_precision"]
            recall = test_metrics[f"{instrument}_recall"]
            f1score = test_metrics[f"{instrument}_f1_score"]

            table.add_row(
                [instrument, f"{loss:.4f}", f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1score:.4f}"]
            )

        with multi_print(file, sys.stdout):
            print("General Loss:", f"{test_metrics['loss']:.4f}")
            print(table)

        return test_metrics


    def test(self):
        with open(self.log_filepath, 'w') as file:
            with multi_print(file, sys.stdout):
                if self.verbose:
                    print("Running model evaluate on the testing data", file=file)

            metrics = self._evaluate_model(file)

            predict_model(self.model, self.generator, self.true_labels, self.classes_names,
                          file=file, log_dir=self.log_dir)
