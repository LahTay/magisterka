from model.conv_model.model_predict import predict_model
from prettytable import PrettyTable
import os


class TestMICModel:
    def __init__(self, model, generator, classes_names, log_dir="", verbose=1):
        self.model = model
        self.generator = generator
        self.true_labels = generator.y
        self.classes_names = classes_names
        self.log_dir = log_dir
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        self.test()

    def test(self):
        log_file_path = os.path.join(self.log_dir, "test_output.txt")
        with open(log_file_path, 'w') as file:
            if self.verbose:
                print("Running model evaluate on the testing data", file=file)
            test_metrics = self.model.evaluate(self.generator, steps=len(self.generator))
            metrics = {
                "Loss": test_metrics[0],
                "Precision": test_metrics[1],
                "Recall": test_metrics[2],
                "F1Score": test_metrics[3],
                "Hamming Loss": test_metrics[4],
            }

            for key, value in metrics.items():
                print(f"{key}: {value:.4f}", file=file)

            predict_model(self.model, self.generator, self.true_labels, self.classes_names,
                          file=file, separate_labels=False, log_dir=self.log_dir)