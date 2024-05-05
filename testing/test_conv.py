from model.conv_model.model_predict import predict_model
from prettytable import PrettyTable
import os




class TestConvModel:
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
            table = PrettyTable()
            table.field_names = ["Instrument", "Loss", "Accuracy", "Precision", "Recall", "F1Score"]
            test_metrics = self.model.evaluate(self.generator, steps=len(self.generator))
            general_loss = test_metrics[0]
            loss_index = 1
            metric_index = 1 + len(self.classes_names)

            for i, instrument in enumerate(self.classes_names):
                loss = test_metrics[loss_index + i]
                accuracy = test_metrics[metric_index + i * 4]
                precision = test_metrics[metric_index + i * 4 + 1]
                recall = test_metrics[metric_index + i * 4 + 2]
                f1score = test_metrics[metric_index + i * 4 + 3]
                table.add_row(
                    [instrument, f"{loss:.4f}", f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1score:.4f}"])

            print("General Loss:", f"{general_loss:.4f}", file=file)
            print(table, file=file)

            predict_model(self.model, self.generator, self.true_labels, self.classes_names, separate_labels=True,
                          file=file, log_dir=self.log_dir)
