import csv
import os
from tensorflow.keras.callbacks import Callback


class CSVLoggerCallback(Callback):
    def __init__(self, log_dir, filename='training_log.csv'):
        super().__init__()
        self.log_dir = log_dir
        self.filename = filename
        self.filepath = os.path.join(self.log_dir, self.filename)
        self.csv_file = open(self.filepath, mode='w', newline='', encoding='utf-8')
        self.csv_writer = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.csv_writer is None:
            self.fieldnames = ['epoch'] + [key for key in logs.keys()]
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
            self.csv_writer.writeheader()

        logs['epoch'] = epoch
        self.csv_writer.writerow(logs)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()