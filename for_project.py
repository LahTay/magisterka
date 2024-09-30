import itertools
import os
import pathlib
import datetime
import tensorflow as tf
from datasets.data_loaders.data_splitter import split_data
from sklearn.preprocessing import MultiLabelBinarizer
from datasets.cut_dataset.load_cut_files import load_cut_files
from generator import AudioGenerator
from training.train_MIC import TrainMICModel, MICType
from training.pretrain_model import PretrainMICModel
from testing.test_MIC import TestMICModel


def return_dirs(log_base, model_base, extension, current_time):
    log_dir = os.path.join(log_base, extension, current_time)
    os.makedirs(log_dir, exist_ok=True)
    model_base = os.path.join(model_base, extension, "_")
    return log_dir, model_base


def start_training(model_type, extension, epoch_num, dt, mlb, model_path="",
                   audio_bits=None, quantization_weights=None, batch_size=32,
                   pretraining=False, pretraining_dt=None, learning_rate=0.001, augment=False, pretrain_lr=0.01,
                   pretraining_mlb=None, epoch_pretrain=30, early_stopping_patience=10):
    # Training set up---------------------------------------------------------------------------------------------------

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir, model_dir = return_dirs("./model/logs/", "./model/saved_model/", extension, current_time)
    multi_label = False

    generator = AudioGenerator(dt.x_train, dt.y_train, batch_size, use_mfcc=True, multi_label=multi_label,
                               classes_names=mlb.classes_, testing=False)
    global_mean = generator.train_mean
    global_std = generator.train_std

    val_generator = AudioGenerator(dt.x_val, dt.y_val, batch_size, use_mfcc=True, multi_label=multi_label,
                                   classes_names=mlb.classes_, testing=True,
                                   train_mean=global_mean, train_std=global_std)
    testing_generator = AudioGenerator(dt.x_test, dt.y_test, batch_size, use_mfcc=True, multi_label=multi_label,
                                       classes_names=mlb.classes_, testing=True,
                                       train_mean=global_mean, train_std=global_std)

    pretrain_model = None
    if pretraining:
        pretraining_generator = AudioGenerator(pretraining_dt.x_train, pretraining_dt.y_train, batch_size, use_mfcc=True,
                                               multi_label=multi_label, classes_names=pretraining_mlb.classes_,
                                               testing=False)

        pretrain_model = PretrainMICModel(pretraining_generator, verbose=1, learning_rate=pretrain_lr,
                                          augment=augment, classes_names=pretraining_mlb.classes_)(epoch_pretrain)

    model = (TrainMICModel(MICType.TFQ, generator, val_generator, log_dir, model_dir, mlb.classes_, model_path,
                           learning_rate=learning_rate, pretrain_model=pretrain_model,
                           early_stopping_patience=early_stopping_patience)(epoch_num))

    TestMICModel(model, testing_generator, mlb.classes_, log_dir=log_dir)()

    return model, generator


def file_loader(files_path, is_string=False):
    filenames_train, labels_train = load_cut_files(pathlib.Path(files_path).absolute(), is_string)

    return filenames_train, labels_train

class Dataset:
    def __init__(self, x_train, y_train, x_val, y_val, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test


class Config:
    def __init__(self):
        # Model configurations
        """
        Model types
        ECA, SGE, TFQ, NONE, NONE_NO_ATTENTION, TFQ_ECA_HYBRID
        """
        self.model_types = ['TFQ', 'NONE', 'NONE_NO_ATTENTION', 'TFQ_ECA_HYBRID', 'SGE']
        self.current_model = self.model_types[0]
        self.use_scheduler = True

        # Hyperparameters
        self.extension = os.path.join("NEW_TEST", f"{self.model_types[0]}_testing")
        self.batch_sizes = [16, 32, 48, 64, 80]
        self.audio_bits = [8, 16, 32]
        self.quantization_weights = [8, 16, 32]
        self.learning_rates = [0.001]


        # Selected hyperparameters
        self.learning_rate = 0.001
        self.audio_bit = 16
        self.batch_size = 32
        self.epoch_num = 200
        self.early_stopping_patience = 10

        # Scheduler parameters
        # Best parameters - scheduler (Cosine Decay): 0.01 - alpha, 0.01 - initial learning rate, 0.98 - decay percent
        self.decay_percent = 0.98
        self.decay_percent_for_config = self.decay_percent
        self.initial_learning_rate = [0.001, 0.0001]
        self.initial_learning_rate_for_config = None
        self.alpha = [0.01, 0.001]
        # Below are updated using self.update_scheduler_params(dt)
        self.lr_limit = 0.0
        self.steps_per_epoch = 0
        self.decay_epochs = 0
        self.decay_steps = 0

        # Pretraining
        self.pretraining_flags = [False, True]
        self.pretraining_flag_for_config = False
        self.augment = False
        self.epoch_pretrain = 40
        self.use_genres = True
        self.pretrain_lr = 0.01

        # Dataset paths
        self.files_path = "./datasets/datasets/musicnet/4s_len.0s_overlap"
        self.pretraining_files_path = "./datasets/datasets/musan/4s_len.0s_overlap"

        # WandB configurations
        self.project_name = "magisterka-instrument-detection"
        self.entity_name = "magisterka-instrument-detection"
        self.wandb_group = "Test of predictions - New tests with pretraining and without"
        self.wandb_job_type = "train"

        # Model checkpointing
        self.load_from_checkpoint = False
        self.model_load_path = ""

    def update_scheduler_params(self, dt, initial_learning_rate, alpha):
        """Update scheduler-related parameters based on the current dataset and batch size."""
        self.steps_per_epoch = dt.x_train.shape[0] // self.batch_size
        self.decay_epochs = int(self.epoch_num * self.decay_percent)
        self.decay_steps = self.steps_per_epoch * self.decay_epochs
        self.lr_limit = alpha * initial_learning_rate

    def wandb_config(self):
        """Return a dictionary of parameters for WandB initialization."""
        config = {
            "epoch_num": self.epoch_num,
            "model": self.current_model,
            "audio_bits": self.audio_bit,
            "batch_size": self.batch_size,
            "pretraining_flag": self.pretraining_flag_for_config,
            "learning_rate": self.learning_rate,
            "augment": False
        }

        if self.use_scheduler:
            config.update({
                "decay_percent": self.decay_percent_for_config,
                "initial_learning_rate": self.initial_learning_rate_for_config,
                "lr_limit": self.lr_limit,
            })

        return config

    def get_all_hyperparameters_combinations(self, *args):
        for arg in args:
            if not hasattr(self, arg):
                raise ValueError(f"{arg} is not a valid attribute of Config class")

        # Retrieve the attributes and their names from the Config instance
        attributes = {arg: getattr(self, arg) for arg in args}

        # Generate the Cartesian product of the specified attributes
        combinations = itertools.product(*attributes.values())

        # Convert each combination to a dictionary
        for combination in combinations:
            yield dict(zip(attributes.keys(), combination))


def main():
    # Uncomment if you want to run model on cpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    config = Config()
    combinations = config.get_all_hyperparameters_combinations("model_types", "learning_rates",
                                                               "initial_learning_rate", "alpha", "pretraining_flags")


    for combination in combinations:
        model_type = combination.get('model_types', config.model_types[0])
        learning_rate = combination.get('learning_rates', config.learning_rates[0])
        config.learning_rate = learning_rate
        pretraining_flag = combination.get('pretraining_flags', config.pretraining_flags[0])

        initial_learning_rate = combination.get('initial_learning_rate', config.initial_learning_rate[0])
        alpha = combination.get('alpha', config.alpha[0])  # Default to 0.01 or any sensible value

        "HYPERPARAMETERS ---------------------------------------------------------------------------------------------"
        config.extension = os.path.join(f"NEW_TEST", f"{model_type}_testing")
        config.current_model = model_type
        if config.use_scheduler:
            print(f"-------------- NOW TRAINING MODEL: {model_type} WITH LR: Scheduler ----------------------\n")
        else:
            print(f"-------------- NOW TRAINING MODEL: {model_type} WITH LR: {learning_rate} -------------------\n")

        filenames, labels = file_loader(config.files_path, False)
        mlb = MultiLabelBinarizer()
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(filenames, labels, mlb, 0.1, 0.1, False)
        dt = Dataset(x_train, y_train, x_val, y_val, x_test, y_test)

        if config.use_scheduler:
            config.update_scheduler_params(dt, initial_learning_rate, alpha)
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=config.decay_steps,
                alpha=alpha
            )

        if pretraining_flag:
            pretraining_filenames, pretraining_labels = file_loader(config.pretraining_files_path, True)
            pretraining_mlb = MultiLabelBinarizer()
            x_pre_train, y_pre_train, x_pre_val, y_pre_val, x_pre_test, y_pre_test = split_data(
                pretraining_filenames, pretraining_labels, pretraining_mlb, 0.0, 0.0, True)
            pretraining_dt = Dataset(x_pre_train, y_pre_train, x_pre_val, y_pre_val, x_pre_test, y_pre_test)
        else:
            pretraining_dt = None
            pretraining_mlb = None

        try:
            model, generator = start_training(model_type, config.extension, config.epoch_num, dt, mlb,
                                              config.model_load_path, config.audio_bit,
                                              config.quantization_weights, config.batch_size,
                                              pretraining_flag, pretraining_dt, learning_rate,
                                              augment=config.augment, pretrain_lr=config.pretrain_lr,
                                              pretraining_mlb=pretraining_mlb, epoch_pretrain=config.epoch_pretrain,
                                              early_stopping_patience=config.early_stopping_patience)
        except Exception as e:
            print(f"An error occurred with batch size {config.batch_size} and audio bit {config.audio_bit}: {e}")


if __name__ == "__main__":
    main()
