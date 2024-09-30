from model.pretraining_models.pretraining_model import create_pretraining_model_with_augmentation

from model.metrics import *
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import tensorflow as tf
import time
class PretrainMICModel:
    def __init__(self, generator, verbose=1, learning_rate=0.01, augment=False, classes_names=None):
        self.generator = generator
        self.verbose = verbose
        self.lr = learning_rate
        self.augment = augment
        self.model = None
        self.num_outputs = self.generator.get_label_num()
        self.classes_names = classes_names

    def __call__(self, epochs, *args, **kwargs):
        self.train(epochs)
        return self.get_model()


    def train(self, epochs):
        input_shape = (*self.generator.data_shape, 1)
        self.model = create_pretraining_model_with_augmentation(input_shape, self.num_outputs, augment=self.augment)

        metrics = [MultiLabelInformedness(self.num_outputs, self.classes_names),
                   MultiLabelMarkedness(self.num_outputs, self.classes_names),
                   MultiLabelMCC(self.num_outputs, self.classes_names),
                   ]
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                           loss="binary_crossentropy",
                           metrics=metrics)

        if self.verbose:
            self.model.summary()

        start_time = time.time()
        self.model.fit(self.generator,
                       epochs=epochs,
                       steps_per_epoch=len(self.generator),
                       verbose=1,
                       use_multiprocessing=False,
                       workers=4
                       )
        end_time = time.time()
        print(
            f"Pretraining time: {end_time - start_time} seconds for {(self.generator.batch_size - 1) * len(self.generator)} samples")

    def get_model(self):
        return self.model
