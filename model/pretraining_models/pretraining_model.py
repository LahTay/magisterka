from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from model.pretraining_models.augment_mfcc import AugmentMFCCs


def create_pretraining_model_with_augmentation(input_shape, num_classes, dropout_rate=0.5, augment=False):
    input_layer = Input(shape=input_shape)

    # Apply MFCC augmentation
    if augment:
        x = AugmentMFCCs(time_shift_ratio=0.1, freq_mask_ratio=0.1, freq_mask_num=3,
                         min_warp_ratio=0.05, max_warp_ratio=0.15, max_no_warp=2)(input_layer)
    else:
        x = input_layer

    # Convolutional layers
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model
