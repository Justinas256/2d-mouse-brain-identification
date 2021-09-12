from models.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import (
    Lambda,
    Input,
    GlobalAveragePooling2D,
    Conv2D,
    BatchNormalization,
    MaxPool2D,
    Flatten,
)
from tensorflow.keras.models import Model


class SimpleCNNModel(BaseModel):
    def __init__(self, input_shape, freeze: bool = False, weights_path: str = None):
        if freeze:
            raise Exception("Freezing layers in this model is not supported")
        super().__init__(input_shape, freeze, weights_path)

    def get_model_name(self):
        return "SimpleCNN"

    def _preprocess_input(self, inputs):
        return Lambda(
            lambda image: tf.keras.applications.resnet_v2.preprocess_input(image)
        )(inputs)

    def _cnn_block(
        self, filters, kernel_size, stride, batch_normalization, max_pool, x
    ):
        x = Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
            activation="relu",
        )(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if max_pool:
            x = MaxPool2D(pool_size=2)(x)
        return x

    def _get_backbone_model(self):
        inputs = Input(self.input_shape)

        # encoder
        batch_normalization = False
        max_pool = True
        kernel_size = 4

        # 1 CNN
        x = self._cnn_block(64, kernel_size, 1, batch_normalization, max_pool, inputs)
        # 2 CNN
        x = self._cnn_block(64, kernel_size, 1, batch_normalization, max_pool, x)
        # 3 CNN
        x = self._cnn_block(128, kernel_size, 1, batch_normalization, max_pool, x)
        # 4 CNN
        x = self._cnn_block(128, kernel_size, 1, batch_normalization, max_pool, x)
        # 5 CNN
        x = self._cnn_block(128, kernel_size, 1, batch_normalization, max_pool, x)
        # 6 CNN
        x = self._cnn_block(256, kernel_size, 1, batch_normalization, max_pool, x)
        # 7 CNN
        x = self._cnn_block(256, kernel_size, 1, batch_normalization, max_pool, x)

        # build the model
        return Model(inputs=inputs, outputs=x)
