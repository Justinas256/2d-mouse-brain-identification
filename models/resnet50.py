from models.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, GlobalAveragePooling2D


class ResNet50V2Model(BaseModel):
    def __init__(self, input_shape, freeze: bool = False, imagenet: bool = True, weights_path: str = None):
        super().__init__(input_shape, freeze, weights_path)
        self.imagenet = imagenet

    def get_model_name(self):
        return "ResNet50v2"

    def _preprocess_input(self, inputs):
        return Lambda(
            lambda image: tf.keras.applications.resnet_v2.preprocess_input(image)
        )(inputs)

    def _get_backbone_model(self):
        # get model
        model = tf.keras.applications.ResNet50V2(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet" if self.imagenet else None,
            pooling="avg",
        )
        # freeze layers till conv5
        if self.freeze:
            self._freeze_layers(model)
        return model

    def _freeze_layers(self, model):
        for layer in model.layers:
            layer.trainable = False
            if layer.name.startswith(tuple(["conv5"])):
                layer.trainable = True
