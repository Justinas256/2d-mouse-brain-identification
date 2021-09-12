from models.base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB4


class EfficientNetModel(BaseModel):
    def __init__(
        self,
        input_shape,
        freeze: bool = False,
        architecture: int = 0,
        imagenet: bool = True,
        weights_path: str = None
    ):
        super().__init__(input_shape, freeze, weights_path)
        if not isinstance(architecture, int) or architecture not in [0, 4]:
            raise Exception("Type a valid architecture number: 0 or 4")
        else:
            self.architecture = architecture
        self.imagenet = imagenet

    def _preprocess_input(self, inputs):
        return Lambda(
            lambda image: tf.keras.applications.efficientnet.preprocess_input(image)
        )(inputs)

    def get_model_name(self):
        if self.architecture == 0:
            return "EfficientNetB0"
        elif self.architecture == 4:
            return "EfficientNetB4"
        else:
            return None

    def _get_backbone_model(self):
        if self.architecture == 0:
            model = EfficientNetB0(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet" if self.imagenet else None,
                pooling="avg",
            )
        elif self.architecture == 4:
            model = EfficientNetB4(
                input_shape=self.input_shape,
                include_top=False,
                weights="imagenet" if self.imagenet else None,
                pooling="avg",
            )
        else:
            raise Exception(
                f"Wrong EfficientNet architecture {self.architecture}. Expected values: 0 or 4"
            )

        if self.freeze:
            model.trainable = False

        return model
