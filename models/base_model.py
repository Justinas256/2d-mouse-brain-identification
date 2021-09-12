from utils.helper import create_folder_if_not_exists

import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.layers import Lambda, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model


class BaseModel(object):
    def __init__(self, input_shape, freeze: bool = False, weights_path: str = None):
        self.input_shape = input_shape
        self.freeze = freeze
        self.model = None
        self.weights_path = weights_path

    # save function that saves the checkpoint in the defined path
    def save(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        create_folder_if_not_exists(checkpoint_path)

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path
    def load(self, checkpoint_path):
        if self.model is None:
            raise Exception("You have to build the model first.")

        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def get_model_name(self):
        raise NotImplementedError

    def _get_backbone_model(self):
        raise NotImplementedError

    def _preprocess_input(self, inputs):
        raise NotImplementedError

    def _freeze_layers(self, model):
        raise NotImplementedError

    def _build_model(self):
        inputs = Input(self.input_shape)

        # preprocess input
        x = self._preprocess_input(inputs)
        # get backbone network
        backbone = self._get_backbone_model()
        # pass input though the backbone network
        outputs = backbone(x)

        # x = tf.keras.layers.BatchNormalization()(outputs)
        x = tf.keras.layers.Flatten()(outputs)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dense(64, activation="linear")(x)
        x = tf.keras.layers.Lambda(lambda tensor: tf.math.l2_normalize(tensor, axis=1))(
            x
        )

        model = Model(inputs=inputs, outputs=x)

        if self.weights_path:
            model.trainable = True
            model.load_weights(self.weights_path)
            if self.freeze:
                self._freeze_layers(backbone)

        return model

    def compile_model(self):
        self.model = self._build_model()
        self.model.summary()
        self.model.compile(
            loss=tfa.losses.TripletSemiHardLoss(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        )

    def get_model(self):
        return self.model
