"""
    Extending the existing spatialfeatures.py for handling
    keras model graph related problems
"""
import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input
from keras.applications.densenet import DenseNet121 as D121
from keras.applications.densenet import preprocess_input as d_preprocess
from keras import backend as K
import tempfile


class VGG19:
    """
    A class that builds a TF graph with a pre-trained VGG19 model (on imagenet)
    Also takes care of preprocessing.
    Input should be a regular RGB image (0-255)
    """

    def __init__(self, input_shape, input_tensor=None):
        self.input_shape = input_shape
        self._build_graph(input_tensor)

    def _build_graph(self, input_tensor):
        with tf.Session() as sess:
            with tf.variable_scope("VGG19"):
                with tf.name_scope("inputs"):
                    if input_tensor is None:
                        input_tensor = tf.placeholder(
                            tf.float32, shape=self.input_shape, name="input_batch"
                        )
                    else:
                        assert self.input_shape == input_tensor.shape[1:]
                    self.input_tensor = input_tensor

                with tf.name_scope("preprocessing"):
                    img_batch = tf.keras.applications.VGG19.preprocess_input(
                        self.input_tensor
                    )

                with tf.variable_scope("model"):
                    self.vgg19 = tf.keras.applications.VGG19(
                        weights="imagenet",
                        include_top=False,
                        input_shape=self.input_shape,
                        input_tensor=img_batch,
                        pooling="max",
                    )

                self.layeroutputs = {l.name: l.output for l in self.vgg19.layers}

                feature_batch = tf.identity(
                    self.vgg19.layers[-1].output, name="feature_batch"
                )

                ## This statement gives the model's output (features)
                ## use self.output for extracting features using the model
                self.output = tf.expand_dims(feature_batch, 1, name="cnn_output")

            self.vgg_weights = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="VGG19/model"
            )

            with tempfile.NamedTemporaryFile() as f:
                self.tf_checkpoint_path = tf.train.Saver(self.vgg_weights).save(
                    sess, f.name
                )

        self.model_weights_tensor = set(self.vgg_weights)

    def load_weights(self):
        sess = tf.get_default_session()
        tf.train.Saver(self.vgg_weights).restore(sess, self.tf_checkpoint_path)

    def __getitem__(self, key):
        return self.layeroutputs[key]


class DenseNet121:
    """
    A class that builds a TF graph with a pre-trained DenseNet121 model (on imagenet)
    Also takes care of preprocessing.
    Input should be a regular RGB image (0-255)
    """

    def __init__(self, input_shape, input_tensor=None):
        self.input_shape = input_shape
        print("DenseNet121(Input) : ", self.input_shape)
        self._build_graph(input_tensor)

    def _build_graph(self, input_tensor):
        with tf.Session() as sess:
            with tf.variable_scope("DenseNet121"):
                with tf.name_scope("inputs"):
                    if input_tensor is None:
                        input_tensor = tf.placeholder(
                            tf.float32, shape=self.input_shape, name="input_batch"
                        )
                    else:
                        assert self.input_shape == input_tensor.shape[1:]
                    self.input_tensor = input_tensor

                with tf.name_scope("preprocessing"):
                    # img_batch = keras.applications.densenet.DenseNet121.preprocess_input(
                    #     self.input_tensor)
                    self.input_tensor = Input(tensor=self.input_tensor)
                    img_batch = d_preprocess(self.input_tensor)

                with tf.variable_scope("model"):
                    # self.densenet121 = keras.applications.densenet.DenseNet121(
                    #     weights='imagenet',
                    #     include_top=False,
                    #     input_shape=self.input_shape,
                    #     input_tensor=img_batch,
                    #     pooling='max')
                    self.densenet121 = D121(
                        weights="imagenet",
                        include_top=False,
                        input_shape=self.input_shape,
                        input_tensor=img_batch,
                        pooling="max",
                    )

                self.layeroutputs = {l.name: l.output for l in self.densenet121.layers}

                feature_batch = tf.identity(
                    self.densenet121.layers[-1].output, name="feature_batch"
                )

                ## This statement gives the model's output (features)
                ## use self.output for extracting features using the model
                self.output = tf.expand_dims(feature_batch, 1, name="cnn_output")

            self.vgg_weights = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="DenseNet121/model"
            )

            with tempfile.NamedTemporaryFile() as f:
                self.tf_checkpoint_path = tf.train.Saver(self.vgg_weights).save(
                    sess, f.name
                )

        self.model_weights_tensor = set(self.vgg_weights)

    def load_weights(self):
        sess = tf.get_default_session()
        tf.train.Saver(self.vgg_weights).restore(sess, self.tf_checkpoint_path)

    def __getitem__(self, key):
        return self.layeroutputs[key]
