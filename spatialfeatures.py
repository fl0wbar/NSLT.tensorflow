"""
Extract spatial features with different CNN's
"""
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.applications import Xception  ###
from keras.applications.resnet50 import ResNet50  ###
from keras.applications.densenet import DenseNet121  ##
from keras.applications.densenet import DenseNet169  ###
from keras.applications.densenet import DenseNet201  ###
from keras.applications.mobilenetv2 import MobileNetV2  #
from keras.applications.vgg16 import preprocess_input as v_preprocess
from keras.applications.densenet import preprocess_input as d_preprocess
from keras.applications.resnet50 import preprocess_input as r_preprocess
from keras.applications.xception import preprocess_input as x_preprocess
from keras.applications.mobilenetv2 import preprocess_input as m_preprocess
import tensorflow as tf


class SFE_VGG16(object):
    """
    Feature extractor using VGG16
    """

    def __init__(self, source):
        """Create the graph of the VGG16 model.

        Args:
            source: Placeholder for the input tensor.
        """
        # Parse input arguments into class variables
        self.input = source
        # Call the create function to build the computational graph of VGG16
        self.create()

    def create(self):

        model = VGG16(weights="imagenet", include_top=False, input_shape=(227, 227, 3))

        self.output = model.predict(self.input)


class SFE_Xception(object):
    """
    Feature extractor using Xception Network
    """

    def __init__(self, source):
        """Create the graph of the Xception model.

        Args:
            source: Placeholder for the input tensor.
        """
        # Parse input arguments into class variables
        self.input = source
        # Call the create function to build the computational graph of Xception
        self.create()

    def create(self):

        model = Xception(
            weights="imagenet", include_top=False, input_shape=(227, 227, 3)
        )

        self.output = model.predict(self.input)


class SFE_ResNet50(object):
    """
    Feature extractor using ResNet50 Network
    """

    def __init__(self, source):
        """Create the graph of the ResNet50 model.

        Args:
            source: Placeholder for the input tensor.
        """
        # Parse input arguments into class variables
        self.input = source
        # Call the create function to build the computational graph of ResNet50
        self.create()

    def create(self):

        model = ResNet50(
            weights="imagenet", include_top=False, input_shape=(227, 227, 3)
        )

        self.output = model.predict(self.input)


class SFE_DenseNet121(object):
    """
    Feature extractor using DenseNet121 Network
    """

    def __init__(self, source):
        """Create the graph of the DenseNet121 model.

        Args:
            source: Placeholder for the input tensor.
        """
        print("DenseNet121(Input) : ", source.shape)
        # Parse input arguments into class variables
        self.input = source
        # Call the create function to build the computational graph of DenseNet121
        self.create()

    def create(self):

        # img_batch = d_preprocess(self.input)
        img_batch = Input(tensor=self.input)
        # batch_features = model(img_batch)
        model = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_shape=(227, 227, 3),
            input_tensor=img_batch,
            pooling="max",
        )
        feature_batch = tf.identity(model.layers[-1].output, name="feature_batch")
        self.output = tf.expand_dims(feature_batch, 1, name="cnn_output")
        # self.output = feature_model.predict(
        #     img_batch, steps=self.input.shape[0])
        # self.output = model.predict(img_data, steps=self.input.shape[0])
        # self.output = model.predict(img_data)


class SFE_DenseNet169(object):
    """
    Feature extractor using DenseNet169 Network
    """

    def __init__(self, source):
        """Create the graph of the DenseNet169 model.

        Args:
            source: Placeholder for the input tensor.
        """
        # Parse input arguments into class variables
        self.input = source
        # Call the create function to build the computational graph of DenseNet169
        self.create()

    def create(self):

        model = DenseNet169(
            weights="imagenet", include_top=False, input_shape=(227, 227, 3)
        )

        self.output = model.predict(self.input)


class SFE_DenseNet201(object):
    """
    Feature extractor using DenseNet201 Network
    """

    def __init__(self, source):
        """Create the graph of the DenseNet201 model.

        Args:
            source: Placeholder for the input tensor.
        """
        # Parse input arguments into class variables
        self.input = source
        # Call the create function to build the computational graph of DenseNet201
        self.create()

    def create(self):

        model = DenseNet201(
            weights="imagenet", include_top=False, input_shape=(227, 227, 3)
        )

        self.output = model.predict(self.input)


class SFE_MobileNetV2(object):
    """
    Feature extractor using MobileNetV2 Network
    """

    def __init__(self, source):
        """Create the graph of the MobileNetV2 model.

        Args:
            source: Placeholder for the input tensor.
        """
        # Parse input arguments into class variables
        self.input = source
        # Call the create function to build the computational graph of MobileNetV2
        self.create()

    def create(self):

        model = MobileNetV2(
            weights="imagenet", include_top=False, input_shape=(227, 227, 3)
        )

        self.output = model.predict(self.input)
