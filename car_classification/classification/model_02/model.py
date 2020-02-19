import sys
sys.path.insert(1, '../')
import numpy as np
import tensorflow as tf 
from tensorflow.keras import Model
import config
from utils import get_conv_layer
from tensorflow.keras.layers import (
    Conv2D, 
    BatchNormalization,
    Activation,
    Input,
    Dropout, 
    MaxPool2D, 
    Flatten,
    Dense,
    Softmax
)

def conv_block(x, filters, kernel_size, strides, training=True, name="unknow", activation='relu', 
                padding='same', batch_norm=True, drop_out=None, use_bias=True):
    """
        make an conv block
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, 
                strides=strides, padding=padding, 
                use_bias=use_bias, name="conv_layer_{}".format(name)) (x)
    if batch_norm:
        x = BatchNormalization(name="batchnorm_layer_{}".format(name)) (x, training)
    x = Activation(activation, name='{}_{}'.format(activation, name)) (x)
    # if drop_out and training:
    #     x = Dropout(drop_out, name="dropout_{}".format(name)) (x, training)
    return x

def max_pooling(x, pool_size=(2,2), strides=2, name="unknow"):
    x = MaxPool2D(pool_size=pool_size, strides=strides, name='maxpool_{}'.format(name)) (x)
    return x

class TiniVgg(object):
    def __init__(self,
                input_size=config.INPUT_SIZE):
        self.input_size = input_size
        self.optimizer = tf.keras.optimizers.Adam

    def model(self, training=False):
        x = input_layer = Input([self.input_size, self.input_size, 3])

        x = conv_block(x, 32, 3, 1, training=training, name="1_1")
        x = conv_block(x, 32, 3, 1, training=training, name="1_2")
        x = max_pooling(x, name="1")

        x = conv_block(x, 64, 3, 1, training=training, name="2_1")
        x = conv_block(x, 64, 3, 1, training=training, name="2_2")
        x = max_pooling(x, name="2")

        x = conv_block(x, 64, 3, 1, training=training, name="3_1")
        x = conv_block(x, 64, 3, 1, training=training, name="3_2")
        x = conv_block(x, 64, 1, 1, training=training, name="3_3")
        x = max_pooling(x, name="3")

        x = conv_block(x, 128, 3, 1, training=training, name="4_1")
        x = conv_block(x, 128, 3, 1, training=training, name="4_2")
        x = conv_block(x, 128, 1, 1, training=training, name="4_3")
        x = max_pooling(x, name="4")

        x = conv_block(x, 128, 3, 1, training=training, name="5_1")
        x = conv_block(x, 128, 3, 1, training=training, name="5_2")
        x = conv_block(x, 128, 1, 1, training=training, name="5_3")
        x = max_pooling(x, name="5")

        x = Flatten() (x)
        x = Dense(1024, activation='relu', use_bias=True, name="dense_1") (x)
        x = Dense(1024, activation='relu', use_bias=True, name='dense_2') (x)
        x = Dense(1, activation='sigmoid', use_bias=True, name='dense_3') (x)
        # x = Activation('sigmoid') (x)

        return Model(input_layer, x, name="tiniVgg")

    def optimizer(self):
        return self.optimizer

    def load_weights(self, weight_file):
        self.model.load_weights(weight_file)
        print("weights loaded")

if __name__=='__main__':
    model = TiniVgg().model(training=False)
    model.summary()
    