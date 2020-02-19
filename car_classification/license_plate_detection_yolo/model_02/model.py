import numpy as np
import tensorflow as tf
import sys
sys.path.insert(1, '../')
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Dense,
    Activation
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)
from config import BATCH_SIZE, IMAGE_SIZE, GRID_SIZE

def conv_block(x, filters, kernel_size, strides, training=True, name="unknow", activation='relu', 
                padding='same', batch_norm=True, drop_out=None, use_bias=True):
    """
        make an conv block
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, 
                strides=strides, padding=padding, 
                use_bias=use_bias, name="conv_layer_{}".format(name)) (x)
    # if batch_norm:
    #     x = BatchNormalization(name="batchnorm_layer_{}".format(name)) (x, training)
    x = Activation(activation, name='{}_{}'.format(activation, name)) (x)
    if training:
        x = Dropout(0.3, name="dropout_{}".format(name)) (x, training)
    return x

def max_pooling(x, pool_size=(2,2), strides=2, name="unknow"):
    x = MaxPool2D(pool_size=pool_size, strides=strides, name='maxpool_{}'.format(name)) (x)
    return x

def DarknetConv(x, filters, size, training, strides=1, batch_norm=True):
    padding = 'same'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=True, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization() (x, training)
        x = LeakyReLU(alpha=0.1)(x)
    return x

# def DarknetResidual(x, filters, training):
#     prev = x
#     x = DarknetConv(x, filters // 2, 1, training)
#     x = DarknetConv(x, filters, 3, training)
#     x = Add()([prev, x])
#     return x

# def DarknetBlock(x, filters, blocks, training):
#     x = DarknetConv(x, filters, 3, training, strides=2)
#     for _ in range(blocks):
#         x = DarknetResidual(x, filters, training)
#     return x

class TiniYOLO(object):
    def __init__(self,
                input_size=IMAGE_SIZE,
                grid_size=GRID_SIZE):
        self.input_size = input_size
        self.grid_size = grid_size
        self.output_size = 5
        self.optimizer = tf.keras.optimizers.Adam
    def model(self, training=False):
        x = input_layer = Input([self.input_size, self.input_size, 3])
        x = DarknetConv(x, 16, 3, training)
        x = MaxPool2D(2, 2, 'same')(x)
        x = DarknetConv(x, 16, 3, training)
        x = MaxPool2D(2, 2, 'same')(x)
        x = DarknetConv(x, 32, 3, training)
        x = MaxPool2D(2, 2, 'same')(x)
        x = DarknetConv(x, 32, 3, training)
        x = MaxPool2D(2, 2, 'same')(x)
        x = DarknetConv(x, 64, 3, training)
        x = MaxPool2D(2, 2, 'same')(x)
        x = DarknetConv(x, 64, 3, training)
        x = MaxPool2D(2, 2, 'same')(x)
        x = DarknetConv(x, self.output_size, 1, training, batch_norm=False)

        return Model(input_layer, x, name='tini_yolo')

    def optimizer(self):
        return self.optimizer
if __name__=='__main__':
    model = TiniYOLO()
    model = model.model(training=True)
    model.summary()