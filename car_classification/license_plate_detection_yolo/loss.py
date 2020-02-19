import tensorflow as tf 
import numpy as np 

from config import (
    COORDINATE_PARAM,
    NO_OBJECT_PARAM
)

def tini_yolo_loss(y_true, y_pred):
    # y_true have shape (grid_y, grid_x, (x, y, w, h, probabilaty))
    # because x_center and y_center in y_true is not over 1
    # so use sigmoid function to make it in range (0, 1)
    # and with one hot vector, using softmax function is good to
    # let try it

    # mask and object mask have shape (grid_y, grid_x, 1)
    # 1 is important because it help us to multiply with ...
    obj_mask = tf.expand_dims(y_true[..., 4], -1)
    # obj_mask = y_true[..., 4]
    no_obj_mask = 1.0 - y_true

    y_pred_xy, y_pred_wh, y_pred_prob = tf.split(y_pred, (2, 2, 1), axis=-1)

    y_pred_xy = tf.sigmoid(y_pred_xy)
    y_pred_wh = tf.cast(y_pred_wh, tf.float32)
    y_pred_prob = tf.sigmoid(y_pred_prob)

    y_true_xy = y_true[..., 0:2]
    y_true_wh = y_true[..., 2:4]
    y_true_prob = tf.expand_dims(y_true[..., 4], -1)


    # object loss
    prob_loss = tf.square(y_true_prob - y_pred_prob)
    obj_prob_loss = tf.reduce_sum(obj_mask * prob_loss)
    noobj_prob_loss = NO_OBJECT_PARAM * tf.reduce_sum(no_obj_mask * prob_loss)

    xy_loss = COORDINATE_PARAM * tf.reduce_sum(obj_mask * tf.square(y_true_xy - y_pred_xy))
    wh_loss = COORDINATE_PARAM * tf.reduce_sum(obj_mask * tf.square(y_true_wh - y_pred_wh))

    # one_hot_loss = tf.keras.losses.CategoricalCrossentropy()(y_true_one_hot, y_pred_one_hot)
    # xy_loss = tf.reduce_sum(tf.square(y_true_xy - y_pred_xy)) 
    # wh_loss = tf.reduce_sum(tf.square(y_true_wh - y_pred_wh)) 

    return obj_prob_loss + noobj_prob_loss + xy_loss + wh_loss
