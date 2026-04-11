import tensorflow as tf
from tensorflow.keras import backend as K


def expand_and_tile(x, outsize):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=1)
    x = K.tile(x, [1, outsize, outsize, 1])
    return x


def expand_and_tile_1(x, outchannels):
    x = K.expand_dims(x, axis=-1)
    x = K.tile(x, [1, 1, outchannels])
    return x


def normalize_by_dim(x, dim=1024.):
    d = tf.convert_to_tensor(dim)
    return x/K.sqrt(d)


def split_dim_concat_batch(x, n):
    return tf.concat(tf.split(x, n, axis=-1), axis=0)


def split_batch_concat_dim(x, n):
    return tf.concat(tf.split(x, n, axis=0), axis=-1)


def normalize(x):
    x = (x+1.)/2.
    return K.clip(x, 1e-6, 1.-1e-6)


def l2_normalize(x):
    return tf.nn.l2_normalize(x, axis=-1, epsilon=1e-6)


def softmax(x):
    return K.softmax(x-tf.reduce_max(x), -1)


def concat_coord(x):
    ins_feat = x  # [N, h, w, c]

    batch_size = tf.shape(x)[0]
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    float_h = tf.cast(h, tf.float32)
    float_w = tf.cast(w, tf.float32)

    y_range = tf.range(h, dtype=tf.float32)
    x_range = tf.range(w, dtype=tf.float32)
    y_range = 2.0 * y_range / tf.maximum(float_h - 1.0, 1.0) - 1.0
    x_range = 2.0 * x_range / tf.maximum(float_w - 1.0, 1.0) - 1.0

    x_coords = tf.tile(x_range[tf.newaxis, :], [h, 1])
    y_coords = tf.tile(y_range[:, tf.newaxis], [1, w])

    x_coords = x_coords[tf.newaxis, :, :, tf.newaxis]
    y_coords = y_coords[tf.newaxis, :, :, tf.newaxis]
    x_coords = tf.tile(x_coords, [batch_size, 1, 1, 1])
    y_coords = tf.tile(y_coords, [batch_size, 1, 1, 1])

    ins_feat_out = tf.concat([ins_feat, x_coords, x_coords, x_coords, y_coords, y_coords, y_coords], axis=-1)

    return ins_feat_out
