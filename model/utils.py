import tensorflow as tf
from keras import backend as K


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
    h_float = tf.cast(h, tf.float32)
    w_float = tf.cast(w, tf.float32)

    y_range = tf.cast(tf.range(h), tf.float32)
    x_range = tf.cast(tf.range(w), tf.float32)

    y_denom = tf.maximum(h_float - 1.0, 1.0)
    x_denom = tf.maximum(w_float - 1.0, 1.0)
    y_range = 2.0 * y_range / y_denom - 1.0
    x_range = 2.0 * x_range / x_denom - 1.0

    x_coords, y_coords = tf.meshgrid(x_range, y_range)
    x_coords = tf.expand_dims(x_coords, axis=0)
    x_coords = tf.expand_dims(x_coords, axis=-1)
    y_coords = tf.expand_dims(y_coords, axis=0)
    y_coords = tf.expand_dims(y_coords, axis=-1)

    multiples = tf.stack([batch_size, 1, 1, 1])
    x_coords = tf.tile(x_coords, multiples)
    y_coords = tf.tile(y_coords, multiples)

    ins_feat_out = tf.concat([ins_feat, x_coords, x_coords, x_coords, y_coords, y_coords, y_coords], axis=-1)

    return ins_feat_out
