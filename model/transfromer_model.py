import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L


class TrigPosEmbedding(L.Layer):
    def call(self, inputs):
        seq_len = inputs.shape[1]
        hidden_dim = inputs.shape[2]
        if seq_len is None or hidden_dim is None:
            return inputs
        position = np.arange(seq_len)[:, None]
        div_term = np.exp(np.arange(0, hidden_dim, 2) * -(np.log(10000.0) / hidden_dim))
        encoding = np.zeros((seq_len, hidden_dim), dtype=np.float32)
        encoding[:, 0::2] = np.sin(position * div_term)
        encoding[:, 1::2] = np.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        return inputs + tf.constant(encoding[None, :, :], dtype=inputs.dtype)


def feed_forward(x, hidden_dim, activation):
    output_dim = int(x.shape[-1])
    y = L.Dense(hidden_dim * 4, activation=activation)(x)
    y = L.Dense(output_dim)(y)
    return y


def encoder_block(x, head_num, hidden_dim, attention_activation, feed_forward_activation, dropout_rate):
    key_dim = max(hidden_dim // head_num, 1)
    attn = L.MultiHeadAttention(num_heads=head_num, key_dim=key_dim, dropout=dropout_rate)(
        query=x,
        value=x,
        key=x,
    )
    x = L.LayerNormalization(epsilon=1e-6)(L.Add()([x, attn]))
    ff = feed_forward(x, hidden_dim, feed_forward_activation)
    return L.LayerNormalization(epsilon=1e-6)(L.Add()([x, ff]))


def decoder_block(x, encoded_layer, head_num, hidden_dim, attention_activation, feed_forward_activation, dropout_rate):
    key_dim = max(hidden_dim // head_num, 1)
    self_attn = L.MultiHeadAttention(num_heads=head_num, key_dim=key_dim, dropout=dropout_rate)(
        query=x,
        value=x,
        key=x,
    )
    x = L.LayerNormalization(epsilon=1e-6)(L.Add()([x, self_attn]))
    cross_attn = L.MultiHeadAttention(num_heads=head_num, key_dim=key_dim, dropout=dropout_rate)(
        query=x,
        value=encoded_layer,
        key=encoded_layer,
    )
    x = L.LayerNormalization(epsilon=1e-6)(L.Add()([x, cross_attn]))
    ff = feed_forward(x, hidden_dim, feed_forward_activation)
    return L.LayerNormalization(epsilon=1e-6)(L.Add()([x, ff]))


def ref_tf(encoder_input,
           decoder_input,
           feat_size,
           encoder_num=2,
           decoder_num=2,
           head_num=8,
           hidden_dim=256,
           num_query=32,
           attention_activation='relu',
           feed_forward_activation='relu',
           dropout_rate=0.1,
           trainable=True,
           balance=True):

    spatial_size = feat_size * feat_size

    encoder_embed = TrigPosEmbedding(name='Encoder-Embedding')(encoder_input)
    encoded_layer = get_encoders(
        encoder_num=encoder_num,
        input_layer=encoder_embed,
        head_num=head_num,
        hidden_dim=hidden_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    decoder_embed = TrigPosEmbedding(name='Decoder-Embedding')(decoder_input)
    decoded_layer = get_decoders(
        decoder_num=decoder_num,
        input_layer=decoder_embed,
        encoded_layer=encoded_layer,
        head_num=head_num,
        hidden_dim=hidden_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        trainable=trainable,
    )

    if balance:
        query_proj = L.Dense(hidden_dim, activation='relu')(decoder_input)
        output_proj = L.Dense(hidden_dim, activation='relu')(decoded_layer)
        output_proj = L.Concatenate()([output_proj, query_proj])
        output_proj = L.Dense(hidden_dim, activation='relu')(output_proj)
        query_confident = L.Dense(1, activation='sigmoid')(output_proj)

        weighted_output = L.Multiply()([query_confident, decoded_layer])
    else:
        weighted_output = decoded_layer

    output_layer = L.Dense(spatial_size, activation='relu')(weighted_output)
    output_layer = L.Reshape((num_query, feat_size, feat_size))(output_layer)
    output_layer = L.Permute((2, 3, 1))(output_layer)

    return output_layer


def lang_tf_enc(vision_input,
                lang_input,
                head_num=8,
                hidden_dim=256):
    decoder_embed_lang = TrigPosEmbedding(name='Fusion-Lang-Decoder-Embedding')(lang_input)
    decoder_embed_vis = TrigPosEmbedding(name='Fusion-Vis-Decoder-Embedding')(vision_input)
    q_inp = L.Dense(hidden_dim, activation='relu')(decoder_embed_vis)
    k_inp = L.Dense(hidden_dim, activation='relu')(decoder_embed_lang)
    v_inp = L.Dense(hidden_dim, activation='relu')(decoder_embed_lang)
    decoded_layer = L.MultiHeadAttention(num_heads=head_num, key_dim=max(hidden_dim // head_num, 1))(
        query=q_inp,
        value=v_inp,
        key=k_inp,
    )
    add_layer = L.Add(name='Fusion-Add')([decoded_layer, vision_input])

    return add_layer


def get_encoders(encoder_num,
                 input_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 name_prefix=''):
    """Get encoders.

    :param encoder_num: Number of encoder components.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    last_layer = input_layer
    for i in range(encoder_num):
        last_layer = encoder_block(
            x=last_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
        )
    return last_layer


def get_decoders(decoder_num,
                 input_layer,
                 encoded_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 name_prefix=''):
    """Get decoders.

    :param decoder_num: Number of decoder components.
    :param input_layer: Input layer.
    :param encoded_layer: Encoded layer from encoder.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    last_layer = input_layer
    for i in range(decoder_num):
        last_layer = decoder_block(
            x=last_layer,
            encoded_layer=encoded_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
        )
    return last_layer
