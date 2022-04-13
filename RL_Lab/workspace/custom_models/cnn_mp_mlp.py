"""CNN with Max Pooling and Gaussian MLP in TensorFlow."""
import tensorflow as tf
from garage.experiment import deterministic


def cnn_mp_mlp(input_var_cnn,
               input_dim_cnn,
               filters_cnn,
               strides_cnn,
               pool_shapes_cnn,
               pool_strides_cnn,
               padding_cnn,
               output_dim,
               input_var2=None,
               concat_layer=-2,
               name='CnnMpMlpModel',
               hidden_nonlinearity_cnn=tf.nn.relu,
               hidden_w_init_cnn=tf.initializers.glorot_uniform(seed=deterministic.get_tf_seed_stream()),
               hidden_b_init_cnn=tf.zeros_initializer(),
               hidden_sizes=(32, 32),
               hidden_nonlinearity=tf.nn.tanh,
               hidden_w_init=tf.initializers.glorot_uniform(seed=deterministic.get_tf_seed_stream()),
               hidden_b_init=tf.zeros_initializer(),
               output_nonlinearity=None,
               output_w_init=tf.initializers.glorot_uniform(seed=deterministic.get_tf_seed_stream()),
               output_b_init=tf.zeros_initializer(),
               layer_normalization=False):


    pool_strides_cnn = [1, pool_strides_cnn[0], pool_strides_cnn[1], 1]
    pool_shapes_cnn = [1, pool_shapes_cnn[0], pool_shapes_cnn[1], 1]

    n_layers = len(hidden_sizes) + 1
    _merge_inputs = False

    if input_var2 is not None:
        _merge_inputs = True
        if n_layers > 1:
            _concat_layer = (concat_layer % n_layers + n_layers) % n_layers
        else:
            _concat_layer = 0

    with tf.compat.v1.variable_scope(name):
        # unflatten
        input_var_cnn = tf.reshape(input_var_cnn, [-1, *input_dim_cnn])

        h = input_var_cnn
        for index, (filter_iter, stride) in enumerate(zip(filters_cnn, strides_cnn)):
            _stride = [1, stride, stride, 1]
            h = _conv(h, 'h{}'.format(index), filter_iter[1], filter_iter[0],
                      _stride, hidden_w_init_cnn, hidden_b_init_cnn, padding_cnn)
            if hidden_nonlinearity_cnn is not None:
                h = hidden_nonlinearity_cnn(h)
            h = tf.nn.max_pool2d(h,
                                 ksize=pool_shapes_cnn,
                                 strides=pool_strides_cnn,
                                 padding=padding_cnn)

        # flatten
        dim = tf.reduce_prod(h.get_shape()[1:].as_list())
        out_cnn = tf.reshape(h, [-1, dim])

        for idx, hidden_size in enumerate(hidden_sizes):
            if _merge_inputs and idx == _concat_layer:
                out_cnn = tf.keras.layers.concatenate([out_cnn, input_var2])

            out_cnn = tf.compat.v1.layers.dense(inputs=out_cnn,
                                              units=hidden_size,
                                              activation=hidden_nonlinearity,
                                              kernel_initializer=hidden_w_init,
                                              bias_initializer=hidden_b_init,
                                              name='hidden_{}'.format(idx))
            if layer_normalization:
                out_cnn = tf.keras.layers.LayerNormalization()(out_cnn)

        if _merge_inputs and _concat_layer == len(hidden_sizes):
            out_cnn = tf.keras.layers.concatenate([out_cnn, input_var2])

        l_out = tf.compat.v1.layers.dense(inputs=out_cnn,
                                          units=output_dim,
                                          activation=output_nonlinearity,
                                          kernel_initializer=output_w_init,
                                          bias_initializer=output_b_init,
                                          name='output')
    return l_out


def _conv(input_var, name, filter_size, num_filter, strides, hidden_w_init,
          hidden_b_init, padding):
    """Helper function for performing convolution.
    Args:
        input_var (tf.Tensor): Input tf.Tensor to the CNN.
        name (str): Variable scope of the convolution Op.
        filter_size (tuple[int]): Dimension of the filter. For example,
            (3, 5) means the dimension of the filter is (3 x 5).
        num_filter (int): Number of filters. For example, (3, 32) means
            there are two convolutional layers. The filter for the first layer
            has 3 channels and the second one with 32 channels.
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
    Return:
        tf.Tensor: The output of the convolution.
    """
    # channel from input
    input_shape = input_var.get_shape()[-1]
    # [filter_height, filter_width, in_channels, out_channels]
    w_shape = [filter_size[0], filter_size[1], input_shape, num_filter]
    b_shape = [1, 1, 1, num_filter]
    
    with tf.compat.v1.variable_scope(name):
        weight = tf.compat.v1.get_variable('weight',
                                           shape=w_shape,
                                           initializer=hidden_w_init)
        bias = tf.compat.v1.get_variable('bias',
                                         b_shape,
                                         initializer=hidden_b_init)

        return tf.nn.conv2d(
            input_var, weight, strides=strides, padding=padding) + bias
