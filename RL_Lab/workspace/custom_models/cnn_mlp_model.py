"""CNN MLP Model."""
import tensorflow as tf
from garage.experiment import deterministic
from garage.tf.models.model import Model
from garage.tf.models.mlp import mlp
from garage.tf.models.cnn import cnn_with_max_pooling


class CnnMpMlp_Model(Model):

    def __init__(self,                 
                 input_dim_cnn,
                 filters_cnn,
                 strides_cnn,
                 output_dim_mlp,
                 hidden_sizes_mlp,
                 name=None,
                 pool_shapes_cnn=(2, 2),
                 pool_strides_cnn=(2, 2),
                 padding_cnn='SAME',
                 hidden_nonlinearity_cnn=tf.nn.relu,
                 hidden_w_init_cnn=tf.initializers.glorot_uniform(seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init_cnn=tf.zeros_initializer(),
                 input_var2_mlp=None,
                 concat_layer_mlp=-2,
                 hidden_nonlinearity_mlp=tf.nn.relu,
                 hidden_w_init_mlp=tf.initializers.glorot_uniform(seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init_mlp=tf.zeros_initializer(),
                 output_nonlinearity_mlp=None,
                 output_w_init_mlp=tf.initializers.glorot_uniform(seed=deterministic.get_tf_seed_stream()),
                 output_b_init_mlp=tf.zeros_initializer(),
                 layer_normalization_mlp=False):
        super().__init__(name)

        self._input_dim_cnn = input_dim_cnn
        self._filters_cnn = filters_cnn
        self._strides_cnn = strides_cnn
        self._pool_shapes_cnn = pool_shapes_cnn
        self._pool_strides_cnn = pool_strides_cnn
        self._padding_cnn = padding_cnn
        self._hidden_nonlinearity_cnn = hidden_nonlinearity_cnn
        self._hidden_w_init_cnn = hidden_w_init_cnn
        self._hidden_b_init_cnn = hidden_b_init_cnn

        self._output_dim_mlp = output_dim_mlp
        self._hidden_sizes_mlp = hidden_sizes_mlp
        self._input_var2_mlp = input_var2_mlp
        self._concat_layer_mlp = concat_layer_mlp
        self._hidden_nonlinearity_mlp = hidden_nonlinearity_mlp
        self._hidden_w_init_mlp = hidden_w_init_mlp
        self._hidden_b_init_mlp = hidden_b_init_mlp
        self._output_nonlinearity_mlp = output_nonlinearity_mlp
        self._output_w_init_mlp = output_w_init_mlp
        self._output_b_init_mlp = output_b_init_mlp
        self._layer_normalization_mlp = layer_normalization_mlp

    # pylint: disable=arguments-differ
    def _build(self, state_input, name=None):
        """Build model given input placeholder(s).
        Args:
            state_input (tf.Tensor): Tensor input for state.
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.
        Return:
            tf.Tensor: Tensor output of the model.
        """
        del name
        out_cnn = cnn_with_max_pooling(input_var=state_input,
                                       input_dim=self._input_dim_cnn,
                                       filters=self._filters_cnn,
                                       strides=self._strides_cnn,
                                       pool_shapes=self._pool_shapes_cnn,
                                       pool_strides=self._pool_strides_cnn,
                                       padding=self._padding_cnn,
                                       hidden_nonlinearity=self._hidden_nonlinearity_cnn,
                                       hidden_w_init=self._hidden_w_init_cnn,
                                       hidden_b_init=self._hidden_b_init_cnn,
                                       name='custom_network/cnn')

        return mlp(input_var=out_cnn,
                   output_dim=self._output_dim_mlp,
                   hidden_sizes=self._hidden_sizes_mlp,
                   input_var2=self._input_var2_mlp,
                   concat_layer=self._concat_layer_mlp,
                   hidden_nonlinearity=self._hidden_nonlinearity_mlp,
                   hidden_w_init=self._hidden_w_init_mlp,
                   hidden_b_init=self._hidden_b_init_mlp,
                   output_nonlinearity=self._output_nonlinearity_mlp,
                   output_w_init=self._output_w_init_mlp,
                   output_b_init=self._output_b_init_mlp,
                   layer_normalization=self._layer_normalization_mlp,
                   name='custom_network/mlp',)