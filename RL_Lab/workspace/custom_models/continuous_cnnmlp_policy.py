"""This modules creates a continuous CNN with max pooling, MLP policy network.
It accepts an observation of the environment and predicts a
continuous action.
"""
import numpy as np
import tensorflow as tf
from garage.experiment import deterministic
from garage.tf.policies.policy import Policy

from custom_models.cnn_mlp_model import CnnMpMlp_Model


# pylint: disable=too-many-ancestors
class ContinuousCnnMlpPolicy(CnnMpMlp_Model, Policy):

    def __init__(self,
                 env_spec,
                 input_dim_cnn,
                 filters_cnn,
                 strides_cnn,
                 pool_shapes_cnn=(2, 2),
                 pool_strides_cnn=(2, 2),
                 padding_cnn='SAME',
                 hidden_sizes_mlp=(64, 64),
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
                 layer_normalization_mlp=False,
                 name='ContinuousCnnMpMlpPolicy'):

        self._env_spec = env_spec
        self._action_dim = env_spec.action_space.flat_dim
        self._obs_dim = env_spec.observation_space.flat_dim

        self._input_dim_cnn = input_dim_cnn
        self._filters_cnn = filters_cnn
        self._strides_cnn = strides_cnn
        self._pool_shapes_cnn = pool_shapes_cnn
        self._pool_strides_cnn = pool_strides_cnn
        self._padding_cnn = padding_cnn
        self._hidden_nonlinearity_cnn = hidden_nonlinearity_cnn
        self._hidden_w_init_cnn = hidden_w_init_cnn
        self._hidden_b_init_cnn = hidden_b_init_cnn

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

        super().__init__(input_dim_cnn=input_dim_cnn,
                        filters_cnn=filters_cnn,
                        strides_cnn=strides_cnn,
                        pool_shapes_cnn=pool_shapes_cnn,
                        pool_strides_cnn=pool_strides_cnn,
                        padding_cnn=padding_cnn,
                        hidden_nonlinearity_cnn=hidden_nonlinearity_cnn,
                        hidden_w_init_cnn=hidden_w_init_cnn,
                        hidden_b_init_cnn=hidden_b_init_cnn,
                        output_dim_mlp=self._action_dim,
                        hidden_sizes_mlp=hidden_sizes_mlp,
                        input_var2_mlp=input_var2_mlp,
                        concat_layer_mlp=concat_layer_mlp,
                        hidden_nonlinearity_mlp=hidden_nonlinearity_mlp,
                        hidden_w_init_mlp=hidden_w_init_mlp,
                        hidden_b_init_mlp=hidden_b_init_mlp,
                        output_nonlinearity_mlp=output_nonlinearity_mlp,
                        output_w_init_mlp=output_w_init_mlp,
                        output_b_init_mlp=output_b_init_mlp,
                        layer_normalization_mlp=layer_normalization_mlp,
                        name = name)

        self._initialize()

    def _initialize(self):
        state_input = tf.compat.v1.placeholder(tf.float32, shape=(None, self._obs_dim))
        outputs = super().build(state_input).outputs

        self._f_prob = tf.compat.v1.get_default_session().make_callable(outputs, feed_list=[state_input])

    # pylint: disable=arguments-differ
    def build(self, obs_var, name=None):
        """Symbolic graph of the action.
        Args:
            obs_var (tf.Tensor): Tensor input for symbolic graph.
            name (str): Name for symbolic graph.
        Returns:
            tf.Tensor: symbolic graph of the action.
        """
        return super().build(obs_var, name=name).outputs

    @property
    def input_dim(self):
        """int: Dimension of the policy input."""
        return self._obs_dim

    def get_action(self, observation):
        """Get single action from this policy for the input observation.
        Args:
            observation (numpy.ndarray): Observation from environment.
        Returns:
            numpy.ndarray: Predicted action.
            dict: Empty dict since this policy does not model a distribution.
        """
        actions, agent_infos = self.get_actions([observation])
        action = actions[0]
        return action, {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        """Get multiple actions from this policy for the input observations.
        Args:
            observations (numpy.ndarray): Observations from environment.
        Returns:
            numpy.ndarray: Predicted actions.
            dict: Empty dict since this policy does not model a distribution.
        """
        if not isinstance(observations[0],
                          np.ndarray) or len(observations[0].shape) > 1:
            observations = self.observation_space.flatten_n(observations)
        actions = self._f_prob(observations)
        actions = self.action_space.unflatten_n(actions)
        return actions, dict()

    def get_regularizable_vars(self):
        """Get regularizable weight variables under the Policy scope.
        Returns:
            list(tf.Variable): List of regularizable variables.
        """
        trainable = self.get_trainable_vars()
        return [
            var for var in trainable
            if 'hidden' in var.name and 'kernel' in var.name
        ]

    @property
    def env_spec(self):
        """Policy environment specification.
        Returns:
            garage.EnvSpec: Environment specification.
        """
        return self._env_spec

    def clone(self, name):
        """Return a clone of the policy.
        It copies the configuration of the primitive and also the parameters.
        Args:
            name (str): Name of the newly created policy.
        Returns:
            garage.tf.policies.ContinuousMLPPolicy: Clone of this object
        """
        new_policy = self.__class__(env_spec=self._env_spec,
                                    input_dim_cnn=self._input_dim_cnn,
                                    filters_cnn=self._filters_cnn,
                                    strides_cnn=self._strides_cnn,
                                    pool_shapes_cnn=self._pool_shapes_cnn,
                                    pool_strides_cnn=self._pool_strides_cnn,
                                    padding_cnn=self._padding_cnn,
                                    hidden_nonlinearity_cnn=self._hidden_nonlinearity_cnn,
                                    hidden_w_init_cnn=self._hidden_w_init_cnn,
                                    hidden_b_init_cnn=self._hidden_b_init_cnn,
                                    hidden_sizes_mlp=self._hidden_sizes_mlp,
                                    input_var2_mlp=self._input_var2_mlp,
                                    concat_layer_mlp=self._concat_layer_mlp,
                                    hidden_nonlinearity_mlp=self._hidden_nonlinearity_mlp,
                                    hidden_w_init_mlp=self._hidden_w_init_mlp,
                                    hidden_b_init_mlp=self._hidden_b_init_mlp,
                                    output_nonlinearity_mlp=self._output_nonlinearity_mlp,
                                    output_w_init_mlp=self._output_w_init_mlp,
                                    output_b_init_mlp=self._output_b_init_mlp,
                                    layer_normalization_mlp=self._layer_normalization_mlp,
                                    name = name)
        new_policy.parameters = self.parameters
        return new_policy

    def __getstate__(self):
        """Object.__getstate__.
        Returns:
            dict: the state to be pickled as the contents for the instance.
        """
        new_dict = super().__getstate__()
        del new_dict['_f_prob']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.
        Args:
            state (dict): unpickled state.
        """
        super().__setstate__(state)
        self._initialize()