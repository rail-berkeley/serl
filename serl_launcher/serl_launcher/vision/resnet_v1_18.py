import jax.lax
import jax.numpy as jnp
import flax.linen as nn
import functools
from typing import (Any, Callable, Iterable, Optional, Tuple, Union)
import h5py
import warnings

from flax.linen.module import compact, merge_param
from jax.nn import initializers
from jax import lax

from serl_launcher.vision.resnet_v1 import SpatialLearnedEmbeddings, SpatialSoftmax

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any


# ---------------------------------------------------------------#
# Normalization
# ---------------------------------------------------------------#
def batch_norm(x, train, epsilon=1e-05, momentum=0.99, params=None, dtype='float32'):
    # we do not use running average in the implementation (set to False)
    if params is None:
        x = BatchNorm(epsilon=epsilon,
                      momentum=momentum,
                      use_running_average=False,        # was not train
                      dtype=dtype)(x)
    else:
        x = BatchNorm(epsilon=epsilon,
                      momentum=momentum,
                      bias_init=lambda *_: jnp.array(params['bias']),
                      scale_init=lambda *_: jnp.array(params['scale']),
                      mean_init=lambda *_: jnp.array(params['mean']),
                      var_init=lambda *_: jnp.array(params['var']),
                      use_running_average=False,        # was not train
                      dtype=dtype)(x)
    return x


def _absolute_dims(rank, dims):
    return tuple([rank + dim if dim < 0 else dim for dim in dims])


class BatchNorm(nn.Module):
    """BatchNorm Module.

    Taken from: https://github.com/google/flax/blob/master/flax/linen/normalization.py

    Attributes:
        use_running_average: if True, the statistics stored in batch_stats
                             will be used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    use_bias:  if True, bias (beta) is added.
    use_scale: if True, multiply by scale (gamma).
               When the next layer is linear (also e.g. nn.relu), this can be disabled
               since the scaling will be done by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
               devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
                       representing subsets of devices to reduce over (default: None). For
                       example, `[[0, 1], [2, 3]]` would independently batch-normalize over
                       the examples on the first two and last two devices. See `jax.lax.psum`
                       for more details.
    """
    use_running_average: Optional[bool] = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    mean_init: Callable[[Shape], Array] = lambda s: jnp.zeros(s, jnp.float32)
    var_init: Callable[[Shape], Array] = lambda s: jnp.ones(s, jnp.float32)
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @compact
    def __call__(self, x, use_running_average: Optional[bool] = None):
        """Normalizes the input using batch statistics.

        NOTE:
        During initialization (when parameters are mutable) the running average
        of the batch statistics will not be updated. Therefore, the inputs
        fed during initialization don't need to match that of the actual input
        distribution and the reduction axis (set with `axis_name`) does not have
        to exist.
        Args:
            x: the input to be normalized.
            use_running_average: if true, the statistics stored in batch_stats
                                 will be used instead of computing the batch statistics on the input.
        Returns:
            Normalized inputs (the same shape as inputs).
        """
        use_running_average = merge_param(
            'use_running_average', self.use_running_average, use_running_average)
        x = jnp.asarray(x, jnp.float32)
        axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis = _absolute_dims(x.ndim, axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

        # see NOTE above on initialization behavior
        initializing = self.is_mutable_collection('params')

        if use_running_average:
            ra_mean = self.variable('batch_stats', 'mean',
                                    self.mean_init,
                                    reduced_feature_shape)
            ra_var = self.variable('batch_stats', 'var',
                                   self.var_init,
                                   reduced_feature_shape)
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
            mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
            if self.axis_name is not None and not initializing:
                concatenated_mean = jnp.concatenate([mean, mean2])
                mean, mean2 = jnp.split(
                    lax.pmean(
                        concatenated_mean,
                        axis_name=self.axis_name,
                        axis_index_groups=self.axis_index_groups), 2)
            var = mean2 - lax.square(mean)

        y = x - mean.reshape(feature_shape)
        mul = lax.rsqrt(var + self.epsilon)
        if self.use_scale:
            scale = self.param('scale',
                               self.scale_init,
                               reduced_feature_shape).reshape(feature_shape)
            mul = mul * scale
        y = y * mul
        if self.use_bias:
            bias = self.param('bias',
                              self.bias_init,
                              reduced_feature_shape).reshape(feature_shape)
            y = y + bias
        return jnp.asarray(y, self.dtype)


LAYERS = {'resnet18': [2, 2, 2, 2]}


class BasicBlock(nn.Module):
    """
    Basic Block.

    Attributes:
        features (int): Number of output channels.
        kernel_size (Tuple): Kernel size.
        downsample (bool): If True, downsample spatial resolution.
        stride (bool): If True, use strides (2, 2). Not used in this module.
                       The attribute is only here for compatibility with Bottleneck.
        param_dict (h5py.Group): Parameter dict with pretrained parameters.
        kernel_init (functools.partial): Kernel initializer.
        bias_init (functools.partial): Bias initializer.
        block_name (str): Name of block.
        dtype (str): Data type.
    """
    features: int
    kernel_size: Union[int, Iterable[int]] = (3, 3)
    downsample: bool = False
    stride: bool = True
    param_dict: h5py.Group = None
    kernel_init: functools.partial = nn.initializers.lecun_normal()
    bias_init: functools.partial = nn.initializers.zeros
    block_name: str = None
    dtype: str = 'float32'

    @nn.compact
    def __call__(self, x, act, train=True):
        """
        Run Basic Block.

        Args:
            x (tensor): Input tensor of shape [N, H, W, C].
            act (dict): Dictionary containing activations.
            train (bool): Training mode.

        Returns:
            (tensor): Output shape of shape [N, H', W', features].
        """
        residual = x

        x = nn.Conv(features=self.features,
                    kernel_size=self.kernel_size,
                    strides=(2, 2) if self.downsample else (1, 1),
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_: jnp.array(
                        self.param_dict['conv1']['weight']),
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = batch_norm(x,
                       train=train,
                       epsilon=1e-05,
                       momentum=0.1,
                       params=None if self.param_dict is None else self.param_dict['bn1'],
                       dtype=self.dtype)
        x = nn.relu(x)

        x = nn.Conv(features=self.features,
                    kernel_size=self.kernel_size,
                    strides=(1, 1),
                    padding=((1, 1), (1, 1)),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_: jnp.array(
                        self.param_dict['conv2']['weight']),
                    use_bias=False,
                    dtype=self.dtype)(x)

        x = batch_norm(x,
                       train=train,
                       epsilon=1e-05,
                       momentum=0.1,
                       params=None if self.param_dict is None else self.param_dict['bn2'],
                       dtype=self.dtype)

        if self.downsample:
            residual = nn.Conv(features=self.features,
                               kernel_size=(1, 1),
                               strides=(2, 2),
                               kernel_init=self.kernel_init if self.param_dict is None else lambda *_: jnp.array(
                                   self.param_dict['downsample']['conv']['weight']),
                               use_bias=False,
                               dtype=self.dtype)(residual)

            residual = batch_norm(residual,
                                  train=train,
                                  epsilon=1e-05,
                                  momentum=0.1,
                                  params=None if self.param_dict is None else self.param_dict['downsample']['bn'],
                                  dtype=self.dtype)

        x += residual
        x = nn.relu(x)
        act[self.block_name] = x
        return x


class ResNet(nn.Module):
    """
    ResNet.

    Attributes:
        output (str):
            Output of the module. Available options are:
                - 'softmax': Output is a softmax tensor of shape [N, 1000]
                - 'log_softmax': Output is a softmax tensor of shape [N, 1000]
                - 'logits': Output is a tensor of shape [N, 1000]
                - 'activations': Output is a dictionary containing the ResNet activations
        pretrained (str):
            Indicates if and what type of weights to load. Options are:
                - 'imagenet': Loads the network parameters trained on ImageNet
                - None: Parameters of the module are initialized randomly
        normalize (bool):
            If True, the input will be normalized with the ImageNet statistics.
        architecture (str):
            Which ResNet model to use:
                - 'resnet18'
        num_classes (int):
            Number of classes.
        block (nn.Module):
            Type of residual block:
                - BasicBlock
        kernel_init (function):
            A function that takes in a shape and returns a tensor.
        bias_init (function):
            A function that takes in a shape and returns a tensor.
        ckpt_dir (str):
            The directory to which the pretrained weights are downloaded.
            Only relevant if a pretrained model is used.
            If this argument is None, the weights will be saved to a temp directory.
        dtype (str): Data type.
    """
    output: str = 'softmax'
    pretrained: str = 'imagenet'
    normalize: bool = True
    architecture: str = 'resnet18'
    num_classes: int = 1000
    block: nn.Module = BasicBlock
    kernel_init: functools.partial = nn.initializers.lecun_normal()
    bias_init: functools.partial = nn.initializers.zeros
    ckpt_dir: str = None
    dtype: str = 'float32'
    pre_pooling: bool = True  # skip pooling

    def setup(self):
        # self.param_dict = None
        if self.pretrained == 'imagenet':
            # ckpt_file = utils.download(self.ckpt_dir, URLS[self.architecture])
            self.param_dict = h5py.File(self.ckpt_dir, 'r')
            # print(f"loaded pretrained weights from {self.ckpt_dir}")

    @nn.compact
    def __call__(self, observations, train=False):
        """
        Args:
            x (tensor): Input tensor of shape [N, H, W, 3]. Images must be in range [0, 1].
            train (bool): Training mode.

        Returns:
            (tensor): Out
            if pre_pooling is True: features of shape (B, 7, 7, 512)
        """
        # assert observations.shape[-3:] == (224, 224, 3)

        if self.normalize:
            mean = jnp.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, -1)
            std = jnp.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, -1)
            x = (observations.astype(jnp.float32) / 255.0 - mean) / std

        if self.pretrained == 'imagenet':
            if self.num_classes != 1000:
                warnings.warn(f'The user specified parameter \'num_classes\' was set to {self.num_classes} '
                              'but will be overwritten with 1000 to match the specified pretrained checkpoint \'imagenet\', if ',
                              UserWarning)
            num_classes = 1000
        else:
            num_classes = self.num_classes

        act = {}

        x = nn.Conv(features=64,
                    kernel_size=(7, 7),
                    kernel_init=self.kernel_init if self.param_dict is None else lambda *_: jnp.array(
                        self.param_dict['conv1']['weight']),
                    strides=(2, 2),
                    padding=((3, 3), (3, 3)),
                    use_bias=False,
                    dtype=self.dtype)(x)
        act['conv1'] = x

        x = batch_norm(x,
                       train=train,
                       epsilon=1e-05,
                       momentum=0.1,
                       params=None if self.param_dict is None else self.param_dict['bn1'],
                       dtype=self.dtype)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))

        # Layer 1
        down = self.block.__name__ == 'Bottleneck'
        for i in range(LAYERS[self.architecture][0]):
            params = None if self.param_dict is None else self.param_dict['layer1'][f'block{i}']
            x = self.block(features=64,
                           kernel_size=(3, 3),
                           downsample=i == 0 and down,
                           stride=i != 0,
                           param_dict=params,
                           block_name=f'block1_{i}',
                           dtype=self.dtype)(x, act, train)

        # Layer 2
        for i in range(LAYERS[self.architecture][1]):
            params = None if self.param_dict is None else self.param_dict['layer2'][f'block{i}']
            x = self.block(features=128,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block2_{i}',
                           dtype=self.dtype)(x, act, train)

        # Layer 3
        for i in range(LAYERS[self.architecture][2]):
            params = None if self.param_dict is None else self.param_dict['layer3'][f'block{i}']
            x = self.block(features=256,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block3_{i}',
                           dtype=self.dtype)(x, act, train)

        # Layer 4
        for i in range(LAYERS[self.architecture][3]):
            params = None if self.param_dict is None else self.param_dict['layer4'][f'block{i}']
            x = self.block(features=512,
                           kernel_size=(3, 3),
                           downsample=i == 0,
                           param_dict=params,
                           block_name=f'block4_{i}',
                           dtype=self.dtype)(x, act, train)

        # if we want the pre_pooling output, return here
        if self.pre_pooling:
            return jax.lax.stop_gradient(x)  # shape (b, 7, 7, 512)

        # Classifier
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(features=num_classes,
                     kernel_init=self.kernel_init if self.param_dict is None else lambda *_: jnp.array(
                         self.param_dict['fc']['weight']),
                     bias_init=self.bias_init if self.param_dict is None else lambda *_: jnp.array(
                         self.param_dict['fc']['bias']),
                     dtype=self.dtype)(x)
        act['fc'] = x

        if self.output == 'softmax':
            return nn.softmax(x)
        if self.output == 'log_softmax':
            return nn.log_softmax(x)
        if self.output == 'activations':
            return act
        return x


resnetv1_18_configs = {
    "resnetv1-18-frozen": functools.partial(
        ResNet, architecture='resnet18',
        ckpt_dir="/examples/box_picking_drq/resnet18_weights.h5",
        pre_pooling=True
    )
}
