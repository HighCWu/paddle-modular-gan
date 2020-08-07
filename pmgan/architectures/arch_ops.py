from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from pmgan.architectures import arch_ops
import gin
from six.moves import range

from paddle import fluid
from paddle.fluid import layers, dygraph as dg
from paddle.fluid.initializer import Normal, Constant, Uniform


class ReLU(dg.Layer):
  def forward(self, x):
    return layers.relu(x)
    
 
class SoftMax(dg.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.kwargs = kwargs
  
  def forward(self, x):
    return layers.softmax(x, **self.kwargs)


@gin.configurable
def no_batch_norm():
  return dg.Sequential()


@gin.configurable(
    blacklist=["num_features", "affine"])
def batch_norm(num_features, affine=True):
  return BatchNorm(num_features, affine=affine)


class BatchNorm(dg.BatchNorm):
  def __init__(self, *args, **kwargs):
    if 'affine' in kwargs:
      affine = kwargs.pop('affine')
      if not affine:
        kwargs['param_attr'] = fluid.ParamAttr(initializer=Constant(value=1.0), trainable=False)
        kwargs['bias_attr'] = fluid.ParamAttr(initializer=Constant(value=0.0), trainable=False)
    else:
      affine = True
    
    super().__init__(*args, **kwargs)
    self.affine = affine
    self.initialized = False
    self.accumulating = False
    self.accumulated_mean = self.create_parameter(shape=[args[0]], default_initializer=Constant(0.0))
    self.accumulated_var = self.create_parameter(shape=[args[0]], default_initializer=Constant(0.0))
    self.accumulated_counter = self.create_parameter(shape=[1], default_initializer=Constant(1e-12))
    self.accumulated_mean.trainable = False
    self.accumulated_var.trainable = False
    self.accumulated_counter.trainable = False

  def forward(self, inputs, *args, **kwargs):
    if not self.initialized:
      self.check_accumulation()
      self.set_initialized(True)
    if self.accumulating:
      self.eval()
      with dg.no_grad():
        axes = [0] + ([] if len(inputs.shape) == 2 else list(range(2,len(inputs.shape))))
        _mean = layers.reduce_mean(inputs, axes, keep_dim=True)
        mean = layers.reduce_mean(inputs, axes, keep_dim=False)
        var = layers.reduce_mean((inputs-_mean)**2, axes)
        self.accumulated_mean.set_value((self.accumulated_mean*self.accumulated_counter + mean) / (self.accumulated_counter + 1))
        self.accumulated_var.set_value((self.accumulated_var*self.accumulated_counter + var) / (self.accumulated_counter + 1))
        self.accumulated_counter.set_value(self.accumulated_counter + 1)
        _mean = self._mean*1.0
        _variance = self.variance*1.0
        self._mean.set_value(self.accumulated_mean)
        self._variance.set_value(self.accumulated_var)
        out = super().forward(inputs, *args, **kwargs)
        self._mean.set_value(_mean)
        self._variance.set_value(_variance)
        return out
    out = super().forward(inputs, *args, **kwargs)
    return out

  def check_accumulation(self):
    if self.accumulated_counter.numpy().mean() > 1-1e-12:
      self._mean.set_value(self.accumulated_mean)
      self._variance.set_value(self.accumulated_var)
      return True
    return False

  def clear_accumulated(self):
    self.accumulated_mean.set_value(self.accumulated_mean*0.0)
    self.accumulated_var.set_value(self.accumulated_var*0.0)
    self.accumulated_counter.set_value(self.accumulated_counter*0.0+1e-2)

  def set_accumulating(self, status=True):
    if status == True:
      self.accumulating = True
    else:
      self.accumulating = False

  def set_initialized(self, status=False):
    if status == False:
      self.initialized = False
    else:
      self.initialized = True
      
  def train(self):
    super().train()
    if self.affine:
      self.weight.stop_gradient = False
      self.bias.stop_gradient = False
    else:
      self.weight.stop_gradient = True
      self.bias.stop_gradient = True
    self._use_global_stats = False
    
  def eval(self):
    super().eval()
    self.weight.stop_gradient = True
    self.bias.stop_gradient = True
    self._use_global_stats = True


@gin.configurable(whitelist=["use_bias"])
def conditional_batch_norm(num_features, num_classes, use_sn, use_bias=False):
  return ConditionalBatchNorm(num_features, num_classes, use_sn, use_bias)


class ConditionalBatchNorm(dg.Layer):
  def __init__(self, num_features, num_classes, use_sn, use_bias=False, epsilon=1e-4, momentum=0.1):
    super(ConditionalBatchNorm, self).__init__()

    sn_fn = lambda layer: SpectralNorm(layer) if use_sn else layer
    self.bn_in_cond = BatchNorm(num_features, affine=False, epsilon=epsilon, momentum=momentum)
    self.gamma_embed = sn_fn(dg.Linear(num_classes, num_features, bias_attr=None if use_bias else False))
    self.beta_embed = sn_fn(dg.Linear(num_classes, num_features, bias_attr=None if use_bias else False))
 
  def forward(self, x, y):
    out = self.bn_in_cond(x)
    gamma = self.gamma_embed(y) + 1
    beta = self.beta_embed(y)
    out = layers.reshape(gamma, (0, 0, 1, 1)) * out + layers.reshape(beta, (0, 0, 1, 1))
    return out


def layer_norm(normalized_shape):
  raise NotImplementedError # return dg.LayerNorm(normalized_shape)


@gin.configurable
class SpectralNorm(dg.Layer):
  def __init__(self, module, name='weight', power_iterations=2):
    super().__init__()
    self.module = module
    self.name = name
    self.power_iterations = power_iterations
    if not self._made_params():
      self._make_params()

  def _update_u(self):
    w = self.weight
    u = self.weight_u

    if w.shape[0] == 4:
      _w = layers.transpose(w, [2,3,1,0])
      _w_t_shape = _w.shape
    _w = layers.reshape(_w, [-1, _w.shape[-1]])
    singular_value = "left" if _w.shape[0] <= _w.shape[1] else "right"
    for _ in range(self.power_iterations):
      if singular_value == "left":
        v = layers.l2_normalize(layers.matmul(_w, u, transpose_x=True), axis=None)
        u = layers.l2_normalize(layers.matmul(_w, v), axis=None)
      else:
        v = layers.l2_normalize(layers.matmul(u, _w, transpose_y=True), axis=None)
        u = layers.l2_normalize(layers.matmul(v, _w), axis=None)

    if singular_value == "left":
      sigma = layers.matmul(layers.matmul(u, _w, transpose_x=True), v)
    else:
      sigma = layers.matmul(layers.matmul(v, _w), u, transpose_y=True)
    _w = w / sigma
    if w.shape[0] == 4:
      _w = layers.transpose(layers.reshape(_w, _w_t_shape), [3,2,0,1])
    else:
      _w = layers.reshape(_w, w.shape)
    setattr(self.module, self.name, _w)
    self.weight_u.set_value(u)

  def _make_params(self):
    # paddle linear weight is similar with tf's, and conv weight is similar with pytorch's.
    w = getattr(self.module, self.name)

    if w.shape[0] == 4:
      _w = layers.transpose(w, [2,3,1,0])
    _w = layers.reshape(_w, [-1, _w.shape[-1]])
    singular_value = "left" if _w.shape[0] <= _w.shape[1] else "right"
    u_shape = (_w.shape[0], 1) if singular_value == "left" else (1, _w.shape[-1])
    
    u = self.create_parameter(shape=[u_shape], default_initializer=Normal(0, 1))
    u.stop_gradient = True
    u.set_value(layers.l2_normalize(u, axis=None))

    del self.module._parameters[self.name]
    self.add_parameter("weight", w)
    self.add_parameter("weight_u", u)

  def forward(self, *args, **kwargs):
    self._update_u()
    return self.module.forward(*args, **kwargs)


def conv2d(*args, **kwargs):
    use_sn = kwargs.pop("use_sn", False)
    use_bias = kwargs.pop("use_bias", True)
    padding = kwargs.pop("padding", 1)
    kwargs["padding"] = padding
    kwargs["bias_attr"] = None if use_bias else False
    conv = dg.Conv2D(*args, **kwargs)
    if use_sn:
      conv = SpectralNorm(conv)
    return conv


def non_local_block(in_dim, use_sn):
  return SelfAttention(in_dim, use_sn)


class SelfAttention(dg.Layer):
  def __init__(self, in_dim, use_sn):
    super().__init__()
    self.chanel_in = in_dim
 
    sn_fn = lambda layer: SpectralNorm(layer) if use_sn else layer
    self.theta = sn_fn(dg.Conv2D(in_dim, in_dim // 8, 1, bias_attr=False))
    self.phi = sn_fn(dg.Conv2D(in_dim, in_dim // 8, 1, bias_attr=False))
    self.pool = dg.Pool2D(2, 'max', 2)
    self.g = sn_fn(dg.Conv2D(in_dim, in_dim // 2, 1, bias_attr=False))
    self.o_conv = sn_fn(dg.Conv2D(in_dim // 2, in_dim, 1, bias_attr=False))
    self.gamma = self.create_parameter([1,], default_initializer=Constant(0.0))
 
    self.softmax = SoftMax(axis=-1)
 
  def forward(self, x):
    m_batchsize, C, width, height = x.shape
    N = height * width
 
    theta = self.theta(x)
    phi = self.phi(x)
    phi = self.pool(phi)
    phi = layers.reshape(phi,(m_batchsize, -1, N // 4))
    theta = layers.reshape(theta,(m_batchsize, -1, N))
    theta = layers.transpose(theta,(0, 2, 1))
    attention = self.softmax(layers.bmm(theta, phi))
    g = layers.reshape(self.pool(self.g(x)),(m_batchsize, -1, N // 4))
    attn_g = layers.reshape(layers.bmm(g, layers.transpose(attention,(0, 2, 1))),(m_batchsize, -1, width, height))
    out = self.o_conv(attn_g)
    return self.gamma * out + x

@gin.configurable
def noise_block(num_features, randomize_noise=True, noise_multiplier=1.0):
  return NoiseBlock(num_features, randomize_noise, noise_multiplier)

class NoiseBlock(dg.Layer):
  def __init__(self, num_features, randomize_noise=True, noise_multiplier=1.0):
    super(NoiseBlock, self).__init__()

    self.randomize_noise = randomize_noise
    self.noise_multiplier = noise_multiplier
    self.noise_strength = self.create_parameter([num_features], default_initializer=Normal())

  def forward(self, x):
    N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    if self.randomize_noise:
      noise = layers.randn(x.shape, dtype=x.dtype)
    else:
      noise = layers.randn(x.shape, dtype=x.dtype) # TODO:seed=0
      noise = layers.expand([noise], [N, 1, 1, 1])
    x = x + noise * layers.cast(self.noise_strength * self.noise_multiplier, x.dtype)
    return x