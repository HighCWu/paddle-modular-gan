from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from absl import logging

from pmgan.architectures import arch_ops
from pmgan.gans import consts
import gin
from six.moves import range

from paddle import fluid
from paddle.fluid import layers
from paddle.fluid.initializer import Normal, Constant, Uniform

import functools

@gin.configurable("weights")
def weight_initializer(initializer=consts.NORMAL_INIT, stddev=0.02):
  pass

def _moving_moments_for_inference(mean, variance, is_training, decay):
  pass

def _accumulated_moments_for_inference(mean, variance, is_training):
  pass

@gin.configurable(whitelist=["decay", "epsilon", "use_cross_replica_mean",
                             "use_moving_averages", "use_evonorm"])
def standardize_batch(inputs,
                      is_training,
                      decay=0.999,
                      epsilon=1e-3,
                      data_format="NHWC",
                      use_moving_averages=True,
                      use_cross_replica_mean=None,
                      use_evonorm=False):
  pass

@gin.configurable(blacklist=["inputs"])
def no_batch_norm(inputs):
  return inputs

@gin.configurable(
    blacklist=["inputs", "is_training", "center", "scale", "name"])
def batch_norm(inputs, is_training, center=True, scale=True, name="batch_norm"):
  pass

@gin.configurable(whitelist=["num_hidden"])
def self_modulated_batch_norm(inputs, z, is_training, use_sn,
                              center=True, scale=True,
                              name="batch_norm", num_hidden=32):
  pass

# evonorm functions

@gin.configurable(whitelist=["nonlinearity"])
def evonorm_s0(inputs,
              data_format="NHWC",
              nonlinearity=True,
              name="evonorm-s0",
              scale=True,
              center=True):
  pass

def instance_std(x, eps=1e-5):
  pass

def group_std(x, groups=32, eps=1e-5):
  pass

#/ evonorm functions

@gin.configurable(whitelist=["use_bias"])
def conditional_batch_norm(inputs, y, is_training, use_sn, center=True,
                           scale=True, name="batch_norm", use_bias=False):
  pass

def layer_norm(input_, is_training, scope):
  pass

@gin.configurable(blacklist=["inputs"])
def spectral_norm(inputs, epsilon=1e-12, singular_value="auto", use_resource=True,
                  save_in_checkpoint=False, power_iteration_rounds=2):
  pass

class Conv2D(dg.Conv2D): 
  def __init__(self, *args, stddev=0.02, use_sn=False, use_bias=True, padding=1, **kwargs):
    weight_attr = fluid.ParamAttr(initializer=weight_initializer(stddev=stddev))
    bias_attr = None if use_bias else False
    super(Conv2D, self).__init__(
        *args, 
        padding=padding, 
        param_attr=weight_attr, bias_attr=bias_attr, **kwargs)
    if use_sn:
      self.sn = SpectralNorm(self, 'weight')
    self.use_sn = use_sn
    self.in_sn = False
  
  def forward(self, *args, **kwargs):
    if self.use_sn and not self.in_sn:
      self.in_sn = True
      return self.sn(*args, **kwargs)
    return super(Conv2D, self).forward(*args, **kwargs)

class Conv1x1(Conv2D):
  def __init__(self, *args, **kwargs):
    super(Conv1x1, self).__init__(*args, filter_size=1, stride=1, padding=0, **kwargs)

def deconv2d(inputs, output_shape, k_h, k_w, d_h, d_w,
             stddev=0.02, name="deconv2d", use_sn=False):
  pass

def lrelu(inputs, leak=0.2, name="lrelu"):
  pass

def weight_norm_linear(input_, output_size,
                       init=False, init_scale=1.0,
                       name="wn_linear",
                       initializer='truncated_normal_initializer',
                       stddev=0.02):
  pass

def weight_norm_conv2d(input_, output_dim,
                       k_h, k_w, d_h, d_w,
                       init, init_scale,
                       stddev=0.02,
                       name="wn_conv2d",
                       initializer='truncated_normal_initializer'):
  pass

def weight_norm_deconv2d(x, output_dim,
                         k_h, k_w, d_h, d_w,
                         init=False, init_scale=1.0,
                         stddev=0.02,
                         name="wn_deconv2d",
                         initializer='truncated_normal_initializer'):
  pass

def non_local_block(x, name, use_sn):
  pass

@gin.configurable(blacklist=['x', 'name'])
def noise_block(x, name, randomize_noise=True, stddev=0.00, noise_multiplier=1.0):
  pass

def censored_normal(shape,
                  mean=0.0,
                  stddev=1.0,
                  clip_min=0.0,
                  clip_max=1.0,
                  dtype='float32',
                  seed=None,
                  name=None):
  pass