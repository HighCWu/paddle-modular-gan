from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import math

from pmgan.architectures import abstract_arch
from pmgan.architectures import arch_ops as ops

from six.moves import range

from paddle.fluid import layers


def unpool(value):
  """Unpooling operation.
  N-dimensional version of the unpooling operation from
  https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf
  Taken from: https://github.com/tensorflow/tensorflow/issues/2169
  Args:
    value: a Tensor of shape [b, d0, d1, ..., dn, ch]
    name: name of the op
  Returns:
    A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
  """
  sh = value.shape
  dim = len(sh[2:])
  out = layers.reshape(value, [-1] + sh[-dim:])
  for i in range(dim+1, 1, -1):
    out = layers.concat([out, layers.zeros_like(out)], i)
  out_size = [-1] + [sh[1]] + [s * 2 for s in sh[2:]]
  out = layers.reshape(out, out_size)
  return out


class ConvBlock(ops.Conv2D):

  def forward(self, inputs, in_channels, scale="none"):
    if inputs.shape[1] != in_channels:
      raise ValueError("Unexpected number of input channels.")
    outputs = inputs
    if scale == "up":
      outputs = unpool(outputs)
    outputs = super(ConvBlock, self).forward(outputs)
    if scale == "down":
      outputs = layers.pool2d(outputs, [2, 2], "avg", pool_stride=[2, 2])

    return outputs


class ResNetBlock(abstract_arch.Module):
  """ResNet block with options for various normalizations."""

  def __init__(self,
               name,
               in_channels,
               out_channels,
               scale,
               is_gen_block,
               layer_norm=False,
               spectral_norm=False,
               batch_norm=None,
               z_dim=None,
               y_dim=None,):
    """Constructs a new ResNet block.
    Args:
      name: Scope name for the resent block.
      in_channels: Integer, the input channel size.
      out_channels: Integer, the output channel size.
      scale: Whether or not to scale up or down, choose from "up", "down" or
        "none".
      is_gen_block: Boolean, deciding whether this is a generator or
        discriminator block.
      layer_norm: Apply layer norm before both convolutions.
      spectral_norm: Use spectral normalization for all weights.
      batch_norm: Function for batch normalization.
      z_dim: Dimension numbers of latent z.
      y_dim: Dimension numbers of classifation features y.
    """
    assert scale in ["up", "down", "none"]
    self._name = name
    self._in_channels = in_channels
    self._out_channels = out_channels
    self._scale = scale
    # In SN paper, if they upscale in generator they do this in the first conv.
    # For discriminator downsampling happens after second conv.
    self._scale1 = scale if is_gen_block else "none"
    self._scale2 = "none" if is_gen_block else scale
    self._layer_norm = layer_norm
    self._spectral_norm = spectral_norm
    self.batch_norm = batch_norm
    self._z_dim = z_dim
    self._y_dim = y_dim

  def _get_conv(self, in_channels, out_channels, scale, suffix,
                kernel_size=(3, 3), strides=(1, 1)):
    """Performs a convolution in the ResNet block."""
    if scale not in ["up", "down", "none"]:
      raise ValueError(
          "Scale: got {}, expected 'up', 'down', or 'none'.".format(scale))

    name = "{}_{}".format("same" if scale == "none" else scale, suffix)
    conv = ConvBlock(
        in_channels, out_channels,
        kernel_size, strides,
        use_sn=self._spectral_norm)
    setattr(self, name, conv)

    return lambda inputs: conv(inputs, in_channels, scale)

  def apply(self):
    """"ResNet block containing possible down/up sampling, shared for G / D.
    """
    z_dim, y_dim = self._z_dim, self._y_dim

    self.conv_shortcut = self._get_conv(
        self._in_channels, self._out_channels, self._scale,
        kernel_size=(1, 1),
        suffix="conv_shortcut")

    self.bn1 = self.batch_norm(self._in_channels, z_dim=z_dim, y_dim=y_dim)
    if self._layer_norm:
      self.ln1 = ops.layer_norm(self._in_channels)

    self.conv1 = self._get_conv(
        self._in_channels, self._out_channels, self._scale1, 
        suffix="conv1")

    self.bn2 = self.batch_norm(self._out_channels, z_dim=z_dim, y_dim=y_dim)
    if self._layer_norm:
      self.ln2 = ops.layer_norm(self._out_channels)

    self.conv2 = self._get_conv(
        self._out_channels, self._out_channels, self._scale2, 
        suffix="conv2")

  def forward(self, inputs, z=None, y=None):
    """Forward for the given inputs.
    Args:
      inputs: a 3d input tensor of feature map.
      z: the latent vector for potential self-modulation. Can be None if use_sbn is set to False.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
      labels.
    Returns:
      output: a 3d output tensor of feature map.
    """
    if inputs.shape[1] != self._in_channels:
      raise ValueError(
          "Unexpected number of input channels (expected {}, got {}).".format(
              self._in_channels, inputs.shape[-1].value))
              
    outputs = inputs

    shortcut = self.conv_shortcut(inputs)
    
    outputs = self.bn1(outputs, z=z, y=y)
    if self._layer_norm:
      ouputs = self.ln1(outputs)
    
    if self._use_relu:
      outputs = layers.relu(outputs)
    outputs = self.conv1(outputs)
    outouts = self.bn2(outputs, z=z, y=y)
    
    if self._use_relu:
      outputs = layers.relu(outputs)
    outputs = self.conv2(outputs)
      
    # Combine skip-connection with the convolved part.
    return outputs + shortcut


class ResNetGenerator(abstract_arch.AbstractGenerator):
  """Abstract base class for generators based on the ResNet architecture."""

  def _resnet_block(self, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["up", "none"]:
      raise ValueError(
          "Unknown generator ResNet block scaling: {}.".format(scale))
    return ResNetBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=True,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)


class ResNetDiscriminator(abstract_arch.AbstractDiscriminator):
  """Abstract base class for discriminators based on the ResNet architecture."""

  def _resnet_block(self, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["down", "none"]:
      raise ValueError(
          "Unknown discriminator ResNet block scaling: {}.".format(scale))
    return ResNetBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=False,
        layer_norm=self._layer_norm,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)