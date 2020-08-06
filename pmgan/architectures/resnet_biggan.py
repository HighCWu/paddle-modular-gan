from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from pmgan.architectures import abstract_arch
from pmgan.architectures import arch_ops as ops
from pmgan.architectures import resnet_ops
from pmgan.architectures import stylegan_ops

from paddle.fluid import layers, initializer


@gin.configurable
class BigGanResNetBlock(resnet_ops.ResNetBlock):
  """ResNet block with options for various normalizations.
  This block uses a 1x1 convolution for the (optional) shortcut connection.
  """

  def __init__(self,
               add_shortcut=True,
               use_relu=True,
               **kwargs):
    """Constructs a new ResNet block for BigGAN.
    Args:
      add_shortcut: Whether to add a shortcut connection.
      use_relu: Whether to use ReLU activation.
      **kwargs: Additional arguments for ResNetBlock.
    """
    name = kwargs.pop("name", None)
    super(BigGanResNetBlock, self).__init__(name, **kwargs)
    self._add_shortcut = add_shortcut
    self._use_relu = use_relu

  def apply(self):
    """"ResNet block containing possible down/up sampling, shared for G / D.
    """
    z_dim, y_dim = self._z_dim, self._y_dim
    self.bn1 = self.batch_norm(self._in_channels, z_dim=z_dim, y_dim=y_dim)
    if self._layer_norm:
      logging.info("[Block] %s using layer_norm", [None,self._in_channels,None,None])
      self.ln1 = ops.layer_norm(self._in_channels)

    if self._use_relu:
        pass
    else:
      logging.info("[Block] %s skipping relu", [None,self._in_channels,None,None])
    self.conv1 = self._get_conv(
        self._in_channels, self._out_channels, self._scale1, 
        suffix="conv1")

    self.bn2 = self.batch_norm(self._out_channels, z_dim=z_dim, y_dim=y_dim)
    if self._layer_norm:
      self.ln2 = ops.layer_norm(self._out_channels)

    if self._use_relu:
      pass
    self.conv2 = self._get_conv(
        self._out_channels, self._out_channels, self._scale2, 
        suffix="conv2")

    # Combine skip-connection with the convolved part.
    if self._add_shortcut:
      self.conv_shortcut = self._get_conv(
          self._in_channels, self._out_channels, self._scale,
          kernel_size=(1, 1),
          suffix="conv_shortcut")
    logging.info("[Block] %s (z=%s, y=%s) -> %s with scale %s", [None, self._in_channels, None, None],
                 None if z_dim is None else [None,z_dim],
                 None if y_dim is None else [None,y_dim], [None, self._out_channels, None, None], [self._scale1, self._scale2])
      
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
      if self._add_shortcut:
        shortcut = self.conv_shortcut(inputs)
        outputs = outputs + shortcut
        
      return outputs


@gin.configurable
class Generator(abstract_arch.AbstractGenerator):
  """ResNet-based generator supporting resolutions 32, 64, 128, 256, 512, 1024."""

  def __init__(self,
               z_dim=128,
               y_dim=1000,
               ch=96,
               blocks_with_attention="64",
               stylegan_z=False,
               hierarchical_z=True,
               embed_z=False,
               embed_y=True,
               embed_y_dim=128,
               embed_bias=False,
               channel_multipliers=None,
               plain_tanh=False,
               use_relu=True,
               use_noise=False,
               randomize_noise=True,
               **kwargs):
    """Constructor for BigGAN generator.
    Args:
      z_dim: Dimension numbers of latent z.
      y_dim: Dimension numbers of classifation features y.
      ch: Channel multiplier.
      blocks_with_attention: Comma-separated list of blocks that are followed by
        a non-local block.
      stylegan_z: Whether to use StyleGAN mode to process z.
      hierarchical_z: Split z into chunks and only give one chunk to each.
        Each chunk will also be concatenated to y, the one hot encoded labels.
      embed_z: If True use a learnable embedding of z that is used instead.
        The embedding will have the length of z.
      embed_y: If True use a learnable embedding of y that is used instead.
      embed_y_dim: Size of the embedding of y.
      embed_bias: Use bias with for the embedding of z and y.
      channel_multipliers: String type list of numbers to multiply with the 'ch' parameter, then get each layer input and output channels. If it's set to None, use the class default channel multiplier. Default: None.
      plain_tanh: Make output with value range (-1,1) or (0,1).
      use_relu: Whether to use ReLU activation.
      use_noise: Whether to add noise.
      randomize_noise: Whether to use randomize noise.
      **kwargs: additional arguments past on to ResNetGenerator.
    """
    super(Generator, self).__init__(**kwargs)
    self._z_dim = z_dim
    self._y_dim = y_dim
    self._ch = ch
    self._blocks_with_attention = set(blocks_with_attention.split(","))
    self._blocks_with_attention.discard('')
    self._channel_multipliers = None if channel_multipliers is None else [int(x.strip()) for x in channel_multipliers.split(",")]
    self._hierarchical_z = hierarchical_z
    self._embed_z = embed_z
    self._embed_y = embed_y
    self._embed_y_dim = embed_y_dim
    self._embed_bias = embed_bias
    self._plain_tanh = plain_tanh
    self._use_relu = use_relu
    self._use_noise = use_noise
    self._randomize_noise = randomize_noise
    if hierarchical_z and stylegan_z:
      raise ValueError("Must set either hierarchical_z or stylegan_z, not both")

  def _resnet_block(self, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["up", "none"]:
      raise ValueError(
          "Unknown generator ResNet block scaling: {}.".format(scale))
    return BigGanResNetBlock(
        use_relu=self._use_relu,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=True,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm,
        z_dim=self._z_dim,
        y_dim=self._y_dim)

  def _get_in_out_channels(self):
    resolution = self._image_shape[1]
    if self._channel_multipliers is not None:
      channel_multipliers = self._channel_multipliers
    elif resolution == 1024:
      channel_multipliers = [16, 16, 8, 8, 4, 2, 1, 1, 1]
    elif resolution == 512:
      channel_multipliers = [16, 16, 8, 8, 4, 2, 1, 1]
    elif resolution == 256:
      channel_multipliers = [16, 16, 8, 8, 4, 2, 1]
    elif resolution == 128:
      channel_multipliers = [16, 16, 8, 4, 2, 1]
    elif resolution == 64:
      channel_multipliers = [16, 16, 8, 4, 2]
    elif resolution == 32:
      channel_multipliers = [4, 4, 4, 4]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    in_channels = [self._ch * c for c in channel_multipliers[:-1]]
    out_channels = [self._ch * c for c in channel_multipliers[1:]]
    return in_channels, out_channels

  @property
  @gin.configurable("Generator_stylegan_z_args")
  def G_main_args(self, **args):
    return args

  def apply(self):
    """Build the generator network
    """
    with gin.config_scope("generator"):
      self._apply()

  def _apply(self):
    logging.info("[Generator] inputs are z=%s, y=%s", 
              [None, self._z_dim], 
              [None, self._y_dim] if self._y_dim > 0 else None)
    seed_size = 4
    in_channels, out_channels = self._get_in_out_channels()
    self.num_blocks = num_blocks = len(in_channels)

    if self._embed_z:
      self.embed_z = ops.linear(self._z_dim, self._z_dim, use_sn=False,
                     use_bias=self._embed_bias)
    if self._y_dim is not None and self._y_dim > 0 and self._embed_y:
      self.embed_y = ops.linear(self._y_dim, self._embed_y_dim, use_sn=False,
                     use_bias=self._embed_bias)
      
    if self._stylegan_z:
      z_args = self.G_main_args
      logging.info('[Generator] scope: %s stylegan_z_args: %s', gin.current_scope_str(), z_args)
      self.stylegan_z = stylegan_ops.G_main(num_blocks + 1, None, latent_size=self._z_dim, **z_args)

    per_z_dim = self._z_dim // (num_blocks + 1) if not self._hierarchical_z else self._z_dim
    per_y_dim = per_z_dim + ((self._embed_y_dim if self._embed_y else self._y_dim) if self._y_dim is not None and self._y_dim > 0 else 0)
    logging.info("[Generator] z0=%s, z_per_block=%s, y_per_block=%s",
                 [None, per_z_dim], [[None, per_z_dim] for _ in num_blocks],
                 [[None, per_y_dim] for _ in num_blocks])

    self.fc_noise = ops.linear(
        self._z_dim, 
        in_channels[0] * seed_size * seed_size,
        use_sn=self._spectral_norm)

    blocks_with_attention = set(self._blocks_with_attention)
    for block_idx in range(num_blocks):
      name = "B{}".format(block_idx + 1)
      if self._use_noise:
        setattr(self, "{}.noise_block".format(name), 
              ops.noise_block(randomize_noise=self._randomize_noise)
        )
      setattr(self, "{}.resnet_block".format(name), 
        self._resnet_block(
          in_channels=in_channels[block_idx],
          out_channels=out_channels[block_idx],
          scale="up")
      )
      res = seed_size * 2 ** (block_idx+1)
      if name in blocks_with_attention or str(res) in blocks_with_attention:
        blocks_with_attention.discard(name)
        blocks_with_attention.discard(str(res))
        logging.info("[Generator] Applying non-local block at %dx%d resolution to %s",
                     res, res, [None,out_channels[block_idx],res,res])
        setattr(self, "{}.non_local_block".format(name), 
              ops.non_local_block(out_channels[block_idx], use_sn=self._spectral_norm)
        )
    assert len(blocks_with_attention) <= 0

    # Final processing of the net.
    # Use unconditional batch norm.
    logging.info("[Generator] before final processing: %s", [None, out_channels[-1], *self._image_shape[1:]])
    if self._use_noise:
      setattr(self, "final_norm.noise_block", 
        ops.noise_block(randomize_noise=self._randomize_noise)
      )
    setattr(self, "final_norm.batch_norm", 
      ops.batch_norm(out_channels[-1])
    )
    if self._use_relu:
      pass
    else:
      logging.info("[Generator] skipping relu")
    self.final_conv = ops.conv2d(out_channels[-1], output_dim=self._image_shape[0], k_h=3, k_w=3,
                     d_h=1, d_w=1,
                     use_sn=self._spectral_norm)
    logging.info("[Generator] after final processing: %s", [-1,*self._image_shape])

  def forward(self, z, y=None):
    """Forward for the given inputs.
    Args:
      z: `Tensor` of shape [batch_size, z_dim] with latent code.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
        labels.
      is_training: boolean, are we in train or eval model.
    Returns:
      A tensor of size [batch_size] + self._image_shape with values in [0, 1].
    """
    # Each block upscales by a factor of 2.
    seed_size = 4
    z_dim = z.shape[1].value
    num_blocks = self.num_blocks
    
    if self._embed_z:
      z = self.embed_z(z)

    if self._embed_y:
      y = self.embed_y(y)

    y_per_block = num_blocks * [y]
    if self._stylegan_z:
      z_per_block = layers.unstack(z_per_block, axis=1)
      z0, z_per_block = z_per_block[0], z_per_block[1:]
      if y is not None:
        y_per_block = [layers.concat([zi, y], 1) for zi in z_per_block]
    elif self._hierarchical_z:
      z_per_block = layers.split(z, num_blocks + 1, dim=1)
      z0, z_per_block = z_per_block[0], z_per_block[1:]
      if y is not None:
        y_per_block = [layers.concat([zi, y], 1) for zi in z_per_block]
    else:
      z0 = z
      z_per_block = num_blocks * [z]

    # Map noise to the actual seed.
    out = self.fc_noise(z0)
    # Reshape the seed to be a rank-4 Tensor.
    out = layers.reshape(out, [0, -1, seed_size, seed_size])

    blocks_with_attention = set(self._blocks_with_attention)
    for block_idx in range(num_blocks):
      name = "B{}".format(block_idx + 1)
      if self._use_noise:
        out = getattr(self, "{}.noise_block".format(name))(out)
      block = getattr(self, "{}.resnet_block".format(name))
      out = block(
          out,
          z=z_per_block[block_idx],
          y=y_per_block[block_idx])
      res = int(out.shape[2])
      if name in blocks_with_attention or str(res) in blocks_with_attention:
        blocks_with_attention.discard(name)
        blocks_with_attention.discard(str(res))
        out = getattr(self, "{}.non_local_block".format(name))(out)
    
    if self._use_noise:
      out = getattr(self, "final_norm.noise_block")(out)
    out = getattr(self, "final_norm.batch_norm")(out)
    if self._use_relu:
      out = layers.relu(out)
    if self._plain_tanh:
      out = layers.tanh(out)
    else:
      out = (layers.tanh(out) + 1.0) / 2.0
      
    return out
    

@gin.configurable
class Discriminator(abstract_arch.AbstractDiscriminator):
  """ResNet-based discriminator supporting resolutions 32, 64, 128, 256, 512, 1024."""

  def __init__(self,
               ch=96,
               blocks_with_attention="64",
               project_y=True,
               project_y_dim=1000,
               channel_multipliers=None,
               use_noise=False,
               randomize_noise=True,
               **kwargs):
    """Constructor for BigGAN discriminator.
    Args:
      ch: Channel multiplier.
      blocks_with_attention: Comma-separated list of blocks that are followed by
        a non-local block.
      project_y: Add an embedding of y in the output layer.
      project_y_dim: Dimension number of embedding y.
      channel_multipliers: String type list of numbers to multiply with the 'ch' parameter, then get each layer input and output channels. If it's set to None, use the class default channel multiplier. Default: None.
      use_noise: Whether to add noise.
      randomize_noise: Whether to use randomize noise.
      **kwargs: additional arguments past on to ResNetDiscriminator.
    """
    super(Discriminator, self).__init__(**kwargs)
    self._ch = ch
    self._blocks_with_attention = set(blocks_with_attention.split(","))
    self._blocks_with_attention.discard('')
    self._channel_multipliers = None if channel_multipliers is None else [int(x.strip()) for x in channel_multipliers.split(",")]
    self._project_y = project_y
    self._project_y_dim = project_y_dim if project_y else None
    self._use_noise = use_noise
    self._randomize_noise = randomize_noise

  def _resnet_block(self, in_channels, out_channels, scale):
    """ResNet block for the generator."""
    if scale not in ["down", "none"]:
      raise ValueError(
          "Unknown discriminator ResNet block scaling: {}.".format(scale))
    return BigGanResNetBlock(
        y_dim=self._project_y_dim,
        in_channels=in_channels,
        out_channels=out_channels,
        scale=scale,
        is_gen_block=False,
        add_shortcut=in_channels != out_channels,
        layer_norm=self._layer_norm,
        spectral_norm=self._spectral_norm,
        batch_norm=self.batch_norm)

  def _get_in_out_channels(self, colors, resolution):
    colors, resolution = self._image_shape[:2]
    if colors not in [1, 3]:
      raise ValueError("Unsupported color channels: {}".format(colors))
    if self._channel_multipliers is not None:
      channel_multipliers = self._channel_multipliers
    elif resolution == 1024:
      channel_multipliers = [1, 1, 1, 2, 4, 8, 8, 16, 16]
    elif resolution == 512:
      channel_multipliers = [1, 1, 2, 4, 8, 8, 16, 16]
    elif resolution == 256:
      channel_multipliers = [1, 2, 4, 8, 8, 16, 16]
    elif resolution == 128:
      channel_multipliers = [1, 2, 4, 8, 16, 16]
    elif resolution == 64:
      channel_multipliers = [2, 4, 8, 16, 16]
    elif resolution == 32:
      channel_multipliers = [2, 2, 2, 2]
    else:
      raise ValueError("Unsupported resolution: {}".format(resolution))
    out_channels = [self._ch * c for c in channel_multipliers]
    in_channels = [colors] + out_channels[:-1]
    return in_channels, out_channels

  def apply(self):
    """Apply the discriminator on a input.
    """
    with gin.config_scope("discriminator"):
      self._apply()

  def _apply(self):
    in_channels, out_channels = self._get_in_out_channels()
    self.num_blocks = num_blocks = len(in_channels)
    
    logging.info("[Discriminator] inputs are x=%s, y=%s", [None,*self._image_shape],
                     None if y is None else [None,self._project_y_dim])

    blocks_with_attention = set(self._blocks_with_attention)
    for block_idx in range(num_blocks):
      name = "B{}".format(block_idx + 1)
      if self._use_noise:
        setattr(self, "{}.noise_block".format(name),
          ops.noise_block(randomize_noise=self._randomize_noise)
        )
      is_last_block = block_idx == num_blocks - 1
      setattr(self,"{}.resnet_block".format(name), 
        self._resnet_block(
          in_channels=in_channels[block_idx],
          out_channels=out_channels[block_idx],
          scale="none" if is_last_block else "down")
      )
      res = self._image_shape[1] * 2 ** (-(block_idx if not is_last_block else block_idx - 1))
      if name in blocks_with_attention or str(res) in blocks_with_attention:
        blocks_with_attention.discard(name)
        blocks_with_attention.discard(str(res))
        logging.info("[Discriminator] Applying non-local block at %dx%d resolution to %s",
                     res, res, [None,out_channels[block_idx],res,res])
        setattr(self, "{}.non_local_block".format(name), 
          ops.non_local_block(out_channels[block_idx], use_sn=self._spectral_norm)
        )
    assert len(blocks_with_attention) <= 0

    # Final part
    logging.info("[Discriminator] before final processing: %s", [None,out_channels[-1],res,res])
    if self._use_noise:
      setattr(self, "final_fc.noise_block",
        ops.noise_block(randomize_noise=self._randomize_noise)
      )
    setattr(self, "final_fc.logit",
      ops.linear(out_channels[-1], 1, use_sn=self._spectral_norm)
    )
    logging.info("[Discriminator] after final processing: %s", [None,1])
    if self._project_y:
      if self._project_y_dim is None or self._project_y_dim < 1:
        raise ValueError("You must provide class information y to project.")
      y_embedding_dim = out_channels[-1]
      kernel = self.create_parameter(shape=[self._project_y_dim, y_embedding_dim], default_initializer=initializer.Xavier())
      if self._spectral_norm:
        setattr(self, "embedding_fc", 
          ops.spectral_norm(kernel)
        )
      else:
        setattr(self, "embedding_fc.kernel", kernel)
      logging.info("[Discriminator] embedded_y for projection: %s",
                     [None,y_embedding_dim])
    
  def forward(self, x, y=None):
    """Forward for the given inputs.
    Args:
      x: `Tensor` of shape [batch_size, ?, ?, ?] with real or fake images.
      y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
            labels.
      Returns:
        Tuple of 3 Tensors, the final prediction of the discriminator, the logits
        before the final output activation function and logits form the second
          last layer.
    """
    out = x
    num_blocks = self.num_blocks
    
    blocks_with_attention = set(self._blocks_with_attention)
    for block_idx in range(num_blocks):
      name = "B{}".format(block_idx + 1)
      if self._use_noise:
        out = getattr(self, "{}.noise_block")(out)
      is_last_block = block_idx == num_blocks - 1
      out = getattr(self,"{}.resnet_block")(out, z=None, y=y)
      res = int(out.shape[2])
      if name in blocks_with_attention or str(res) in blocks_with_attention:
        blocks_with_attention.discard(name)
        blocks_with_attention.discard(str(res))
        out = getattr(self, "{}.non_local_block")(out)
        assert len(blocks_with_attention) <= 0
    
    # Final part
    if self._use_noise:
      out = getattr(self, "final_fc.noise_block")(out)
    net = layers.relu(net)
    h = layers.reduce_sum(net, dim=[-1, -2])
    out_logit = getattr(self, "final_fc.logit")(h)
    if self._project_y:
      if y is None:
        raise ValueError("You must provide class information y to project.")
      if self._spectral_norm:
        kernel, norm = getattr(self, "embedding_fc")()
      else:
        kernel = getattr(self, "embedding_fc.kernel")
      embedded_y = layers.matmul(y, kernel)
      out_logit = out_logit + layers.reduce_sum(embedded_y * h, dim=1, keep_dim=True)
    out = layers.sigmoid(out_logit)
    return out, out_logit, h