from paddle import fluid

from pmgan.architectures.resnet_biggan import Generator, Discriminator


place = fluid.CPUPlace()# fluid.CUDAPlace(fluid.dygraph.ParallelEnv().dev_id)
fluid.enable_dygraph(place)

g_256 = Generator(ch=8, image_shape=[3,256,256])