from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from network import arch_ops as ops

from six.moves import range
import tensorflow as tf
from network.resnet_biggan import Generator, Discriminator, BigGanResNetBlock


class sampler(Generator):
    def __init__(self, **kwargs):
        super(sampler, self).__init__(**kwargs)

    def apply(self, z, y, is_training):
        """Build the generator network for the given inputs.

        Args:
          z: `Tensor` of shape [batch_size, z_dim] with latent code.
          y: `Tensor` of shape [batch_size, num_classes] with one hot encoded
            labels.
          is_training: boolean, are we in train or eval model.

        Returns:
          A tensor of size [batch_size] + self._image_shape with values in [0, 1].
        """
        shape_or_none = lambda t: None if t is None else t.shape
        logging.info("[Generator] inputs are z=%s, y=%s", z.shape, shape_or_none(y))
        # Each block upscales by a factor of 2.
        seed_size = 4
        z_dim = z.shape[1].value

        sample_dict = {'z': z}

        in_channels, out_channels = self._get_in_out_channels()
        num_blocks = len(in_channels)

        if self._embed_z:
            z = ops.linear(z, z_dim, scope="embed_z", use_sn=False,
                           use_bias=self._embed_bias)
        if self._embed_y:
            y = ops.linear(y, self._embed_y_dim, scope="embed_y", use_sn=False,
                           use_bias=self._embed_bias)

        sample_dict.update({'embed_z': z})
        y_per_block = num_blocks * [y]
        if self._hierarchical_z:
            z_per_block = tf.split(z, num_blocks + 1, axis=1)
            z0, z_per_block = z_per_block[0], z_per_block[1:]
            if y is not None:
                y_per_block = [tf.concat([zi, y], 1) for zi in z_per_block]
        else:
            z0 = z
            z_per_block = num_blocks * [z]

        logging.info("[Generator] z0=%s, z_per_block=%s, y_per_block=%s",
                     z0.shape, [str(shape_or_none(t)) for t in z_per_block],
                     [str(shape_or_none(t)) for t in y_per_block])

        # Map noise to the actual seed.
        net = ops.linear(
            z0,
            in_channels[0] * seed_size * seed_size,
            scope="fc_noise",
            use_sn=self._spectral_norm)

        sample_dict.update({'linear_z0': net})
        # Reshape the seed to be a rank-4 Tensor.
        net = tf.reshape(
            net,
            [-1, seed_size, seed_size, in_channels[0]],
            name="fc_reshaped")

        for block_idx in range(num_blocks):
            name = "B{}".format(block_idx + 1)
            block = self._resnet_block(
                name=name,
                in_channels=in_channels[block_idx],
                out_channels=out_channels[block_idx],
                scale="up")
            net = block(
                net,
                z=z_per_block[block_idx],
                y=y_per_block[block_idx],
                is_training=is_training)
            sample_dict.update({'ResBlock%d' % block_idx: net})
            if name in self._blocks_with_attention:
                logging.info("[Generator] Applying non-local block to %s", net.shape)
                net = ops.non_local_block(net, "non_local_block",
                                          use_sn=self._spectral_norm)
                sample_dict.update({"ResNonLocal%d" % block_idx: net})
        # Final processing of the net.
        # Use unconditional batch norm.
        logging.info("[Generator] before final processing: %s", net.shape)
        net = ops.batch_norm(net, is_training=is_training, name="final_norm")
        net = tf.nn.relu(net)
        net = ops.conv2d(net, output_dim=self._image_shape[2], k_h=3, k_w=3,
                         d_h=1, d_w=1, name="final_conv",
                         use_sn=self._spectral_norm)
        logging.info("[Generator] after final processing: %s", net.shape)
        net = (tf.nn.tanh(net) + 1.0) / 2.0
        sample_dict.update({'img': net})
        return sample_dict

