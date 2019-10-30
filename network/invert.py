import numpy as np
import tensorflow as tf


class InvMap(object):
    def __init__(self, latent_size=120, depth=4, dtype='float32'):
        self._latent_size = latent_size
        self._depth = depth
        self._dtype = dtype

    @property
    def trainable_variables(self):
        return [v for v in tf.global_variables() if 'invert' in v.name]

    def __call__(self, latent_in, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('invert', values=[latent_in], reuse=reuse):
            output = G_mapping_ND(latent_in, latent_size=self._latent_size, depth=self._depth,
                                  dtype=self._dtype)
        return output


def G_mapping_ND(
    latents_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    latent_size             = 120,          # Latent vector (Z) dimensionality.
    hidden_width            = 120,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = None,         # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
    depth                   = 4,            # Number of mapping layers.
    dtype                   = 'float32',    # Data type to use for activations and outputs.
    **_kwargs):                             # Ignore unrecognized keyword args.

    # Inputs.
    latents_in.set_shape([None, latent_size])
    latents_in = tf.cast(latents_in, dtype)
    net = latents_in

    # mapping layers.
    for i in reversed(range(depth)):
        net = step(str(i), z=net,  width=hidden_width, reverse=True)

    # Broadcast.
    if dlatent_broadcast is not None:
        with tf.variable_scope('Broadcast'):
            net = tf.tile(net[:, np.newaxis], [1, dlatent_broadcast, 1])

    # Output.
    assert net.dtype == tf.as_dtype(dtype)
    return tf.identity(net, name='dlatents_out')


def G_mapping_NE(
    feature_in,                             # First input: Latent vectors (Z) [minibatch, latent_size].
    feature_size            = 120,          # Latent vector (Z) dimensionality.
    hidden_width            = 120,          # Disentangled latent (W) dimensionality.
    dlatent_broadcast       = 1,            # Output disentangled latent (W) as [minibatch, dlatent_size] or [minibatch, dlatent_broadcast, dlatent_size].
    depth                   = 4,            # Number of mapping layers.
    dtype                   = 'float32',    # Data type to use for activations and outputs.
    **_kwargs):                             # Ignore unrecognized keyword args.

    # Inputs.
    feature_in.set_shape([None, feature_size])
    feature_in = tf.cast(feature_in, dtype)
    net = feature_in

    # mapping layers w==>e.
    for i in range(depth):
        net = step(str(i), z=net,  width=hidden_width, reverse=False)

    # Output.
    assert net.dtype == tf.as_dtype(dtype)
    return tf.identity(net, name='dlatents_out')


def f(name, x, width, n_out=None, use_wscale=True, mapping_lrmul=0.01):
    with tf.variable_scope(name):
        with tf.variable_scope('dense1'):
            x = dense(x, fmaps=width, gain=np.sqrt(2), use_wscale=use_wscale, lrmul=mapping_lrmul)
            x = apply_bias(x, lrmul=mapping_lrmul)
            x = leaky_relu(x)
        with tf.variable_scope('dense2'):
            x = dense(x, fmaps=n_out, gain=np.sqrt(2), use_wscale=use_wscale, lrmul=mapping_lrmul)
            x = apply_bias(x, lrmul=mapping_lrmul)
            x = leaky_relu(x)
    return x


def reverse_features(h):
    return h[:, ::-1]

flow_coupling = 0
def step(name_scope, z, width, reverse):
    with tf.variable_scope(name_scope):

        n_z = z.get_shape()[1]

        if not reverse:

            z = reverse_features(z)

            z1 = z[:, :n_z // 2]
            z2 = z[:, n_z // 2:]

            #z2 += f("f_inv", z1, width, n_out=n_z//2)
            if flow_coupling == 0:
                z2 += f("f_inv", z1, width, n_out=n_z//2)
            elif flow_coupling == 1:
                h = f("f_inv", z1, width, n_out=n_z)
                shift = h[:, 0::2]
                scale = tf.nn.sigmoid(h[:, 1::2] + 2.)
                z2 += shift
                z2 *= scale

            z = tf.concat([z1, z2], 1)

        else:

            z1 = z[:, :n_z // 2]
            z2 = z[:, n_z // 2:]

            #z2 -= f("f_inv", z1, width, n_out=n_z//2)
            if flow_coupling == 0:
                z2 -= f("f_inv", z1, width, n_out=n_z//2)
            elif flow_coupling == 1:
                h = f("f_inv", z1, width, n_out=n_z)
                shift = h[:, 0::2]
                scale = tf.nn.sigmoid(h[:, 1::2] + 2.)
                z2 /= scale
                z2 -= shift

            z = tf.concat([z1, z2], 1)

            z = reverse_features(z)

    return z


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable('weight', shape=shape, initializer=init) * runtime_coef

#----------------------------------------------------------------------------
# Fully-connected layer.

def dense(x, fmaps, **kwargs):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Apply bias to the given activation tensor.

def apply_bias(x, lrmul=1.):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. More efficient than tf.nn.leaky_relu() and supports FP16.

def leaky_relu(x, alpha=0.2):
    with tf.variable_scope('LeakyReLU'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        @tf.custom_gradient
        def func(x):
            y = tf.maximum(x, x * alpha)
            @tf.custom_gradient
            def grad(dy):
                dx = tf.where(y >= 0, dy, dy * alpha)
                return dx, lambda ddx: tf.where(y >= 0, ddx, ddx * alpha)
            return y, grad
        return func(x)
