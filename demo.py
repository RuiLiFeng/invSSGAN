# @title Imports and utility functions

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import IPython
from IPython.display import display
import numpy as np
import PIL.Image
import pandas as pd
import six
import tensorflow as tf
import tensorflow_hub as hub


def imgrid(imarray, cols=8, pad=1):
  pad = int(pad)
  assert pad >= 0
  cols = int(cols)
  assert cols >= 1
  N, H, W, C = imarray.shape
  rows = int(np.ceil(N / float(cols)))
  batch_pad = rows * cols - N
  assert batch_pad >= 0
  post_pad = [batch_pad, pad, pad, 0]
  pad_arg = [[0, p] for p in post_pad]
  imarray = np.pad(imarray, pad_arg, 'constant')
  H += pad
  W += pad
  grid = (imarray
          .reshape(rows, cols, H, W, C)
          .transpose(0, 2, 1, 3, 4)
          .reshape(rows*H, cols*W, C))
  return grid[:-pad, :-pad]


def imshow(a, format='png', jpeg_fallback=True):
  a = np.asarray(a, dtype=np.uint8)
  if six.PY3:
    str_file = six.BytesIO()
  else:
    str_file = six.StringIO()
  PIL.Image.fromarray(a).save(str_file, format)
  png_data = str_file.getvalue()
  try:
    disp = display(IPython.display.Image(png_data))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print ('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format)
      return imshow(a, format='jpeg')
    else:
      raise
  return disp


class Generator(object):

  def __init__(self, module_spec):
    self._module_spec = module_spec
    self._sess = None
    self._graph = tf.Graph()
    self._load_model()

  @property
  def z_dim(self):
    return self._z.shape[-1].value

  @property
  def conditional(self):
    return self._labels is not None

  def _load_model(self):
    with self._graph.as_default():
      self._generator = hub.Module(self._module_spec, name="gen_module",
                                   tags={"gen", "bsNone"})
      input_info = self._generator.get_input_info_dict()
      inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
                for k, v in self._generator.get_input_info_dict().items()}
      self._samples = self._generator(inputs=inputs, as_dict=True)["generated"]
      print("Inputs:", inputs)
      print("Outputs:", self._samples)
      self._z = inputs["z"]
      self._labels = inputs.get("labels", None)

  def _init_session(self):
    if self._sess is None:
      self._sess = tf.Session(graph=self._graph)
      self._sess.run(tf.global_variables_initializer())

  def get_noise(self, num_samples, seed=None):
    if np.isscalar(seed):
      np.random.seed(seed)
      return np.random.normal(size=[num_samples, self.z_dim])
    z = np.empty(shape=(len(seed), self.z_dim), dtype=np.float32)
    for i, s in enumerate(seed):
      np.random.seed(s)
      z[i] = np.random.normal(size=[self.z_dim])
    return z

  def get_samples(self, z, labels=None):
    with self._graph.as_default():
      self._init_session()
      feed_dict = {self._z: z}
      if self.conditional:
        assert labels is not None
        assert labels.shape[0] == z.shape[0]
        feed_dict[self._labels] = labels
      samples = self._sess.run(self._samples, feed_dict=feed_dict)
      return np.uint8(np.clip(256 * samples, 0, 255))


class Discriminator(object):

  def __init__(self, module_spec):
    self._module_spec = module_spec
    self._sess = None
    self._graph = tf.Graph()
    self._load_model()

  @property
  def conditional(self):
    return "labels" in self._inputs

  @property
  def image_shape(self):
    return self._inputs["images"].shape.as_list()[1:]

  def _load_model(self):
    with self._graph.as_default():
      self._discriminator = hub.Module(self._module_spec, name="disc_module",
                                       tags={"disc", "bsNone"})
      input_info = self._discriminator.get_input_info_dict()
      self._inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k)
                      for k, v in input_info.items()}
      self._outputs = self._discriminator(inputs=self._inputs, as_dict=True)
      print("Inputs:", self._inputs)
      print("Outputs:", self._outputs)

  def _init_session(self):
    if self._sess is None:
      self._sess = tf.Session(graph=self._graph)
      self._sess.run(tf.global_variables_initializer())

  def predict(self, images, labels=None):
    with self._graph.as_default():
      self._init_session()
      feed_dict = {self._inputs["images"]: images}
      if "labels" in self._inputs:
        assert labels is not None
        assert labels.shape[0] == images.shape[0]
        feed_dict[self._inputs["labels"]] = labels
      return self._sess.run(self._outputs, feed_dict=feed_dict)


# @title Load Model

model_name = "SSGAN 128x128 (FID 20.6, IS 24.9)"
models = {
    "SSGAN 128x128": "https://tfhub.dev/google/compare_gan/ssgan_128x128/1",
}

module_spec = models[model_name.split(" (")[0]]
print("Module spec:", module_spec)

tf.reset_default_graph()
print("Loading model...")
sampler = Generator(module_spec)
print("Model loaded.")

# @title Sampling { run: "auto" }

num_rows = 3  # @param {type: "slider", min:1, max:16}
num_cols = 4  # @param {type: "slider", min:1, max:16}
noise_seed = 23  # @param {type:"slider", min:0, max:100, step:1}

num_samples = num_rows * num_cols
z = sampler.get_noise(num_samples, seed=noise_seed)

samples = sampler.get_samples(z)
imshow(imgrid(samples, cols=num_cols))

# @title Discriminator

disc = Discriminator(module_spec)

batch_size = 4
num_classes = 1000
images = np.random.random(size=[batch_size] + disc.image_shape)

disc.predict(images)