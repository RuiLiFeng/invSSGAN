from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from network.invert import G_mapping_ND


class Generator(object):

  def __init__(self, module_spec, trainable=True):
    self._module_spec = module_spec
    self._trainable = trainable
    self._module = hub.Module(self._module_spec, name="gen_module",
                              tags={"gen", "bsNone"}, trainable=self._trainable)
    self.input_info = self._module.get_input_info_dict()

  def __call__(self, z):
    """
    Build tensorflow graph for Generator
    :param input_dict: {'z_': <hub.ParsedTensorInfo shape=(?, 120) dtype=float32 is_sparse=False>,
    'labels': None or (?,)}
    :return:{'generated': <hub.ParsedTensorInfo shape=(?, 128, 128, 3) dtype=float32 is_sparse=False>}
    """
    inv_input = {'z': z}
    self.samples = self._module(inputs=inv_input, as_dict=True)['generated']
    return self.samples

  @property
  def trainable_variables(self):
    return [var for var in tf.trainable_variables() if 'generator' in var.name]


class Discriminator(object):
  def __init__(self, module_spec, trainable=True):
    self._module_spec = module_spec
    self._trainable = trainable
    self._module = hub.Module(self._module_spec, name="disc_module",
                              tags={"disc", "bsNone"}, trainable=self._trainable)
    self.input_info = self._module.get_input_info_dict()

  def build_graph(self, input_dict):
    """

    :param input_dict: {'images': <hub.ParsedTensorInfo shape=(?, 128, 128, 3) dtype=float32 is_sparse=False>}
    :return: {'pre_logits': <hub.ParsedTensorInfo shape=(?, 1536) dtype=float32 is_sparse=False>,
              'prediction': <hub.ParsedTensorInfo shape=(?, 1) dtype=float32 is_sparse=False>,
              'logits': <hub.ParsedTensorInfo shape=(?, 1) dtype=float32 is_sparse=False>}
    """
    output = self._module(inputs=input_dict, as_dict=True)
    self.logits = output['logits']
    self.prop = output['prediction']
    del output['pre_logits']
    return self.logits, self.prop

  @property
  def trainable_variables(self):
    return [var for var in tf.trainable_variables() if 'discriminator' in var.name]


