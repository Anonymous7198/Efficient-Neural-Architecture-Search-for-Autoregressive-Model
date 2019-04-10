from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time


import numpy as np
import tensorflow as tf

import image_ops as nn

from utils import count_model_params
from utils import get_train_ops
from common_ops import create_weight

import data.cifar10_data as cifar10_data
from tensorflow.contrib.framework.python.ops import arg_scope



class MicroChild(object):
  def __init__(self,
               num_layers=2,
               num_cells=5,
               num_gpu=8,
               out_filters=256,
               key_size=16,
               value_size=128,
               keep_prob=1.0,
               drop_path_keep_prob=None,
               nr_logistic_mix=10,
               learning_rate=0.001,
               lr_decay=0.999995,
               batch_size=16,
               num_train_batch=391,
               polyak_decay=0.9995,
               num_epochs=None,
               seed=None,
               log_interval=1,
               name="child",
               data_dir=None,
               **kwargs
              ):  

    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.nr_logistic_mix = nr_logistic_mix
    self.lr_decay = lr_decay
    self.learning_rate = learning_rate
    self.keep_prob = keep_prob
    self.drop_path_keep_prob = drop_path_keep_prob
    self.out_filters = out_filters
    self.num_layers = num_layers
    self.num_cells = num_cells
    self.key_size = key_size
    self.value_size = value_size
    self.seed = seed
    self.num_gpu = num_gpu
    self.polyak_decay = polyak_decay
    self.log_interval = log_interval
    self.data_dir = data_dir
    self.name = name
    self.num_train_batch = num_train_batch

    self.global_step = tf.Variable(
      0, dtype=tf.int32, trainable=False, name="global_step")

    if self.drop_path_keep_prob is not None:
      assert num_epochs is not None, "Need num_epochs to drop_path"

    self.rng = np.random.RandomState(self.seed)
    tf.set_random_seed(self.seed)


    print("Build data ops")
    self.DataLoader = cifar10_data.DataLoader
    # training data
    self.train_data = self.DataLoader(self.data_dir, 'train', self.batch_size * self.num_gpu, rng=self.rng, shuffle=True, return_labels=None)
    self.test_data = self.DataLoader(self.data_dir, 'test', self.batch_size * self.num_gpu, rng=self.rng, shuffle=True, return_labels=None)
    self.obs_shape = self.train_data.get_observation_size() # e.g. a tuple (32,32,3)
    assert len(self.obs_shape) == 3, 'assumed right now'

    self.xs = [tf.placeholder(tf.float32, shape=(self.batch_size, ) + self.obs_shape) for i in range(self.num_gpu)]
    self.tf_lr = tf.placeholder(tf.float32, shape=[])

    self.loss_func = nn.discretized_mix_logistic_loss

    self.child_params = [var for var in tf.trainable_variables() if 'child' in var.name]
    self.ema = tf.train.ExponentialMovingAverage(decay=self.polyak_decay)
    self.maintain_averages_op = tf.group(self.ema.apply(self.child_params))
    self.ema_params = [self.ema.average(p) for p in self.child_params]

    self.num_train_steps = self.num_epochs * self.num_train_batch
  
  def _apply_drop_path(self, x, layer_id):
    drop_path_keep_prob = self.drop_path_keep_prob

    layer_ratio = (layer_id + 1) / float(self.num_layers)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)

    step_ratio = tf.to_float(self.global_step + 1) / tf.to_float(self.num_train_steps)
    step_ratio = tf.minimum(1.0, step_ratio)
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)

    x = nn.drop_path(x, drop_path_keep_prob)
    return x


  def _model(self, x, ema=None, is_training=False):
    """Computes a discretized mixture of logistics given image-conditionals"""

    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      # the first two inputs
      counters = {}
      with arg_scope([nn.conv2d, nn.dense, nn.DiagonalwiseRefactorization, nn.grouped_conv2d], counters=counters, ema=ema):
                    
        out_filters = self.out_filters
        xs = nn.int_shape(x)
        # add channel of ones to distinguish image from padding later on
        x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)

        skip_outputs = []

        with tf.variable_scope("stem_conv"):
          x_pad = nn.causal_shift_nin(nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=out_filters, filter_size=[2,3])), num_filters=out_filters) + nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=out_filters, filter_size=[1,2]))
        
        layers = [x_pad, x_pad]
        # building layers in the micro space
        for layer_id in range(self.num_layers):
          with tf.variable_scope("layer_{0}".format(layer_id)):
            x = self._enas_layer(
            layer_id, layers, self.arc, out_filters, is_training)
          
            skip_outputs.append(x)
            out = x + nn.causal_shift_nin(layers[-1], num_filters=out_filters)
          
          print("Layer {0:>2d}: {1}".format(layer_id, x))
          layers = [layers[-1], out]

      # sum -> Relu -> 1x1 -> Relu -> 1x1 -> MixtureofLogistic
      with tf.variable_scope("post_processing"):
        total = sum(skip_outputs)

        transformed1 = nn.layer_norm(tf.nn.relu(total))
        conv1 = nn.causal_shift_nin(transformed1, num_filters=out_filters)

        transformed2 = nn.layer_norm(tf.nn.relu(total))
        conv2 = nn.causal_shift_nin(transformed2, num_filters=10*self.nr_logistic_mix)

    return conv2

  def apply_act(self, x, act_id):

    out = [
      x * tf.nn.sigmoid(x),
      tf.nn.relu(x),
      tf.nn.sigmoid(x),
      tf.nn.tanh(x),
      tf.nn.leaky_relu(x),
      x
    ]

    out = tf.stack(out , axis=0)
    out = out[act_id,:,:,:,:]

    return out

  def apply_op(self, x, out_filters, op_id):

    with tf.variable_scope('causal_conv_3x3'):
      conv_1 = nn.causal_conv_3x3(x, num_filters=out_filters, stride=[1, 1]) #3x3 causal conv

    with tf.variable_scope('causal_conv_5x5'):
      conv_2 = nn.causal_conv_5x5(x, num_filters=out_filters, stride=[1, 1]) #5x5 causal conv

    with tf.variable_scope('causal_depthwise_conv_3x3'):
      depthwise_1 = nn.causal_depthwise_conv_3x3(x, num_filters=out_filters, stride=[1,1])

    with tf.variable_scope('causal_depthwise_conv_5x5'):
      depthwise_2 = nn.causal_depthwise_conv_5x5(x, num_filters=out_filters, stride=[1,1]) 

    with tf.variable_scope('causal_group_conv_3x3'):
      group_1 = nn.causal_group_conv_3x3(x, num_filters=out_filters, stride=[1,1])

    with tf.variable_scope('causal_group_conv_5x5'):
      group_2 = nn.causal_group_conv_5x5(x, num_filters=out_filters, stride=[1,1])

    with tf.variable_scope('causal_attention'):
      key   = nn.causal_shift_nin(x, num_filters=self.key_size)
      query = nn.causal_shift_nin(x, num_filters=self.key_size)
      value = nn.causal_shift_nin(x, num_filters=self.value_size)

      attention = nn.causal_shift_nin(nn.causal_attention(key, value, query), num_filters=out_filters) #causal self-attention

    out = [
      conv_1,
      conv_2,
      depthwise_1,
      depthwise_2,
      group_1,
      group_2,
      attention,
      x
    ]

    out = tf.stack(out, axis=0)
    out = out[op_id,:,:,:,:]

    return out
    

  #norm->conv->act
  def _enas_cell(self, x, op_id, act_id, out_filters):
    """Performs an enas operation specified by op_id & act_id."""
    return self.apply_act(self.apply_op(nn.layer_norm(x), out_filters, op_id), act_id)


  def _enas_layer(self, layer_id, prev_layers, arc, out_filters, is_training):
    """
    Args:
      layer_id: current layer
      prev_layers: cache of previous layers. for skip connections
      start_idx: where to start looking at. technically, we can infer this
        from layer_id, but why bother...
    """

    assert len(prev_layers) == 2, "need exactly 2 inputs"
    layers = [prev_layers[0], prev_layers[1]]
    used = []
    for cell_id in range(self.num_cells):
      prev_layers = tf.stack(layers, axis=0)
      with tf.variable_scope("cell_{0}".format(cell_id)):
        with tf.variable_scope("x"):
          x_id = arc[7 * cell_id]
          x_op = arc[7 * cell_id + 2]
          x_act = arc[7 * cell_id + 4]
          x = prev_layers[x_id, :, :, :, :]
          x = self._enas_cell(x, x_op, x_act, out_filters)


          if (x_op in [0, 1, 2, 3, 4, 5, 6, 7] and self.drop_path_keep_prob is not None and is_training==True):
            x = self._apply_drop_path(x, layer_id)

          x_used = tf.one_hot(x_id, depth=self.num_cells + 2, dtype=tf.int32)

        with tf.variable_scope("y"):
          y_id = arc[7 * cell_id + 1]
          y_op = arc[7 * cell_id + 3]
          y_act = arc[7 * cell_id + 5]
          y = prev_layers[y_id, :, :, :, :]
          y = self._enas_cell(y, y_op, y_act, out_filters)

          if (y_op in [0, 1, 2, 3, 4, 5, 6, 7] and self.drop_path_keep_prob is not None):
            y = self._apply_drop_path(y, layer_id)

          y_used = tf.one_hot(y_id, depth=self.num_cells + 2, dtype=tf.int32)
        

        out =  [x + y, x * y, nn.causal_shift_nin(tf.concat([x,y], axis=3), num_filters=out_filters)]

        comb_id = arc[7 * cell_id + 6]

        out = tf.stack(out, axis=0)
        out = out[comb_id,:,:,:,:]

        layers.append(out)
        used.append(out)

    out = tf.concat(used, axis=3)

    out = nn.causal_shift_nin(out, num_filters=out_filters)   

    return out

  def make_feed_dict(self, data, init=False):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    if init:
        feed_dict = {self.x_init: x}
        if y is not None:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, self.num_gpu)
        feed_dict = {self.xs[i]: x[i] for i in range(self.num_gpu)}
        if y is not None:
            y = np.split(y, self.num_gpu)
            feed_dict.update({ys[i]: y[i] for i in range(self.num_gpu)})
    return feed_dict


  def build_train_child(self):

    self.grad = []
    self.loss_gen_train = []

    self.increment_op = tf.assign_add(self.global_step, 1)

    for i in range(self.num_gpu):
      with tf.device('/gpu:%d' % i):
        #train
        out = self._model(self.xs[i], is_training=True)
        self.loss_gen_train.append(self.loss_func(self.xs[i], out))
        #grad
        self.grad.append(nn.replace_none_with_zero(tf.gradients(self.loss_gen_train[i], self.child_params, colocate_gradients_with_ops=True)))
        
    with tf.device('/gpu:0'):
      for i in range(1,self.num_gpu):
          self.loss_gen_train[0] += self.loss_gen_train[i]
          for j in range(len(self.grad[0])):
              self.grad[0][j] += self.grad[i][j]
      # training op
      self.optimizer = tf.group(nn.adam_updates(self.child_params, self.grad[0], lr=self.tf_lr, mom1=0.95, mom2=0.9995), self.maintain_averages_op)

    #convert loss to bits/dim
    self.bits_per_dim = self.loss_gen_train[0]/(self.num_gpu*np.log(2.)*np.prod(self.obs_shape)*self.batch_size)

    return self.bits_per_dim, self.optimizer, self.increment_op

  def build_evaluate(self):

    self.loss_gen_validation = []
    
    for i in range(self.num_gpu):
      with tf.device('/gpu:%d' % i):
        #likelihood
        out = self._model(self.xs[i], ema=self.ema)
        self.loss_gen_validation.append(self.loss_func(self.xs[i], out))

    with tf.device('/gpu:0'):
      for i in range(1, self.num_gpu):
        self.loss_gen_validation[0] += self.loss_gen_validation[i]
    
    #bit_per_dim
    self.bits_per_dim_test = self.loss_gen_validation[0]/(self.num_gpu*np.log(2.)*np.prod(self.obs_shape)*self.batch_size)


  def connect_controller(self, controller_model):
    sample_arc = controller_model._sample_controller()
    self.arc = sample_arc
    return sample_arc

