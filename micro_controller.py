from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import time

import numpy as np
import tensorflow as tf

from controller import Controller
from utils import get_train_ops
from common_ops import stack_lstm

from tensorflow.python.training import moving_averages

class MicroController(Controller):
  def __init__(self,
               num_branches_conv=8,
               num_branches_act=6,
               num_branches_comb=3,
               num_cells=5,
               lstm_size=32,
               lstm_num_layers=2,
               lstm_keep_prob=1.0,
               tanh_constant=None,
               op_tanh_reduce=1.0,
               temperature=None,
               lr_init=1e-3,
               lr_dec_start=0,
               lr_dec_every=100,
               lr_dec_rate=0.9,
               l2_reg=0,
               entropy_weight=0.4,
               clip_mode=None,
               grad_bound=None,
               use_critic=False,
               bl_dec=0.999,
               optim_algo="adam",
               name="controller",
               **kwargs):

    print("-" * 80)
    print("Building ConvController")

    self.num_cells = num_cells
    self.num_branches_conv = num_branches_conv
    self.num_branches_act = num_branches_act
    self.num_branches_comb = num_branches_comb

    self.lstm_size = lstm_size
    self.lstm_num_layers = lstm_num_layers 
    self.lstm_keep_prob = lstm_keep_prob
    self.tanh_constant = tanh_constant
    self.op_tanh_reduce = op_tanh_reduce
    self.temperature = temperature
    self.lr_init = lr_init
    self.lr_dec_start = lr_dec_start
    self.lr_dec_every = lr_dec_every
    self.lr_dec_rate = lr_dec_rate
    self.l2_reg = l2_reg
    self.clip_mode = clip_mode
    self.grad_bound = grad_bound
    self.use_critic = use_critic
    self.bl_dec = bl_dec
    self.entropy_weight = entropy_weight

    self.optim_algo = optim_algo
    self.name = name

    self._create_params()

  def _sample_controller(self):
    sample_arc, entropy, log_prob, c, h = self._build_sampler(use_bias=True)
    self.arc = sample_arc
    self.sample_log_prob = log_prob
    self.sample_entropy = entropy

    return sample_arc


  def _create_params(self):
    initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
    with tf.variable_scope(self.name, initializer=initializer, reuse=tf.AUTO_REUSE):
      with tf.variable_scope("lstm"):
        self.w_lstm = []
        for layer_id in range(self.lstm_num_layers):
          with tf.variable_scope("layer_{}".format(layer_id)):
            w = tf.get_variable("w", [2 * self.lstm_size, 4 * self.lstm_size])
            self.w_lstm.append(w)

      with tf.variable_scope("emb", reuse=tf.AUTO_REUSE):
        self.w_emb_conv = tf.get_variable("w_conv", [self.num_branches_conv, self.lstm_size])
        self.w_emb_act = tf.get_variable("w_act", [self.num_branches_act, self.lstm_size])
        self.w_emb_comb = tf.get_variable("w_comb", [self.num_branches_comb, self.lstm_size])
        self.g_emb = tf.get_variable("g_emb", [1, self.lstm_size])

      
      with tf.variable_scope("softmax", reuse=tf.AUTO_REUSE):
        #weights for conv ops
        self.w_soft_conv = tf.get_variable("w_conv", [self.lstm_size, self.num_branches_conv])
        b_init_conv = np.array([10.0, 10.0] + [0] * (self.num_branches_conv - 2),
                          dtype=np.float32)
        self.b_soft_conv = tf.get_variable(
          "b_conv", [1, self.num_branches_conv],
          initializer=tf.constant_initializer(b_init_conv))

        b_soft_no_learn_conv = np.array(
          [0.25, 0.25] + [-0.25] * (self.num_branches_conv - 2), dtype=np.float32)
        b_soft_no_learn_conv = np.reshape(b_soft_no_learn_conv, [1, self.num_branches_conv])
        self.b_soft_no_learn_conv = tf.constant(b_soft_no_learn_conv, dtype=tf.float32)

        #weights for act ops
        self.w_soft_act = tf.get_variable("w_act", [self.lstm_size, self.num_branches_act])
        b_init_act = np.array([10.0, 10.0] + [0] * (self.num_branches_act - 2),
                          dtype=np.float32)
        self.b_soft_act = tf.get_variable(
          "b_act", [1, self.num_branches_act],
          initializer=tf.constant_initializer(b_init_act))

        b_soft_no_learn_act = np.array(
          [0.25, 0.25] + [-0.25] * (self.num_branches_act - 2), dtype=np.float32)
        b_soft_no_learn_act = np.reshape(b_soft_no_learn_act, [1, self.num_branches_act])
        self.b_soft_no_learn_act = tf.constant(b_soft_no_learn_act, dtype=tf.float32)

        #weights for combine ops
        self.w_soft_comb = tf.get_variable("w_comb", [self.lstm_size, self.num_branches_comb])
        b_init_comb = np.array([10.0, 10.0] + [0] * (self.num_branches_comb - 2),
                          dtype=np.float32)
        self.b_soft_comb = tf.get_variable(
          "b_comb", [1, self.num_branches_comb],
          initializer=tf.constant_initializer(b_init_comb))

        b_soft_no_learn_comb = np.array(
          [0.25, 0.25] + [-0.25] * (self.num_branches_comb - 2), dtype=np.float32)
        b_soft_no_learn_comb = np.reshape(b_soft_no_learn_comb, [1, self.num_branches_comb])
        self.b_soft_no_learn_comb = tf.constant(b_soft_no_learn_comb, dtype=tf.float32)

      with tf.variable_scope("attention", reuse=tf.AUTO_REUSE):
        self.w_attn_1 = tf.get_variable("w_1", [self.lstm_size, self.lstm_size])
        self.w_attn_2 = tf.get_variable("w_2", [self.lstm_size, self.lstm_size])
        self.v_attn = tf.get_variable("v", [self.lstm_size, 1])

  def _build_sampler(self, prev_c=None, prev_h=None, use_bias=False):
    """Build the sampler ops and the log_prob ops."""

    print("-" * 80)
    print("Build controller sampler")

    anchors = tf.TensorArray(
      tf.float32, size=self.num_cells + 2, clear_after_read=False)
    anchors_w_1 = tf.TensorArray(
      tf.float32, size=self.num_cells + 2, clear_after_read=False)
    arc_seq = tf.TensorArray(tf.int32, size=self.num_cells * 7)
    if prev_c is None:
      assert prev_h is None, "prev_c and prev_h must both be None"
      prev_c = [tf.zeros([1, self.lstm_size], tf.float32)
                for _ in range(self.lstm_num_layers)]
      prev_h = [tf.zeros([1, self.lstm_size], tf.float32)
                for _ in range(self.lstm_num_layers)]
    inputs = self.g_emb

    for layer_id in range(2):
      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
      prev_c, prev_h = next_c, next_h
      anchors = anchors.write(layer_id, tf.zeros_like(next_h[-1]))
      anchors_w_1 = anchors_w_1.write(
        layer_id, tf.matmul(next_h[-1], self.w_attn_1))

    def _condition(layer_id, *args):
      return tf.less(layer_id, self.num_cells + 2)

    def _body(layer_id, inputs, prev_c, prev_h, anchors, anchors_w_1, arc_seq,
              entropy, log_prob):
      indices = tf.range(0, layer_id, dtype=tf.int32)
      start_id = 7 * (layer_id - 2)
      prev_layers = []
      for i in range(2):  # index_1, index_2
        next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
        prev_c, prev_h = next_c, next_h
        query = anchors_w_1.gather(indices)
        query = tf.reshape(query, [layer_id, self.lstm_size])
        query = tf.tanh(query + tf.matmul(next_h[-1], self.w_attn_2))
        query = tf.matmul(query, self.v_attn)
        logits = tf.reshape(query, [1, layer_id])
        if self.temperature is not None:
          logits /= self.temperature
        if self.tanh_constant is not None:
          logits = self.tanh_constant * tf.tanh(logits)
        index = tf.multinomial(logits, 1)
        index = tf.to_int32(index)
        index = tf.reshape(index, [1])
        arc_seq = arc_seq.write(start_id + i, index)
        curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=index)
        log_prob += curr_log_prob
        curr_ent = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(
                  logits=logits, labels=tf.nn.softmax(logits)))
        entropy += curr_ent
        prev_layers.append(anchors.read(tf.reduce_sum(index)))
        inputs = prev_layers[-1]

      for i in range(2):  # op_1, op_2 (causal conv + attention)
        next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
        prev_c, prev_h = next_c, next_h
        logits = tf.matmul(next_h[-1], self.w_soft_conv) + self.b_soft_conv
        if self.temperature is not None:
          logits /= self.temperature
        if self.tanh_constant is not None:
          op_tanh = self.tanh_constant / self.op_tanh_reduce
          logits = op_tanh * tf.tanh(logits)
        if use_bias:
          logits += self.b_soft_no_learn_conv
        op_id = tf.multinomial(logits, 1)
        op_id = tf.to_int32(op_id)
        op_id = tf.reshape(op_id, [1])
        arc_seq = arc_seq.write(start_id + i + 2, op_id)
        curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=op_id)
        log_prob += curr_log_prob
        curr_ent = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=tf.nn.softmax(logits)))
        entropy += curr_ent
        inputs = tf.nn.embedding_lookup(self.w_emb_conv, op_id)

      for i in range(2): # act_1, act_2 (activation function)
        next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm) 
        prev_c, prev_h = next_c, next_h
        logits = tf.matmul(next_h[-1], self.w_soft_act) + self.b_soft_act
        if self.temperature is not None:
          logits /= self.temperature
        if self.tanh_constant is not None:
          op_tanh = self.tanh_constant / self.op_tanh_reduce
          logits = op_tanh * tf.tanh(logits)
        if use_bias:
          logits += self.b_soft_no_learn_act
        act_id = tf.multinomial(logits, 1)
        act_id = tf.to_int32(act_id)
        act_id = tf.reshape(act_id, [1])
        arc_seq = arc_seq.write(start_id + i + 4, act_id)
        curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=act_id)
        log_prob += curr_log_prob
        curr_ent = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=tf.nn.softmax(logits)))
        entropy += curr_ent
        inputs = tf.nn.embedding_lookup(self.w_emb_act, act_id)

      #combination function
      next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
      prev_c, prev_h = next_c, next_h
      logits = tf.matmul(next_h[-1], self.w_soft_comb) + self.b_soft_comb
      if self.temperature is not None:
        logits /= self.temperature
      if self.tanh_constant is not None:
        op_tanh = self.tanh_constant / self.op_tanh_reduce
        logits = op_tanh * tf.tanh(logits)
      if use_bias:
        logits += self.b_soft_no_learn_comb
      comb_id = tf.multinomial(logits, 1)
      comb_id = tf.to_int32(comb_id)
      comb_id = tf.reshape(comb_id, [1])
      arc_seq = arc_seq.write(start_id + 6, comb_id)
      curr_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=comb_id)
      log_prob += curr_log_prob
      curr_ent = tf.stop_gradient(tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=tf.nn.softmax(logits)))
      entropy += curr_ent
      inputs = tf.nn.embedding_lookup(self.w_emb_comb, comb_id)


      next_h, next_c = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
      prev_c, prev_h = next_c, next_h
      anchors = anchors.write(layer_id, next_h[-1])
      anchors_w_1 = anchors_w_1.write(layer_id, tf.matmul(next_h[-1], self.w_attn_1))
      inputs = self.g_emb

      return (layer_id + 1, inputs, next_c, next_h, anchors, anchors_w_1,
              arc_seq, entropy, log_prob)

    loop_vars = [
      tf.constant(2, dtype=tf.int32, name="layer_id"),
      inputs,
      prev_c,
      prev_h,
      anchors,
      anchors_w_1,
      arc_seq,
      tf.constant([0.0], dtype=tf.float32, name="entropy"),
      tf.constant([0.0], dtype=tf.float32, name="log_prob"),
    ]
    
    loop_outputs = tf.while_loop(_condition, _body, loop_vars, parallel_iterations=1)

    arc_seq = loop_outputs[-3].stack()
    arc_seq = tf.reshape(arc_seq, [-1])
    entropy = tf.reduce_sum(loop_outputs[-2])
    log_prob = tf.reduce_sum(loop_outputs[-1])

    last_c = loop_outputs[-7]
    last_h = loop_outputs[-6]

    return arc_seq, entropy, log_prob, last_c, last_h

  def build_trainer(self, child_model): 
    """
    Need to change this part
    """
    child_model.build_evaluate()
    self.likelihood_valid = (tf.to_float(child_model.loss_gen_validation[0]) / 
                             tf.to_float(child_model.num_gpu * child_model.batch_size))
    #negative log likelihood

    self.reward = 20000 / self.likelihood_valid

    if self.entropy_weight is not None:
      self.reward += self.entropy_weight * self.sample_entropy

    self.sample_log_prob = tf.reduce_sum(self.sample_log_prob)

    self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    baseline_update = tf.assign_sub(
      self.baseline, (1 - self.bl_dec) * (self.baseline - self.reward))

    with tf.control_dependencies([baseline_update]):
      self.reward = tf.identity(self.reward)

    self.loss = self.sample_log_prob * (self.reward - self.baseline)
    self.train_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="train_step")

    tf_variables = [var for var in tf.trainable_variables() if var.name.startswith(self.name)]
    print("-" * 80)
    for var in tf_variables:
      print(var)

    self.train_op, self.lr, self.grad_norm, self.optimizer = get_train_ops(
      self.loss,
      tf_variables,
      self.train_step,
      clip_mode=self.clip_mode,
      grad_bound=self.grad_bound,
      l2_reg=self.l2_reg,
      lr_init=self.lr_init,
      lr_dec_start=self.lr_dec_start,
      lr_dec_every=self.lr_dec_every,
      lr_dec_rate=self.lr_dec_rate,
      optim_algo=self.optim_algo,
      )

    return {
            "train_step": self.train_step,
            "loss": self.loss,
            "train_op": self.train_op,
            "lr": self.lr,
            "grad_norm": self.grad_norm,
            "likelihood_valid": self.likelihood_valid,
            "optimizer": self.optimizer,
            "baseline": self.baseline,
            "entropy": self.sample_entropy,
            "sample_arc": self.arc
            }


