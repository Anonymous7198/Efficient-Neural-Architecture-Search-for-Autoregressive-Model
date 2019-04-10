import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages

from common_ops import create_weight
from common_ops import create_bias

from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.ops import variables
import functools


def drop_path(x, keep_prob):
  """Drops out a whole example hiddenstate with the specified probability."""

  batch_size = tf.shape(x)[0]
  noise_shape = [batch_size, 1, 1, 1]
  random_tensor = keep_prob
  random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
  binary_tensor = tf.floor(random_tensor)
  x = tf.div(x, keep_prob) * binary_tensor

  return x

#Layernorm
def layer_norm(inputs, epsilon = 1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.
      
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
    
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta= tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
        outputs = gamma * normalized + beta
        
    return outputs

#Replace none gradients with zero
def replace_none_with_zero(l):
    return [0 if i == None else i for i in l]

#Taken from PixelCNN++
def int_shape(x):
    return list(map(int, x.get_shape()))

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat([x, -x], axis))

#for depthwise separable conv
def get_mask(in_channels, filter_size=[3,3]):
    mask = np.zeros((filter_size[0], filter_size[1], in_channels, in_channels))
    for _ in range(in_channels):
        mask[:, :, _, _] = 1.
    return mask

#for grouped shuffle conv
def channel_shuffle(name, x, num_groups):
    with tf.variable_scope(name):
      n, h, w, c = x.shape.as_list()
      x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
      x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
      output = tf.reshape(x_transposed, [-1, h, w, c])
    return output

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keepdims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keepdims=True))


def discretized_mix_logistic_loss(x,l,sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = int_shape(l) # predicted distribution, e.g. (B,32,32,100)
    nr_mix = int(ls[-1] / 10) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:,:nr_mix]
    l = tf.reshape(l[:,:,:,nr_mix:], xs + [nr_mix*3])
    means = l[:,:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,:,nr_mix:2*nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    m2 = tf.reshape(means[:,:,:,1,:] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0],xs[1],xs[2],1,nr_mix])
    m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs,3) + log_prob_from_logits(logit_probs)
    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs),[1,2])

def sample_from_discretized_mix_logistic(l,nr_mix):
    ls = int_shape(l)
    xs = ls[:-1] + [3]
    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix*3])
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3), depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1,nr_mix])
    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:,:nr_mix]*sel,4)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,:,nr_mix:2*nr_mix]*sel,4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,:,2*nr_mix:3*nr_mix])*sel,4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)
    return tf.concat([tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1]), tf.reshape(x2,xs[:-1]+[1])],3)

def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    #if ema is not None:
     #   v = ema.average(v)
    return v

def get_vars_maybe_avg(var_names, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    vars = []
    for vn in var_names:
        vars.append(get_var_maybe_avg(vn, ema, **kwargs))
    return vars

def adam_updates(params, cost_or_grads, lr=0.001, mom1=0.9, mom2=0.999):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

@add_arg_scope
def dense(x_, num_units, nonlinearity=None, counters={}, ema=None, **kwargs):
    ''' fully connected layer '''
    name = get_name('dense', counters)
    with tf.variable_scope(name):
        W = get_var_maybe_avg('W', ema, shape=[int(x_.get_shape()[1]),num_units], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_units], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        x = tf.matmul(x_, W)
        x += tf.reshape(b, [1, num_units])

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def conv2d(x_, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, counters={}, ema=None, **kwargs):
    ''' convolutional layer '''
    name = get_name('conv2d', counters)
    with tf.variable_scope(name):
        W = get_var_maybe_avg('W', ema, shape=filter_size+[int(x_.get_shape()[-1]),num_filters], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)
        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.conv2d(x_, W, [1] + stride + [1], pad), b)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x

@add_arg_scope
def DiagonalwiseRefactorization(x_, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, counters={}, ema=None, groups=16, **kwargs):
        '''Depthwise Convolution'''
        name = get_name('depthwise_conv', counters)
        with tf.variable_scope(name):
          channels = int(x_.get_shape()[-1]) // groups
          mask = tf.constant(get_mask(channels, filter_size=filter_size).tolist(), dtype=tf.float32,
                             shape=(filter_size[0], filter_size[1], channels, channels))
          splitw = [
                get_var_maybe_avg('W_%d' % _, ema, shape=(filter_size[0], filter_size[1], channels, channels), dtype=tf.float32,
                                    initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
              for _ in range(groups)
          ]

          splitb = [
                get_var_maybe_avg('b_%d' % _, ema, shape=[channels], dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.), trainable=True)
              for _ in range(groups)
          ]

          splitw = [tf.multiply(w, mask) for w in splitw]
          splitx = tf.split(x_, groups, 3)
          splitx = [tf.nn.conv2d(x, w, [1] + stride + [1] , pad)
                    for x, w in zip(splitx, splitw)]

          splitx = [tf.nn.bias_add(x, b)
                    for x, b in zip(splitx, splitb)]
          

          x = tf.concat(splitx, 3)

          if nonlinearity is not None:
            x = nonlinearity(x)

          return x

@add_arg_scope
def grouped_conv2d(x_, num_filters, filter_size=[3,3], stride=[1,1], pad='SAME', nonlinearity=None, counters={}, ema=None, groups=8, **kwargs):
      '''Shuffle Grouped Convolution'''
      name = get_name('group_conv', counters) 
      with tf.variable_scope(name):
        group_size = x_.get_shape()[3].value // groups
        conv_groups = [
            conv2d(x_[:, :, :, i * group_size:i * group_size + group_size], num_filters=num_filters // groups, 
                     filter_size=filter_size, stride=stride, pad=pad, **kwargs) 
            for i in range(groups)]
        conv_g = tf.concat(conv_groups, axis=-1)

      if nonlinearity is not None:
        conv_g = nonlinearity(conv_g)

      conv_g = channel_shuffle('channel_shuffle', conv_g, groups)

      return conv_g

@add_arg_scope
def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
    x = dense(x, num_units, **kwargs)
    return tf.reshape(x, s[:-1]+[num_units])


''' utilities for shifting the image around, efficient alternative to masking convolutions '''

def down_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],1,xs[2],xs[3]]), x[:,:xs[1]-1,:,:]],1)

def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],xs[1],1,xs[3]]), x[:,:,:xs[2]-1,:]],2)

def left_shift(x, step=1):
    xs = int_shape(x)
    return tf.concat([x[:, :, step:, :], tf.zeros([xs[0], xs[1], step, xs[3]]),], 2)

@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)

@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1, 0], [filter_size[1]-1, 0],[0,0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)

@add_arg_scope
def down_shifted_depthwise_conv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    #Depthwise Conv
    x = DiagonalwiseRefactorization(x, filter_size=filter_size, stride=stride, pad='VALID', **kwargs)
    #Pointwise Conv
    x = causal_shift_nin(x, num_filters=num_filters)
    return x

@add_arg_scope
def down_right_shifted_depthwise_conv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1, 0], [filter_size[1]-1, 0],[0,0]])
    #Depthwise Conv
    x = DiagonalwiseRefactorization(x, filter_size=filter_size, stride=stride, pad='VALID', **kwargs)
    #Pointwise Conv
    x = causal_shift_nin(x, num_filters=num_filters)
    return x

@add_arg_scope
def down_shifted_group_conv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    #Shuffle Grouped Conv
    x = grouped_conv2d(x, num_filters=num_filters, filter_size=filter_size, stride=stride, pad='VALID', **kwargs)
    return x

@add_arg_scope
def down_right_shifted_group_conv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1, 0], [filter_size[1]-1, 0],[0,0]])
    #Shuffle Grouped Conv
    x = grouped_conv2d(x, num_filters=num_filters, filter_size=filter_size, stride=stride, pad='VALID', **kwargs)
    return x


def causal_shift_nin(x, num_filters, **kwargs):
    chns = int_shape(x)[-1]
    assert chns % 4 == 0
    left, upleft, up, upright = tf.split(x, 4, axis=-1)
    return nin(
            tf.concat(
                [right_shift(left), right_shift(down_shift(upleft)), down_shift(up), down_shift(left_shift(upleft))],
                axis=-1
                ),
            num_filters,
            **kwargs
            )

def causal_conv_3x3(x, num_filters, stride=[1,1], **kwargs):
    vertical_output   = causal_shift_nin(down_shifted_conv2d(x, num_filters=num_filters, 
                                                              filter_size=[2,3]), num_filters=num_filters)
    horizontal_output = down_right_shifted_conv2d(x, num_filters=num_filters, filter_size=[1,2])
    
    horizontal_output += vertical_output
    return horizontal_output

def causal_conv_5x5(x, num_filters, stride=[1,1], **kwargs):
    vertical_output   = causal_shift_nin(down_shifted_conv2d(x, num_filters=num_filters, 
                                                              filter_size=[3,5]), num_filters=num_filters)
    horizontal_output = down_right_shifted_conv2d(x, num_filters=num_filters, filter_size=[1,3])
    horizontal_output += vertical_output
    return horizontal_output

def causal_depthwise_conv_3x3(x, num_filters, stride=[1,1], **kwargs):
    vertical_output   = causal_shift_nin(down_shifted_depthwise_conv2d(x, num_filters=num_filters, 
                                                              filter_size=[2,3]), num_filters=num_filters)
    horizontal_output = down_right_shifted_depthwise_conv2d(x, num_filters=num_filters, filter_size=[1,2])
    
    horizontal_output += vertical_output
    return horizontal_output

def causal_depthwise_conv_5x5(x, num_filters, stride=[1,1], **kwargs):
    vertical_output   = causal_shift_nin(down_shifted_depthwise_conv2d(x, num_filters=num_filters, 
                                                              filter_size=[3,5]), num_filters=num_filters)
    horizontal_output = down_right_shifted_depthwise_conv2d(x, num_filters=num_filters, filter_size=[1,3])
    horizontal_output += vertical_output
    return horizontal_output

def causal_group_conv_3x3(x, num_filters, stride=[1,1], **kwargs):
    vertical_output   = causal_shift_nin(down_shifted_group_conv2d(x, num_filters=num_filters, 
                                                              filter_size=[2,3]), num_filters=num_filters)
    horizontal_output = down_right_shifted_depthwise_conv2d(x, num_filters=num_filters, filter_size=[1,2])
    horizontal_output += vertical_output
    return horizontal_output

def causal_group_conv_5x5(x, num_filters, stride=[1,1], **kwargs):
    vertical_output   = causal_shift_nin(down_shifted_group_conv2d(x, num_filters=num_filters, 
                                                              filter_size=[3,5]), num_filters=num_filters)
    horizontal_output = down_right_shifted_group_conv2d(x, num_filters=num_filters, filter_size=[1,3])
    horizontal_output += vertical_output
    return horizontal_output

from tensorflow.python.framework import function
@add_arg_scope
def mem_saving_causal_shift_nin(x, num_filters, init, counters, **kwargs):
    if init:
        return causal_shift_nin(x, num_filters, init=init, counters=counters, **kwargs)

    shps = int_shape(x)
    @function.Defun(tf.float32)
    def go(ix):
        tf.get_variable_scope().reuse_variables()
        ix.set_shape(shps)
        return causal_shift_nin(ix, num_filters, init=init, counters=counters, **kwargs)
    temp = go(x)
    temp.set_shape([shps[0], shps[1], shps[2], num_filters])
    return temp

import functools
@functools.lru_cache(maxsize=32)
def get_causal_mask(canvas_size, rate=1):
    causal_mask = np.zeros([canvas_size, canvas_size], dtype=np.float32)
    for i in range(canvas_size):
        causal_mask[i, :i] = 1.
    causal_mask = tf.constant(causal_mask, dtype=tf.float32)

    if rate > 1:
        dim = int(np.sqrt(canvas_size))
        causal_mask = tf.reshape(causal_mask, [canvas_size, dim, dim, 1])
        causal_mask = -tf.nn.max_pool(-causal_mask, [1, rate, rate, 1], [1, rate, rate, 1], 'SAME')

    causal_mask = tf.reshape(causal_mask, [1, canvas_size, -1])
    return causal_mask


def causal_attention(key, mixin, query, downsample=1, use_pos_enc=False):
    bs, nr_chns = int_shape(key)[0], int_shape(key)[-1]


    if downsample > 1:
        pool_shape = [1, downsample, downsample, 1]
        key = tf.nn.max_pool(key, pool_shape, pool_shape, 'SAME')
        mixin = tf.nn.max_pool(mixin, pool_shape, pool_shape, 'SAME')

    xs = int_shape(mixin)
    if use_pos_enc:
        pos1 = tf.range(0., xs[1]) / xs[1]
        pos2 = tf.range(0., xs[2]) / xs[1]
        mixin = tf.concat([
            mixin,
            tf.tile(pos1[None, :, None, None], [xs[0], 1, xs[2], 1]),
            tf.tile(pos2[None, None, :, None], [xs[0], xs[2], 1, 1]),
        ], axis=3)


    mixin_chns = int_shape(mixin)[-1]
    canvas_size = int(np.prod(int_shape(key)[1:-1]))
    canvas_size_q = int(np.prod(int_shape(query)[1:-1]))
    causal_mask = get_causal_mask(canvas_size_q, downsample)

    dot = tf.matmul(
        tf.reshape(query, [bs, canvas_size_q, nr_chns]),
        tf.reshape(key, [bs, canvas_size, nr_chns]),
        transpose_b=True
    ) - (1. - causal_mask) * 1e10
    dot = dot - tf.reduce_max(dot, axis=-1, keep_dims=True)

    causal_exp_dot = tf.exp(dot / np.sqrt(nr_chns).astype(np.float32)) * causal_mask
    causal_probs = causal_exp_dot / (tf.reduce_sum(causal_exp_dot, axis=-1, keep_dims=True) + 1e-6)

    mixed = tf.matmul(
        causal_probs,
        tf.reshape(mixin, [bs, canvas_size, mixin_chns])
    )

    return tf.reshape(mixed, int_shape(query)[:-1] + [mixin_chns])


def non_cached_get_causal_mask(canvas_size, causal_unit):
    assert causal_unit == 1
    ones = tf.ones([canvas_size, canvas_size], dtype=tf.float32)
    lt = tf.matrix_band_part(ones, -1, 0) - tf.matrix_diag(tf.ones([canvas_size,], dtype=tf.float32))
    return lt[None, ...]

def mem_saving_causal_attention(_key, _mixin, _query, causal_unit=1):
    # @function.Defun(tf.float32, tf.float32, tf.float32)
    def go(key, mixin, query,):
        key.set_shape(int_shape(_key))
        mixin.set_shape(int_shape(_mixin))
        query.set_shape(int_shape(_query))
        bs, nr_chns = int_shape(key)[0], int_shape(key)[-1]
        mixin_chns = int_shape(mixin)[-1]
        canvas_size = int(np.prod(int_shape(key)[1:-1]))
        causal_mask = non_cached_get_causal_mask(canvas_size, causal_unit=causal_unit)

        dot = tf.matmul(
                tf.reshape(query, [bs, canvas_size, nr_chns]),
                tf.reshape(key, [bs, canvas_size, nr_chns]),
                transpose_b=True
                ) - (1. - causal_mask) * 1e10
        dot = dot - tf.reduce_max(dot, axis=-1, keep_dims=True)

        causal_exp_dot = tf.exp(dot / np.sqrt(nr_chns).astype(np.float32)) * causal_mask
        causal_probs = causal_exp_dot / (tf.reduce_sum(causal_exp_dot, axis=-1, keep_dims=True) + 1e-6)

        mixed = tf.matmul(
                causal_probs,
                tf.reshape(mixin, [bs, canvas_size, mixin_chns])
                )

        return tf.reshape(mixed, int_shape(mixin))
    temp = go(_key, _mixin, _query)
    temp.set_shape(int_shape(_mixin))
    return temp