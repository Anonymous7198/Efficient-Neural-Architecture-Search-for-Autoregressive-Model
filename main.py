from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

import utils
from utils import Logger
from utils import DEFINE_boolean
from utils import DEFINE_float
from utils import DEFINE_integer
from utils import DEFINE_string
from utils import print_user_flags
import image_ops as nn

from micro_controller import MicroController
from micro_child import MicroChild

flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
DEFINE_integer("num_epochs", 310, "")

DEFINE_integer("batch_size", 16, "")
DEFINE_integer("child_num_epochs", 5, "")
DEFINE_integer("child_num_layers", 5, "")
DEFINE_integer("child_num_cells", 5, "")
DEFINE_integer("child_out_filters", 256, "")
DEFINE_float("child_lr", 0.1, "")
DEFINE_float("child_lr_dec_rate", 0.1, "")
DEFINE_float("child_keep_prob", 0.5, "")
DEFINE_float("child_drop_path_keep_prob", 1.0, "minimum drop_path_keep_prob")
DEFINE_integer("child_num_gpu", 8, "")
DEFINE_integer("child_key_size", 16, "")
DEFINE_integer("child_value_size", 128, "")
DEFINE_integer("child_num_logistics", 10, "")
DEFINE_float("child_polyak_decay", 0.9995, "")
DEFINE_integer("child_random_seed", 1, "")
DEFINE_integer("child_num_conv", 8, "")
DEFINE_integer("child_num_act", 6, "")
DEFINE_integer("child_num_comb", 3, "")
DEFINE_integer("child_log_interval", 1, "")
DEFINE_integer("child_num_train_batch", 391, "")

DEFINE_float("controller_lr", 1e-3, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", None, "")
DEFINE_float("controller_op_tanh_reduce", 1.0, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_float("controller_entropy_weight", None, "")
DEFINE_integer("controller_train_steps", 50, "")
DEFINE_boolean("controller_use_critic", False, "")

DEFINE_integer("log_every", 50, "How many steps to log (controller)")
DEFINE_integer("save_epochs", 10, "How many epochs to save best architecture")




ControllerClass = MicroController
ChildClass = MicroChild
  
child_model = ChildClass(
  num_layers=FLAGS.child_num_layers,
  num_gpu=FLAGS.child_num_gpu,
  num_cells=FLAGS.child_num_cells,
  out_filters=FLAGS.child_out_filters,
  key_size=FLAGS.child_key_size,
  value_size=FLAGS.child_value_size,
  keep_prob=FLAGS.child_keep_prob,
  drop_path_keep_prob=FLAGS.child_drop_path_keep_prob,
  nr_logistic_mix=FLAGS.child_num_logistics,
  num_epochs=FLAGS.child_num_epochs,
  batch_size=FLAGS.batch_size,
  learning_rate=FLAGS.child_lr,
  lr_decay=FLAGS.child_lr_dec_rate,
  polyak_decay=FLAGS.child_polyak_decay,
  seed=FLAGS.child_random_seed,
  log_interval=FLAGS.child_log_interval,
  data_dir=FLAGS.data_path,
  num_train_batch=FLAGS.child_num_train_batch
)

controller_model = ControllerClass(
  num_branches_conv=FLAGS.child_num_conv,
  num_branches_act=FLAGS.child_num_act,
  num_branches_comb=FLAGS.child_num_comb,
  num_cells=FLAGS.child_num_cells,
  lstm_size=100,
  lstm_num_layers=1,
  lstm_keep_prob=1.0,
  tanh_constant=FLAGS.controller_tanh_constant,
  op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
  temperature=FLAGS.controller_temperature,
  lr_init=FLAGS.controller_lr,
  lr_dec_start=0,
  lr_dec_every=1000000,  # never decrease learning rate
  l2_reg=FLAGS.controller_l2_reg,
  entropy_weight=FLAGS.controller_entropy_weight,
  bl_dec=FLAGS.controller_bl_dec,
  use_critic=FLAGS.controller_use_critic,
  optim_algo="adam",
  )
  

def train(controller_model, child_model):
  print("-" * 80)
  print("Starting session")
  config = tf.ConfigProto(allow_soft_placement=True)
  sess = tf.Session(config=config)
  #build controller model graph
  child_ops = child_model.connect_controller(controller_model)
  #build train child model graph and optimizer  
  bits_per_dim_train, optimizer, increment_op = child_model.build_train_child()
  #build controller graph and optimizer
  controller_ops = controller_model.build_trainer(child_model)
  #init
  initializer = tf.global_variables_initializer()
  sess.run(initializer)
  #save controller paraks
  controller_params = [var for var in tf.trainable_variables() if 'controller' in var.name]
  saver = tf.train.Saver(controller_params)

  #TRAIN
  print("Started to train")
  start_time = time.time()
  #interleaving between training of shared w and controller
  for epoch in range(FLAGS.num_epochs):
    #sample 1 model and trained shared w
    _ = sess.run(child_ops)
    #sample
    lr = child_model.learning_rate
    start_time = time.time()
    for child_epoch in range(child_model.num_epochs):

      bpd_training = []
      for d in child_model.train_data:
        feed_dict = child_model.make_feed_dict(d)
        # forward/backward/update model on each gpu
        lr *= child_model.lr_decay
        feed_dict.update({child_model.tf_lr: lr })
        bpd_train, _, child_step = sess.run([bits_per_dim_train, optimizer, increment_op], feed_dict)
        bpd_training.append(bpd_train)
      bpd_trains = np.mean(bpd_training)

      if child_epoch % child_model.log_interval == 0:
        curr_time = time.time()
        log_string = ""
        log_string += "epoch={:<6d}".format(child_epoch)
        log_string += " bpd_train={:<8.6f}".format(bpd_trains)
        log_string += " mins={:<.2f}".format(
                float(curr_time - start_time) / 60)
        print(log_string)

    sess.run(tf.assign(child_model.global_step, 0))


    for step in range(FLAGS.controller_train_steps):
      #train controller with some steps
      #sample new arc and evaluate
      _ =  sess.run(child_ops)
      
      #train controller with REINFORCE
      run_ops = [
        controller_ops["loss"],
        controller_ops["entropy"],
        controller_ops["lr"],
        controller_ops["grad_norm"],
        controller_ops["likelihood_valid"],
        controller_ops["baseline"],
        controller_ops["train_op"],
      ]

      iter = 0
      for d in child_model.test_data:
        valid_feed_dict = child_model.make_feed_dict(d)
        if iter == 0:
          break

      loss, entropy, lr, gn, likelihood, bl, _ = sess.run(run_ops, valid_feed_dict)
      controller_step = sess.run(controller_ops["train_step"])

      if step % FLAGS.log_every == 0:
        curr_time = time.time()
        log_string = ""
        log_string += "ctrl_step={:<6d}".format(controller_step)
        log_string += " loss={:<7.3f}".format(loss)
        log_string += " ent={:<5.2f}".format(entropy)
        log_string += " lr={:<6.4f}".format(lr)
        log_string += " |g|={:<8.4f}".format(gn)
        log_string += " acc={:<6.4f}".format(likelihood)
        log_string += " bl={:<5.2f}".format(bl)
        log_string += " mins={:<.2f}".format(
              float(curr_time - start_time) / 60)
        print(log_string)

    if epoch % FLAGS.save_epochs == 0:
      bpd_optimal= 100000000000
      best_arc = None

      for _ in range(10):  #search for new architectures
        arc = sess.run(child_ops)

        bpd_test = []

        for d in child_model.test_data:
          evaluate_feed_dict = child_model.make_feed_dict(d)
          bpd_testing = sess.run([child_model.bits_per_dim_test], evaluate_feed_dict)
          bpd_test.append(bpd_testing)
        bits_per_dim_validation = np.mean(bpd_test)

        if bpd_optimal > bits_per_dim_validation:
          bpd_optimal = bits_per_dim_validation
          best_arc = arc

      path = os.path.join(FLAGS.output_dir, str(epoch))
      os.makedirs(path, exist_ok=True)
      saver.save(sess, os.path.join(path, 'params_controller.ckpt'))

      print(np.reshape(best_arc, [-1]))
      print("bit_per_dims={:<6.4f}".format(bpd_optimal))
      print("-" * 80)
    


def main(_):
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  
  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)

  utils.print_user_flags()
  train(controller_model, child_model)


if __name__ == "__main__":
  tf.app.run()


