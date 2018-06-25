# Copyright 2018 Siu-Kei Muk (David). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def define_fc_layers(input_tensor, hidden_dims, scope=None):
  net = input_tensor
  with tf.variable_scope(scope):
    for i, dim in enumerate(hidden_dims):
      net = tf.layers.dense(
          net, dim,
          activation=tf.nn.relu,
          kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
          name='fc_%d' % (i+1))
  return net


def model_fn(features, labels, mode, params):
  out_dim = params['num_classes']
  hidden_dims = params['hidden_dims']
  net = define_fc_layers(features, hidden_dims, scope='DNN')

  logits = tf.layers.dense(
      net, out_dim,
      activation=None,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
      name='logits')

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

  xent = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=labels)

  tf.identity(xent, name='cross_entropy')
  tf.summary.scalar('cross_entropy', xent)

  loss = xent
  weight_decay = params.get('weight_decay', 0)
  if weight_decay > 0:
    l2_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(tf.case(v, tf.float32))
         for v in tf.trainable_variables()])
    loss += l2_loss
    tf.summary.scalar('l2_loss', l2_loss)

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    learning_rate = params.get('learning_rate', 0.001)

    optimizer = tf.train.AdamOptimizer(learning_rate)

    minimize_op = optimizer.minimize(loss, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(labels, predictions['classes'])
  metrics = {'accuracy': accuracy}

  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)
