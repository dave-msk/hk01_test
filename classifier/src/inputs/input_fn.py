# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""
This file is inspired from preprocessing code from the repository
"tensorflow/models/official/resnet/".

The copyright and license block is preserved to fulfill license requirement.
Credits go to The TensorFlow Authors.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from functools import partial

import tensorflow as tf


# Taken from "tensorflow/models/official/resnet/imagenet_main.py", and modified.
def get_filenames(is_training, data_dir):
  filename = 'train' if is_training else 'valid'
  return [os.path.join(data_dir, '%s.tfrecords' % filename)]


# Taken from "tensorflow/models/official/resnet/imagenet_main.py", and modified.
def _parse_example_proto(example_serialized, voc_size):
  feature_map = {
      'label': tf.FixedLenFeature([1], dtype=tf.int64,
                                  default_value=-1),
      'feature': tf.FixedLenFeature([voc_size+1], dtype=tf.float32,
                                    default_value=0.0),
  }

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['label'], dtype=tf.int32)

  return features['feature'], label


# Taken from "tensorflow/models/official/resnet/imagenet_main.py", and modified.
def parse_record(raw_record, voc_size):
  feature, label = _parse_example_proto(raw_record, voc_size)

  # Add extra preprocessing logic here
  return feature, label


# Taken from "tensorflow/models/official/resnet/resnet_run_loop.py",
# and modified.
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1):
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  dataset = dataset.repeat(num_epochs)

  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda value: parse_record_fn(value),
          batch_size=batch_size,
          num_parallel_batches=1))

  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  return dataset


# Taken from "tensorflow/models/official/resnet/imagenet_main.py", and modified.
def get_input_fn(is_training, data_dir, voc_size, batch_size,
                 shuffle_buffer=2000, num_epochs=1):
  def input_fn():
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)

    if is_training:
      dataset = dataset.shuffle(buffer_size=len(filenames))

    dataset = dataset.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=10))

    parse_record_fn = partial(parse_record, voc_size=voc_size)

    return process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer,
        parse_record_fn=parse_record_fn,
        num_epochs=num_epochs)
  return input_fn
