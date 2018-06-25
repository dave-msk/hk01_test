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

import argparse
import os
from collections import Counter

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--data', type=str, required=True,
                    help='Data file.',
                    metavar="<D>")

parser.add_argument('-s', '--voc_size', type=int, required=True,
                    help='Vocabulary size.',
                    metavar="<S>")

parser.add_argument('-o', '--out_file', type=str, required=True,
                    help='Output file.',
                    metavar="<O>")


def transformed_data_to_tfrecords(file, voc_size, out_file):
  with open(file, 'r') as fin, tf.python_io.TFRecordWriter(out_file) as fout:
    for line in fin:
      l, seq = line.split('\t')
      label = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(l)]))
      counter = Counter([int(t) for t in seq.split(' ')])
      v = np.zeros(voc_size)
      for i, c in counter.most_common(): v[i] = c
      v /= np.linalg.norm(v)
      feature = tf.train.Feature(float_list=tf.train.FloatList(value=v))
      example = tf.train.Example(features=tf.train.Features(feature={
          'label': label,
          'feature': feature}))

      fout.write(example.SerializeToString())


def main(flags):
  out_dir = os.path.dirname(flags.out_file)
  if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
  transformed_data_to_tfrecords(flags.data, flags.voc_size, flags.out_file)


if __name__ == "__main__":
  flags = parser.parse_args()
  main(flags)
