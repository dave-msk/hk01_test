"""Data set loader."""
import os
from collections import Counter

import numpy as np
import tensorflow as tf

from inputs.example_base import Example


def load_raw_file(file, is_training):
  data = []
  with open(file, 'r') as f:
    example = None
    next(f)
    for line in f:
      if example is None:
        example = Example.from_line(line, is_training)
        data.append(example)
      else:
        example.append_text(line)
      if line.strip().endswith('"'):
        example = None
  return data


def gen_voc_map(data, token_fn, voc_size=None):
  voc = {}
  labels = set()
  for ex in data:
    tokens = token_fn(ex.text)
    if ex.tag not in labels:
      labels.add(ex.tag)
    for t in tokens:
      voc[t] = voc.get(t, 0) + 1

  idx_to_l = sorted(labels)
  l_to_idx = {k: i for i, k in enumerate(idx_to_l)}
  counter = Counter(voc)
  voc = counter.most_common(voc_size)
  idx_to_token = ['<UNK>']
  for t, _ in voc:
    idx_to_token.append(t)
  token_to_idx = {k: i for i, k in enumerate(idx_to_token)}
  return token_to_idx, idx_to_token, l_to_idx, idx_to_l


def transform_data_to_file(token_to_idx, label_to_idx,
                           data, token_fn, out_file):
  with open(out_file, 'w') as t_out:
    for ex in data:
      txt_idxs = [token_to_idx.get(t, 0) for t in token_fn(ex.text)]
      label_idx = label_to_idx.get(ex.tag, -1)
      t_out.write('%s\t%s\n' % (label_idx, ' '.join(str(i) for i in txt_idxs)))


def transform_data_files(train_file, test_file, out_dir,
                         token_fn, voc_size=None):
  train_data = load_raw_file(train_file, is_training=True)
  t_to_idx, idx_to_t, l_to_idx, idx_to_l = (
      gen_voc_map(train_data, token_fn, voc_size))

  out_root = os.path.join(out_dir, 'transformed')
  if not os.path.isdir(out_root):
    os.makedirs(out_root)
  train_out = os.path.join(out_root, 'train.txt')
  test_out = os.path.join(out_root, 'test.txt')
  voc_out = os.path.join(out_root, 'voc.txt')
  label_out = os.path.join(out_root, 'labels.txt')

  transform_data_to_file(t_to_idx, l_to_idx, train_data, token_fn, train_out)

  test_data = load_raw_file(test_file, is_training=False)
  transform_data_to_file(t_to_idx, l_to_idx, test_data, token_fn, test_out)

  with open(voc_out, 'w') as fout:
    [fout.write("%s\t%s\n" % (t, i)) for i, t in enumerate(idx_to_t)]

  with open(label_out, 'w') as fout:
    [fout.write("%s\t%s\n" % (l, i)) for i, l in enumerate(idx_to_l)]

  return train_out, test_out, len(idx_to_t) - 1


def transformed_data_to_tfrecords(file, voc_size, out_file):
  with open(file, 'r') as fin, tf.python_io.TFRecordWriter(out_file) as fout:
    for line in fin:
      l, seq = line.split('\t')
      label = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(l)]))
      counter = Counter([int(t) for t in seq.split(' ')])
      v = np.zeros(voc_size + 1)
      for i, c in counter.most_common(): v[i] = c
      v /= np.linalg.norm(v)
      feature = tf.train.Feature(float_list=tf.train.FloatList(value=v))
      example = tf.train.Example(features=tf.train.Features(feature={
          'label': label,
          'feature': feature}))

      fout.write(example.SerializeToString())
