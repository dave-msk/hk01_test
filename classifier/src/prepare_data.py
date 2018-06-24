import argparse
import os
import re
from itertools import chain

from inputs.preparation \
    import transform_data_files, transformed_data_to_tfrecords

parser = argparse.ArgumentParser()

parser.add_argument('-tr', '--train', type=str, required=True,
                    help='Raw training file.',
                    metavar="<TR>")

parser.add_argument('-ts', '--test', type=str, required=True,
                    help='Raw test file.',
                    metavar="<TS>")

parser.add_argument('-d', '--out_dir', type=str, required=True,
                    help='Output directory for prepared files.',
                    metavar='<D>')

parser.add_argument('-vs', '--voc_size', type=int, default=None,
                    help='Vocabulary size.',
                    metavar='<VS>')

p = re.compile(u'[\u4e00-\u9fff]+')


def token_fn(text):
  global p
  return list(chain.from_iterable(p.findall(text)))


def main(flags):
  if not os.path.isdir(flags.out_dir):
    os.makedirs(flags.out_dir)
  tfm_train, tfm_test, voc_size = transform_data_files(
      flags.train, flags.test, flags.out_dir, token_fn, flags.voc_size)
  tf_out_root = os.path.join(flags.out_dir, 'tf_records')
  if not os.path.isdir(tf_out_root):
    os.makedirs(tf_out_root)
  tf_train_out = os.path.join(tf_out_root, 'train.tfrecords')
  tf_test_out = os.path.join(tf_out_root, 'test.tfrecords')
  transformed_data_to_tfrecords(tfm_train, voc_size, tf_train_out)
  transformed_data_to_tfrecords(tfm_test, voc_size, tf_test_out)


if __name__ == "__main__":
  flags = parser.parse_args()
  main(flags)
