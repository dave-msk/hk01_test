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
import argparse
import os
import re
from itertools import chain

from inputs.preparation import transform_data_files

parser = argparse.ArgumentParser()

parser.add_argument('-tr', '--train', type=str, required=True,
                    help='Raw training file.',
                    metavar="<TR>")

parser.add_argument('-ts', '--test', type=str, required=True,
                    help='Raw test file.',
                    metavar="<TS>")

parser.add_argument('-v', '--valid', type=str, required=True,
                    help='Raw validation file.',
                    metavar='<V>')

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
  transform_data_files(
      flags.train, flags.test, flags.out_dir, token_fn, flags.voc_size)


if __name__ == "__main__":
  flags = parser.parse_args()
  main(flags)
