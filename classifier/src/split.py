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
import random

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--filename', type=str, required=True,
                    help='Text file to be split by lines.',
                    metavar='<F>')

parser.add_argument('-fo', '--first_out', type=str, required=True,
                    help='First output file.',
                    metavar='<FO>')

parser.add_argument('-so', '--second_out', type=str, required=True,
                    help='Second output file.',
                    metavar='<SO>')

parser.add_argument('-r', '--ratio', type=float, required=True,
                    help='Ratio of lines that goes to the first output file.')


def split_by_lines(in_file, out_1, out_2, prob):
  with open(in_file, 'r') as fin, \
       open(out_1, 'w') as fout_1,\
       open(out_2, 'w') as fout_2:
    for line in fin:
      if random.random() < prob:
        fout_1.write(line)
      else:
        fout_2.write(line)


def main(flags):
  prob = flags.ratio
  if not (0 < prob < 1):
    raise ValueError("Ratio must be between 0 and 1. Given: {}".format(prob))
  split_by_lines(flags.filename, flags.first_out, flags.second_out, prob)


if __name__ == "__main__":
  flags = parser.parse_args()
  main(flags)
