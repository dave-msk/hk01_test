from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

from run_loop import run_loop_main

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--model_params', type=str, required=True,
                    help='Config file of model parameters.',
                    metavar="<P>")

parser.add_argument('-r', '--run_params', type=str, required=True,
                    help='Config file of run parameters.',
                    metavar="<R>")

parser.add_argument('-d', '--data_dir', type=str, required=True,
                    help='Data directory.',
                    metavar="<D>")

parser.add_argument('-x', '--export_dir', type=str, required=True,
                    help='Export directory.',
                    metavar="<E>")

parser.add_argument('-m', '--model_dir', type=str, required=True,
                    help='Model directory.',
                    metavar="<M>")


def load_config(file):
  with open(file, 'r') as f:
    return yaml.load(f)


def main(flags):
  model_params = load_config(flags.model_params)
  run_params = load_config(flags.run_params)
  run_loop_main(model_params,
                run_params,
                flags.data_dir,
                flags.model_dir,
                flags.export_dir)


if __name__ == "__main__":
  flags = parser.parse_args()
  main(flags)
