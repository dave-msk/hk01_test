from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models import model_fn
from inputs import get_input_fn
from export import build_tensor_serving_input_receiver_fn


def run_loop_main(model_params, run_params, data_dir, model_dir, export_dir):
  batch_size = run_params.get('batch_size', 1)
  epochs = run_params.get('epochs', 1)
  voc_size = model_params['voc_size']
  input_fn_train = get_input_fn(True, data_dir, voc_size, batch_size)
  input_fn_valid = get_input_fn(False, data_dir, voc_size, batch_size)

  estimator = tf.estimator.Estimator(model_fn=model_fn,
                                     model_dir=model_dir,
                                     params=model_params)

  for i in range(epochs):
    estimator.train(input_fn=input_fn_train)
    estimator.evaluate(input_fn=input_fn_valid)

  input_receiver_fn = build_tensor_serving_input_receiver_fn(
      [voc_size], batch_size=batch_size)
  estimator.export_savedmodel(export_dir, input_receiver_fn)
