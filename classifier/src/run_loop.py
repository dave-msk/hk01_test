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
