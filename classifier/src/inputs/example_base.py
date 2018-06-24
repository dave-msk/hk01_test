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
import re


class Example(object):
  train_p = re.compile('(?P<id>\\d+),"(?P<tag>(?!").+)","(?P<text>.*)')
  test_p = re.compile('(?P<id>\\d+),"(?P<text>.*)')

  def __init__(self, id, tag=None):
    self.id = int(id)
    self.text = ''
    self.tag = tag

  def append_text(self, text):
    if text.strip().endswith('"'):
      text = text.strip()[:-1]
    self.text += text

  @staticmethod
  def from_line(line, is_training):
    tag = None
    if is_training:
      m = Example.train_p.match(line)
      tag = m.group('tag')
    else:
      m = Example.test_p.match(line)
    id = int(m.group('id'))
    text = m.group('text')
    example = Example(id, tag)
    example.append_text(text)
    return example
