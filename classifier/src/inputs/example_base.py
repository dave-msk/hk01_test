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
