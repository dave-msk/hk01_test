import abc
import argparse
import os
from datetime import datetime as dt
from datetime import timedelta as td

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--log_file', type=str, required=True,
                    help='Path to log file.',
                    metavar='<F>')

parser.add_argument('-e', '--ext', type=str, required=True,
                    help='Extension to count.')

parser.add_argument('-fd', '--from_date', type=str, default=None,
                    help='Date from which to count data.',
                    metavar='<FD>')

parser.add_argument('-td', '--to_date', type=str, default=None,
                    help='Date to which to count data. (Inclusive)',
                    metavar='<TD>')

parser.add_argument('-v', '--verbose', action="store_true", default=False,
                    help="Display result verbosely.")


class Matcher(abc.ABC):
  @abc.abstractmethod
  def hit(self, obj):
    pass


class DateMatcher(Matcher):
  def __init__(self, from_date, to_date):
    self.from_date = from_date
    self.to_date = to_date + td(days=1)
  
  @staticmethod
  def compile(from_date=None, to_date=None):
    from_date = (dt.min if from_date is None
                 else dt.strptime(from_date, "%Y-%m-%d"))
    
    to_date = (dt.max if to_date is None
               else dt.strptime(to_date, "%Y-%m-%d"))
    return DateMatcher(from_date, to_date)

  def hit(self, obj):
    if not isinstance(obj, (dt, str)):
      return False
    if isinstance(obj, str):
      obj = dt.strptime(obj, "%Y-%m-%d")
    return self.from_date <= obj < self.to_date


class ExtMatcher(Matcher):
  def __init__(self, ext):
    self.ext = ext
  
  def hit(self, obj):
    if not isinstance(obj, str):
      return False
    ext = os.path.splitext(obj)[1][1:]
    return ext == self.ext


def extract_fields(rec):
  raw_fields = rec.split("\t")
  datetime = dt.strptime("%s %s" % (raw_fields[0], raw_fields[1]),
                         "%Y-%m-%d %H:%M:%S")
  size = int(raw_fields[2])
  url = raw_fields[3]
  return (datetime, size, url)


def count(filename, size_fn):
  total = 0
  with open(filename, 'r') as f:
    for line in f:
      total += size_fn(line)
  return total


def main(flags):
  ext_mtr = ExtMatcher(flags.ext)
  date_mtr = DateMatcher.compile(flags.from_date, flags.to_date)
  def size_fn(rec):
    rec = rec.strip()
    if rec.startswith("#"):
      return 0
    datetime, size, url = extract_fields(rec)
    hit = ext_mtr.hit(url) and date_mtr.hit(datetime)
    return size if hit else 0
  
  total_size = count(flags.log_file, size_fn)
  
  if flags.verbose:
    print("Total data transfer with extension {} in \"{} - {}\" is: {}"
          .format(flags.ext,
                  str(date_mtr.from_date.date()),
                  str(date_mtr.to_date.date()),
                  total_size))
  else:
    print(total_size)


if __name__ == "__main__":
  flags = parser.parse_args()
  main(flags)
