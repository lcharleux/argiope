import pandas as pd
import StringIO

def read_field_report(path, names = None):
  """
  Reads a field output report.
  """
  lines = open(path, "rb").readlines()
  pos, count = 0, 0
  for i in xrange(len(lines)):
    if lines[i].startswith("-"): 
      pos    = i
      count += 1 
    if count == 2: break
  return pd.read_csv(StringIO.StringIO("\n".join(
                         lines[pos + 1:])), 
                     delim_whitespace = True,
                     names = ["label"] + names,
                     index_col = 0)  
