import pandas as pd
import io
def read_field_report(path, data_flag = "*DATA", meta_data_flag = "*METADATA"):
  """
  Reads a field output report.
  """
  text = open(path).read()
  mdpos = text.find(meta_data_flag)
  dpos = text.find(data_flag)
  mdata = io.StringIO( "\n".join(text[mdpos:dpos].split("\n")[1:]))
  data = io.StringIO( "\n".join(text[dpos:].split("\n")[1:]))
  data = pd.read_csv(data, index_col = 0)
  mdata = pd.read_csv(mdata, sep = "=", header = None, index_col = 0)
  #mdata.set_index(["values"])
  mdata.index.name = "keys"
  #mdata = mdata[1]
  mdata = mdata.transpose()
  return mdata, data
  """
  
  pos, count = 0, 0
  for i in xrange(len(lines)):
    if lines[i].startswith("-"): 
      pos    = i
      count += 1 
    if count == 2: break
  if names == None:
    names =  [w for w in lines[pos-2].strip().split("  ") if w != ""]
  data = pd.read_csv(io.StringIO("\n".join(
                         lines[pos + 1:])), 
                     delim_whitespace = True,
                     names = names,
                     index_col = 0)  
  metadata_string = "\n".join(lines[:pos])
  metadata = {}
  metadata['step_time'] = float(metadata_string.split("Step Time = ")[1]
                                .split()[0])
  metadata['step']      = metadata_string.split("Step:")[1].split("\n")[0].strip()
  metadata['source']    =  metadata_string.split("ODB:")[1].split("\n")[0].strip()
  metadata['frame']     = (int(metadata_string.split("Frame: Increment")
                          [1].split(":")[0]))
  return data, metadata
  """                                                
