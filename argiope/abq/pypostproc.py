import pandas as pd
import io, argiope

def read_history_report(path, steps, x_name = "t"):
  """
  Reads an history output report.
  """
  data = pd.read_csv(path, delim_whitespace = True)
  for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors = "coerce").fillna(0.)
  
  if x_name != None:
    data[x_name] = data.X
    del data["X"]
    
  data["step"] = 0
  t = 0.
  for i in range(len(steps)):
    dt = steps[i].duration
    loc = data[data[x_name] == t].index
    if len(loc) == 2:
      data.loc[loc[1]:, "step"] = i
    t += dt 
  return data
  
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
  data = data.groupby(data.index).mean()
  mdata = pd.read_csv(mdata, sep = "=", header = None, index_col = 0)[1]
  mdata = mdata.to_dict()
  out = {}
  out["step_num"] = int(mdata["step_num"])
  out["step_label"] = mdata["step_label"]
  out["frame"] = int(mdata["frame"])
  out["frame_value"] = float(mdata["frame_value"])
  out["part"] = mdata["instance"]
  position_map = {"NODAL": "node", 
                  "ELEMENT_CENTROID": "element", 
                  "WHOLE_ELEMENT": "element"}
  out["position"] = position_map[mdata["position"]]
  out["label"] = mdata["label"]  
  out["data"] = data
  field_class = getattr(argiope.mesh, mdata["argiope_class"])
  return field_class(**out)
                                   
