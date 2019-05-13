import numpy as np
import pandas as pd

class Mesh:
  _null = "v"
  def __init__(self, nlabels = None, coords = None, nsets = None, 
               elabels = None, etypes = None, stypes = None, conn = None, 
               esets = None, surfaces = None, fields = None):
    self.set_nodes(labels = nlabels, coords = coords, sets = nsets)
    self.set_elements(labels = elabels, types = etypes, stypes = stypes, 
                      conn = conn, sets = esets, surfaces = surfaces)
    self.set_fields(fields)
  
  def __repr__(self):
    return "<Mesh, {0} nodes, {1} elements, {2} fields>".format(
           self.nodes.index.size,
           self.elements.index.size,
           len(self.fields.keys()))
  
  def set_nodes(self, labels, coords, sets):
    """
    Sets the node data.
    """ 
    columns = pd.MultiIndex.from_tuples((("coords", "x"), 
                                         ("coords", "y"), 
                                         ("coords", "z")))
    self.nodes = pd.DataFrame(data = coords, 
                              columns = columns,
                              index = labels)
    for k, v in sets.items(): self.nodes["sets", k] = v
  
  def set_elements(self, labels= None, types = None, stypes = None, conn = None, 
                   sets = None, surfaces = None):
    """
    Sets the element data
    """
    tuples = [("type", "argiope", self._null)] 
    columns = pd.MultiIndex.from_tuples(tuples)
    self.elements = pd.DataFrame(data = types, 
                                 columns = columns,
                                 index = labels)
    
    self.elements.loc[:, ("type", "solver", "v")] = None
    for i in range(len(conn[0])):
      self.elements["conn", "n{0}".format(i), 0] = conn[:, i]
    for k, v in sets.items(): self.elements["sets", k, self._null] = v
    for k, v in surfaces.items():
      for fk, vv in v.items():
         self.elements["surfaces", k, fk] = vv
  
  def set_fields(self, fields):
    """
    Sets the field data
    """
    self.fields = {}
    for k, v in fields.items():
      self.fields[k] = v
  
  
class Field:
  _positions = ["node", "element"]
  def __init__(self, position = "node", step = None, frame = None, time = None,   
               data = None, index = None, columns = None, custom = None):
     # Infos
     if position not in self._positions: 
       raise ValueError("'position' must be in {0}, got '{1}'".format(
                        self._positions, position))
     info = {"position" : position,
             "step" : step,
             "frame": frame,
             "time" : time}
     self.info = pd.Series(info)
     # Data
     if hasattr(self, "_columns"): columns = self._columns
     self.data =  pd.DataFrame(index = index, data = data, columns = columns)
     self.data.index.name = position
     # Custom data
     self.custom = pd.Series(custom)
     

class Tensor6Field(Field):
  _columns = ["v11", "v22", "v33", "v12", "v13", "v23"]
   
class VectorField(Field):
  _columns = ["v1", "v2", "v3"]
  
class ScalarField(Field):
  _columns = ["v"]  

    
# Nodes    
nn = 10
nlabels = np.arange(1, nn+1)
coords = np.random.rand(nn, 3)
nsets = {"top": coords[:,1] > .9,
         "bottom": coords[:,1] < .1 }

# Elements
ne = 5       
elabels = np.arange(1, ne+1) 
etypes = ["Tri3" for i in range(ne)]
conn = np.random.randint(1, nn, size = (ne, 4))
esets = {"odds": (elabels % 2) > 0 }
stop = np.unique(np.random.randint(1, ne+1, size = 3000))
surfaces = {"top" : {"f{0}".format(i): np.random.randint(2, size = ne) == 0 for i in range(1,5)},
         "bottom" : {"f{0}".format(i): np.random.randint(2, size = ne) == 0 for i in range(1,5)}}

# Fields
fields = {"S": Tensor6Field(position = "node",
                            index = elabels, 
                            time = 1.,
                            step = "Loading",
                            frame = 100, 
                            data = np.random.rand(ne, 6))}

mesh = Mesh(nlabels = nlabels, coords = coords, nsets = nsets,
            elabels = elabels, etypes = etypes, conn = conn, esets = esets,
            surfaces = surfaces, fields = fields)
            
 
# Tests:
df1 = pd.DataFrame({"a": [1,2,3]})
df2 = pd.DataFrame({"d":[df1]})    
df2.to_csv("test.csv")
