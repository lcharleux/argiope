import numpy as np
import pandas as pd
import matplotlib as mpl
import StringIO

################################################################################
# CONSTANTS AND DEFINITIONS
################################################################################
ELEMENTS = { "Line2": {"space": 1, 
                       "nvert": 2,
                       "simplex": np.array([[0, 1]])
                      },
             "Tri3": {"space": 2, 
                      "nvert": 3,
                      "faces": np.array([0, 1, 2]),
                      "edges": np.array([[0, 1],
                                         [1, 2],
                                         [2, 0]]),
                      "simplex": np.array([[0, 1, 2]])
                      },
             "Quad4": {"space": 2, 
                       "nvert": 4,
                       "faces": np.array([0, 1, 2, 3]),
                       "edges": np.array([[0, 1],
                                          [1, 2],
                                          [2, 3],
                                          [3, 0]]),
                       "simplex": np.array([[0, 1, 3], 
                                            [1, 2, 3]])
                      },                           
             "Tetra4":{"space": 3, 
                       "nvert": 4,
                       "edges": np.array([[0, 1],
                                          [1, 2],
                                          [2, 3],
                                          [3, 0]]),
                       "faces": np.array([[0, 1, 2],
                                               [0, 3, 1],
                                               [1, 3, 2],
                                               [2, 3, 0]]),                   
                       "simplex": np.array([[0, 1, 3, 4]])
                       },                           
             "Pyra5":{"space": 3, 
                       "nvert": 5,
                       "edges": np.array([[0, 1],
                                          [1, 2],
                                          [2, 3],
                                          [3, 0],
                                          [0, 4],
                                          [1, 4],
                                          [2, 4],
                                          [3, 4]]),
                       "faces": np.array([[0, 1, 2, 3],
                                          [0, 1, 4],
                                          [1, 2, 4],
                                          [2, 3, 4],
                                          [3, 0, 4]]),          
                       "simplex": np.array([[0, 1, 3, 4],
                                            [1, 2, 3, 4]])
                       },     
             "Prism6":{"space": 3, 
                       "nvert": 6,
                       "edges": np.array([[0, 1],
                                          [1, 2],
                                          [2, 0],
                                          [3, 4],
                                          [4, 5],
                                          [5, 3],
                                          [0, 3],
                                          [1, 4],
                                          [2, 5]]),
                       "faces": np.array([[0, 1, 2],
                                          [3, 5, 4],
                                          [0, 3, 4, 1],
                                          [1, 4, 5, 2],
                                          [2, 5, 3, 0]]),   
                       "simplex": np.array([[0, 1, 2, 3],
                                            [1, 2, 3, 4],
                                            [2, 3, 4, 5]]) 
                       },     
             "Hexa8":{"space": 3, 
                      "nvert": 8,
                      "edges": np.array([[0, 1],
                                         [1, 2],
                                         [2, 3],
                                         [3, 0],
                                         [4, 5],
                                         [5, 6],
                                         [6, 7],
                                         [7, 4],
                                         [0, 4],
                                         [1, 5],
                                         [2, 6],
                                         [3, 7]]),
                       "faces": np.array([[0, 1, 2, 3],
                                          [4, 7, 6, 5],
                                          [0, 4, 5, 1],
                                          [1, 5, 6, 2],
                                          [2, 6, 7, 3],
                                          [3, 7, 4, 0]]),
                       "simplex": np.array([[0, 1, 3, 4],
                                            [1, 2, 3, 4],
                                            [3, 2, 7, 4],  
                                            [2, 6, 7, 4],
                                            [1, 5, 2, 4],
                                            [2, 5, 6, 4]])
                       },     
             }

def tri_area(vertices):
  u = vertices[0]
  v = vertices[1]
  w = vertices[2]
  return np.linalg.norm(np.cross( v-u, w-u)) / 2.
    
def tetra_volume(vertices):
  u = vertices[0]
  v = vertices[1]
  w = vertices[2]
  x = vertices[3]
  return np.cross(v-u, w-u).dot(x-u) / 6. 
################################################################################
                  

################################################################################
# MESH CLASSES 
################################################################################
class Container(object):
  
  def __init__(self, master = None, sets = {}):
     self.master = master
     self.sets = {}
     self.fields = {}
     for tag, labels in sets.iteritems(): self.add_set(tag, labels)
   
  def __repr__(self): return self.data.__repr__()
  
  def add_set(self, tag, labels):
    """
    Adds a set.
    """   
    self.sets[tag] = pd.Series(labels) 
    
     
class Nodes(Container):

  def __init__(self, labels, coords, *args, **kwargs):
    self.data = pd.DataFrame(coords, columns = list("xyz"), index = labels)        
    Container.__init__(self, *args, **kwargs)  

  def add_set_by_func(self, tag, func):
    df = self.data
    x, y, z = np.array(df.x), np.array(df.y), np.array(df.z)
    labels = np.array(df.index)
    self.add_set(tag, labels[func(x, y, z, labels)])
     
    
class Elements(Container):

  def __init__(self, labels, etypes, connectivity, surfaces = {}, maxconn = 8, *args, **kwargs):
    maxconn = max( maxconn, max([len(c) for c in connectivity]))
    data = {"etype": etypes}
    for i in range(maxconn): data["n{0}".format(i)] = []
    for c in connectivity:
      lc = len(c)
      for i in range(maxconn):
        if i >= lc: 
          data["n{0}".format(i)] .append(np.nan)
        else:
          data["n{0}".format(i)] .append(c[i])   
    self.data = pd.DataFrame(data, index = labels)
    for tag, data in surfaces.iteritems(): self.add_surface(tag, data)
      
    Container.__init__(self, *args, **kwargs)       
  
  def add_surface(surface, tag, data):
    """
    Adds a surface.
    """   
    self.surfaces[tag] = pd.DataFrame(data, columns = ["element", "face"]) 
  
  def space(self):
    df = self.data
    return pd.Series([ELEMENTS[e]["space"] for e in df.etype], index = df.index)
  
  def nvert(self):
    df = self.data
    return pd.Series([ELEMENTS[e]["nvert"] for e in df.etype], index = df.index)
  
  def to_faces(self):
    pass
  
  def to_polycollection(self):
    pass
      

def ScalarField(object):
  def __init__(self, labels, time, values, master = None):
    self.data = pd.DataFrame({"label":labels, "time": time, "value": values})
    self.master = master
  
"""
def ScalarField(object):
  def __init__(self, labels, time, values, master = None):
    self.data = pd.DataFrame({"label":labels, "time": time, "value": values})
    self.master = master
"""  
  
class Mesh(object):
  def __repr__(self): return "*Nodes:\n{0}\n*Elements\n{1}".format(
                          str(self.nodes), str(self.elements))
  
  def __init__(self, nlabels, coords, elabels, etypes, connectivity, nsets = {}, esets = {}, surfaces = {}, h5path = None):
    self.nodes    = Nodes(    labels = nlabels, coords = coords, sets = nsets, 
                              master = self)
    self.elements = Elements( labels = elabels, connectivity = connectivity, 
                              etypes = etypes, sets = esets, 
                              surfaces = surfaces, master = self)
    self.h5path = h5path
################################################################################
    

################################################################################
# PARSERS
################################################################################
def read_msh(path):
  elementMap = { 1:"Line2",
                 2:"Tri3",
                 3:"Quad4",
                 4:"Tetra4",
                 5:"Hexa8",
                 6:"Prism6",
                 7:"Pyra4",
               }
  lines = np.array(open(path, "rb").readlines())
  locs = {}
  nl = len(lines)
  for i in xrange(nl):
    line = lines[i].lower().strip()
    if line.startswith("$"):
      if line.startswith("$end"):
        locs[env].append(i)
      else:
        env = line[1:]  
        locs[env] = [i]
  nodes = pd.read_csv(
          StringIO.StringIO("\n".join(
                   lines[locs["nodes"][0]+2:locs["nodes"][1]])), 
                   sep = " ",
                   names = ["labels", "x", "y", "z"])
  
  elements = {"labels":[], "etype":[], "conn": [], "tags":[]}
  for line in lines[locs["elements"][0]+2:locs["elements"][1]]:
    d = np.array([int(w) for w in line.split()])
    elements["labels"].append( d[0] )
    elements["etype"].append(elementMap[d[1]] )
    elements["tags"].append( d[3: 3+d[2]] ) 
    elements["conn"].append(d[3+d[2]:])
  physicalNames = {}
  for line in lines[locs["physicalnames"][0]+2:locs["physicalnames"][1]]:
    w = line.split()
    physicalNames[int(w[1])] = w[2].replace('"', '')
  sets = {}
  tags = np.array([t[0] for t in elements["tags"]])
  for k in physicalNames.keys():
    sets[physicalNames[k]] = [t == k for t in tags]     
  elements["sets"] = sets  
   
  return Mesh(nlabels = nodes["labels"], 
              coords = np.array(nodes[["x", "y", "z"]]),
              elabels = elements["labels"],
              connectivity = elements["conn"],
              etypes = elements["etype"],
              esets = elements["sets"])
  
################################################################################
# WRITERS
################################################################################

def write_xdmf(mesh, path, dataformat = "XML"):
  """
  Dumps the mesh to XDMF format.
  """
  cell_map = {
      "Tri3":   4,
      "Quad4":  5,
      "Tetra4": 6,
      "Pyra5":  7,
      "Prism6": 8,
      "Hexa8":  9}
  if dataformat == "HDF":
    hdf = pd.HDFStore(path + ".h5")
  # NODES 
  if dataformat == "XML":
    nodes_string = "\n".join([11*" " + "{0} {1} {2}".format(row.x, row.y, row.z) 
        for index, row in mesh.nodes.data.iterrows()])
  elif dataformat == "HDF":
    hdf.put("COORDS", mesh.nodes.data[list("xyz")])
    nodes_string = 11*" " + "{0}.h5:/COORDS/block0_values".format(path)
  # ELEMENTS
  conn = []
  lconn = 0
  for index, row in mesh.elements.data.iterrows():
    econn  = [row[k] for k in row.keys() if k.startswith("n") 
              and pd.isnull(row[k]) == False]
    econn  = [mesh.nodes.data.index.get_loc(l) for l in econn]
    lconn += len(econn) +1
    conn += [[cell_map[row.etype]] + econn]
  elements_string = "\n".join([11*" " + " ".join([str(n)for n in c]) for c in conn])
  
  pattern = """<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="2.0" xmlns:xi="http://www.w3.org/2001/XInclude">
   <Domain>
     <Grid Name="Mesh">
       <Topology TopologyType="Mixed" NumberOfElements="#ELEMENT_NUMBER">
         <DataItem Format="XML" Dimensions="#CONN_DIMENSION">
#CONN_PATH
         </DataItem>
       </Topology>
       <Geometry GeometryType="XYZ">
         <DataItem Format="#DATAFORMAT" Dimensions="#NODE_NUMBER 3">
#NODE_PATH
         </DataItem>
       </Geometry>  
     </Grid>
   </Domain>
</Xdmf>"""
  pattern = pattern.replace("#ELEMENT_NUMBER", str(len(m2.elements.data)))
  pattern = pattern.replace("#CONN_DIMENSION", str(lconn))
  pattern = pattern.replace("#CONN_PATH", elements_string)
  pattern = pattern.replace("#NODE_NUMBER", str(len(m2.nodes.data)))
  pattern = pattern.replace("#NODE_PATH", nodes_string)
  pattern = pattern.replace("#DATAFORMAT", dataformat)
  open(path + ".xdmf", "wb").write(pattern)
  if dataformat == "HDF":
    print hdf
    hdf.close() 

################################################################################
# TESTS
################################################################################
if __name__ == '__main__':
    m2 = read_msh("../doc/mesh/demo.msh")
    write_xdmf(m2, "test", dataformat = "HDF")

