import numpy as np
import pandas as pd
import matplotlib as mpl
import os, subprocess, inspect, StringIO, copy
import argiope
from string import Template

MODPATH = os.path.dirname(inspect.getfile(argiope))

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
   
  def __str__(self): 
    return self.data.__repr__()
  
  def add_set(self, tag, labels):
    """
    Adds a set.
    """   
    self.sets[tag] = set(labels) 
  
  def drop_set(self, tag):
    """
    Drops a set and all associated data.
    """
    dropped_labels = self.sets[tag]
    labels  = set(self.data.index)
    new_labels = labels - dropped_labels
    self.data = self.data.loc[list(new_labels)]
    del self.sets[tag]  
    
     
class Nodes(Container):

  def __init__(self, labels = None, coords = None, *args, **kwargs):
    self.data = pd.DataFrame(coords, columns = list("xyz"), index = labels)        
    Container.__init__(self, *args, **kwargs)  
  
  def __repr__(self):
    return "{0} Nodes ({1} sets)".format(len(self.data), len(self.sets))
  
  def add_set_by_func(self, tag, func):
    df = self.data
    x, y, z = np.array(df.x), np.array(df.y), np.array(df.z)
    labels = np.array(df.index)
    self.add_set(tag, labels[func(x, y, z, labels)])
  
  def save(self):
    hdf = pd.HDFStore(self.master.h5path)
    hdf["nodes/xyz"] = self.data
    for k, s in self.sets.iteritems():
      hdf["nodes/sets/{0}".format(k)] = pd.Series(list(s))
    hdf.close() 
    
class Elements(Container):

  def __init__(self, labels = None, etypes =  None, connectivity = None, surfaces = {}, maxconn = 8, *args, **kwargs):
    if connectivity != None: 
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
    else:
      cols = ["n{0}".format(i) for i in range(maxconn)]            
      self.data = pd.DataFrame(columns = cols)
    self.surfaces = {}
    for tag, data in surfaces.iteritems(): 
      self.add_surface(tag, data)
    Container.__init__(self, *args, **kwargs)       
  
  def __repr__(self):
    return "{0} Elements  ({1} sets, {2} surfaces)".format(len(self.data), len(self.sets), len(self.surfaces))
  
  def add_surface(self, tag, data):
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
  
  def save(self):
    hdf = pd.HDFStore(self.master.h5path)
    hdf["elements/connectivity"] = self.data
    for k, s in self.sets.iteritems():
      hdf["elements/sets/{0}".format(k)] = pd.Series(list(s))
    for k, s in self.surfaces.iteritems():
      hdf["elements/surfaces/{0}".format(k)] = s
    
    hdf.close()
  
  def faces(self):
    """
    Returns the faces of the elements.
    """
    data = self.data
    element_faces = []
    face_labels   = []
    face_number   = []
    element_labels = np.array(data.index)
    conn_keys = self._connectivity_keys()
    element_connectivity = np.array(data[conn_keys])
    element_etype        = np.array(data.etype)

    for i in xrange(len(element_labels)):
      etype = element_etype[i]
      conn  = element_connectivity[i]
      label = element_labels[i]
      if   ELEMENTS[etype]["space"] == 1:
        faces = []
      elif   ELEMENTS[etype]["space"] == 2:
        faces = ELEMENTS[etype]["edges"]
      elif ELEMENTS[etype]["space"] == 3:
        faces = ELEMENTS[etype]["faces"]
      fn = 0
      for face in faces:
        element_faces.append(set(np.int32(conn[face])))
        face_labels.append(label)
        face_number.append(fn)
        fn += 1
    return face_labels, face_number, element_faces
  
     
  def to_polycollection(self):
    pass
  
  def _connectivity_keys(self):
    return ["n{0}".format(i) for i in xrange(self.data.shape[1]-1)]      

class Field(object):
  def __init__(self, info = None, data = None, master = None):
    if info == None: info = {}
    self.info = pd.Series(info)
    self.data = pd.DataFrame(data)
    self.master = master

  def save(self, tag): 
    hdf = pd.HDFStore(self.master.h5path)
    hdf["fields/{0}/data".format(tag)] = self.data
    hdf["fields/{0}/info".format(tag)] = self.info
    hdf.close()
    
class Mesh(object):
  def __repr__(self): return "<Mesh: {0} / {1}>".format(
                          self.nodes.__repr__(), self.elements.__repr__())
  
  def __init__(self, nlabels = None, coords = None, elabels = None, etypes = None, connectivity = None, nsets = {}, esets = {}, surfaces = {}, fields = {}, h5path = None):
    self.nodes    = Nodes(    labels = nlabels, coords = coords, sets = nsets, 
                              master = self)
    self.elements = Elements( labels = elabels, connectivity = connectivity, 
                              etypes = etypes, sets = esets, 
                              surfaces = surfaces, master = self)
    self.fields = {}
    self.h5path = h5path

  def save(self, h5path = None):
    """
    Saves the mesh instance to the hdf store.
    """
    if h5path != None:
      self.h5path = h5path
    self.nodes.save()
    self.elements.save()
    for tag, field in self.fields.iteritems():
      field.save(tag= tag)
  
  def to_inp(self, path = None, element_map = {}):
    return write_inp(self, path, element_map)
  
  def element_set_to_node_set(self, tag):
    """
    Makes a node set from an element set.
    """
    keys = self.elements._connectivity_keys()
    eset = list(self.elements.sets[tag])
    labels = np.array(list(set(self.elements.data.loc[eset][keys].as_matrix().flatten())))
    labels = labels[np.isnan(labels) == False].astype(np.int32)
    labels.sort()
    self.nodes.add_set(tag, labels)
    
  def node_set_to_surface(self, tag):
    """
    Converts a node set to surface.
    """
    elabels, flabels, faces = self.elements.faces()
    nset = self.nodes.sets[tag]
    surf = []
    for i in xrange(len(faces)):
      if nset.issuperset(faces[i]): surf.append((elabels[i], flabels[i]))
    self.elements.add_surface(tag, surf)
    
  def add_field(self, tag, field):
    """
    Add a field to the mesh instance.
    """
    field.master = self
    self.fields[tag] = field  
    
################################################################################
    

################################################################################
# PARSERS
################################################################################
def read_h5(h5path):
  """
  Reads a mesh saved in the HDF5 format.
  """
  hdf = pd.HDFStore(h5path)
  m = Mesh(h5path = h5path)
  m.elements.data = hdf["elements/connectivity"]
  m.nodes.data    = hdf["nodes/xyz"]
  for key in hdf.keys():
    if key.startswith("/nodes/sets"):
      k = key.replace("/nodes/sets/", "")
      m.nodes.sets[k] = set(hdf[key])
    if key.startswith("/elements/sets"):
      k = key.replace("/elements/sets/", "")
      m.elements.sets[k] = set(hdf[key])
    if key.startswith("/elements/surfaces"):
      k = key.replace("/elements/surfaces/", "")
      m.elements.surfaces[k] = hdf[key]
    if key.startswith("/fields/"):
      if key.endswith("/info"):
        tag = key.split("/")[2]
        f = Field()
        f.info = hdf["fields/{0}/info".format(tag)]
        f.data = hdf["fields/{0}/data".format(tag)]
        f.master = m
        m.add_field(tag, f)
  hdf.close()  
  return m
  

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
  
  elements = {"labels":[], "etype":[], "conn": [], "tags":[], "sets":{}}
  for line in lines[locs["elements"][0]+2:locs["elements"][1]]:
    d = np.array([int(w) for w in line.split()])
    elements["labels"].append( d[0] )
    elements["etype"].append(elementMap[d[1]] )
    elements["tags"].append( d[3: 3+d[2]] ) 
    elements["conn"].append(d[3+d[2]:])
  elements["labels"] = np.array(elements["labels"])
  physicalNames = {}
  for line in lines[locs["physicalnames"][0]+2:locs["physicalnames"][1]]:
    w = line.split()
    physicalNames[int(w[1])] = w[2].replace('"', '')
  sets = {}
  tags = np.array([t[0] for t in elements["tags"]])
  for k in physicalNames.keys():
    sets[physicalNames[k]] = np.array([t == k for t in tags])     
  for tag, values in sets.iteritems():
    elements["sets"][tag] = elements["labels"][values]     
   
  return Mesh(nlabels = nodes["labels"], 
              coords = np.array(nodes[["x", "y", "z"]]),
              elabels = elements["labels"],
              connectivity = elements["conn"],
              etypes = elements["etype"],
              esets = elements["sets"])

def read_inp(path):
  """
  Reads Abaqus inp file
  """
  
  def lineInfo(line):
    out =  {"type": "data"}
    if line[0] == "*":
      if line[1] == "*": 
        out["type"] = "comment"
        out["text"] = line[2:]
      else:
        out["type"] = "command"
        words = line[1:].split(",")
        out["value"] = words[0].strip()
        out["options"] = {}
        for word in words[1:]:
          key, value =  [s.strip() for s in word.split("=")]
          out["options"][key] = value
    return out
  
  def elementMapper(inpeltype):
    if inpeltype == "t3d2": return "Line2"
    if inpeltype[:3] in ["cps", "cpe", "cax"]:
      if inpeltype[3] == "3": return "Tri3"
      if inpeltype[3] == "4": return "Quad4"
    if inpeltype[:3] in ["c3d"]:
      if inpeltype[3] == "4": return "Tetra4"
      if inpeltype[3] == "5": return "Pyra5"
      if inpeltype[3] == "6": return "Prism6"
      if inpeltype[3] == "8": return "Hexa8"
    
  nlabels      = []
  coords       = []
  nsets        = {}
  elabels      = []
  etypes       = []
  connectivity = []
  esets        = {}
  surfaces     = {}
  
  # File preprocessing
  lines = np.array([l.strip().lower() for l in open(path).readlines()])
  # Data processing
  env, setlabel = None, None
  for line in lines: 
    d = lineInfo(line)
    if d["type"] == "command": 
      env = d["value"]
      # Nodes
      if env == "node":
        opt = d["options"]
        currentset = None
        if "nset" in opt.keys(): 
          currentset = opt["nset"]
          nsets[currentset] = []
           
      # Elements
      if env == "element":
        opt = d["options"]
        eltype = elementMapper(opt["type"])
        currentset = None
        if "elset" in opt.keys(): 
          currentset = opt["elset"]
          esets[currentset] = []
          
      # Nsets
      if env == "nset":
        opt = d["options"]      
        currentset = opt["nset"]
        nsets[currentset] = []
        
      # Elsets     
      if env == "elset":
        opt = d["options"]      
        currentset = opt["elset"]
        nsets[currentset] = []
      
      # Surfaces
      if env == "surface":
        opt = d["options"]
        currentsurface = opt["name"]
        if opt["type"] == "element":
          surfaces[currentsurface] = []  
                    
             
    if d["type"] == "data": 
      words = line.strip().split(",")
      if env == "node":
        label = int(words[0]) 
        nlabels.append(label) 
        coords.append(
            np.array([np.float64(w) for w in words[1:4]])
                     )
        if currentset != None: nsets[currentset].append(label)
            
              
      if env == "element": 
        label  = int(words[0])
        elabels.append(label)
        connectivity.append(
            np.array( [np.int32(w) for w in words[1:] if len(w) != 0 ])
                           )
        etypes.append(eltype)                   
        if currentset != None: esets[currentset].append(label)
      
      if env == "nset":
        nsets[currentset] = [int(w) for w in words if len(w) != 0]   
        
      if env == "elset":
        esets[currentset] = [int(w) for w in words if len(w) != 0]  
      
      if env == "surface":
        if opt["type"] == "element":
          surfaces[currentsurface].append([w.strip() for w in words])
  
  surfaces2 = {}        
  for tag, surface in surfaces.iteritems():
    surfaces2[tag] = []
    for sdata in surface:
      labels = esets[sdata[0]]
      face = int(sdata[1].split("s")[1].strip())-1
      for label in labels:
        surfaces2[tag].append((label, face))             
  
  return Mesh(nlabels = nlabels,
              coords  = coords,
              nsets   = nsets,
              elabels = elabels,
              etypes  = etypes,
              connectivity = connectivity,
              esets = esets,)
              #surfaces = surfaces2)
  
################################################################################
# WRITERS
################################################################################

def write_xdmf(mesh, path, dataformat = "XML"):
  """
  Dumps the mesh to XDMF format.
  """
  pattern = Template(open(MODPATH + "/templates/mesh/xdmf.xdmf").read())
  attribute_pattern = Template(open(MODPATH + "/templates/mesh/xdmf_attribute.xdmf").read())
  # MAPPINGS
  cell_map = {
      "Tri3":   4,
      "Quad4":  5,
      "Tetra4": 6,
      "Pyra5":  7,
      "Prism6": 8,
      "Hexa8":  9}
  # REFERENCES
  nodes, elements = mesh.nodes.data, mesh.elements.data
  fields = mesh.fields
  # NUMBERS
  Ne, Nn = len(elements), len(nodes)
  # NODES
  nodes_map = np.arange(nodes.index.max()+1)
  nodes_map[nodes.index] = np.arange(len(nodes.index))
  nodes_map[0] = -1
  # ELEMENTS
  cols = ["n{0}".format(i) for i in xrange(elements.shape[1]-1)]
  connectivities  = mesh.elements.data[cols].as_matrix()
  connectivities[np.isnan(connectivities)] = 0
  connectivities = connectivities.astype(np.int32)
  connectivities = nodes_map[connectivities]
  labels          = np.array(elements.index)
  etypes          = np.array([cell_map[t] for t in elements.etype])
  lconn           = Ne + (connectivities != -1).sum()
  # FIELDS
  fields_string = ""
  fstrings_dict = {}
  for tag, field in fields.iteritems():
      field.data.sort_index(inplace = True)
      fshape = field.data.shape[1]
      if   fshape  == 1: ftype = "Scalar"
      elif fshape  == 3: ftype = "Vector"
      elif fshape  == 2: 
        ftype = "Vector"
        # UGLY HACK...
        field = copy.copy(field)
        field.data["v3"] = np.zeros_like(field.data.index)
        fields[tag] = field
        # BACK TO NORMAL  
      elif fshape  == 6: ftype = "Tensor6"
      elif fshape  == 4: 
        ftype = "Tensor6"
        # UGLY HACK...
        field = copy.copy(field)
        field.data["v13"] = np.zeros_like(field.data.index)
        field.data["v23"] = np.zeros_like(field.data.index)
        fields[tag] = field
        # BACK TO NORMAL  
      if field.info.position == "Nodal": 
        position = "Node"
      if field.info.position == "Element":
        position = "Cell"  
      fstrings_dict[tag] = {
           "TAG" : tag,
           "ATTRIBUTETYPE" : ftype,
           "FORMAT" :dataformat,
           "FIELD_DIMENSION" : " ".join([str(l) for l in field.data.shape]),
           "POSITION" :position
                           }
  if dataformat == "XML":
    #NODES
    nodes_string = "\n".join([11*" " + "{0} {1} {2}".format(
                              n.x, 
                              n.y, 
                              n.z) 
                        for i, n in nodes.iterrows()])
    # ELEMENTS
    elements_string = ""
    for i in xrange(Ne):
      elements_string += 11*" " + str(etypes[i]) + " "
      c = connectivities[i]
      c = c[np.where(c != -1)]
      elements_string += " ".join([str(i) for i in c]) + "\n"
    elements_strings = elements_string[:-1]  
    # FIELDS
    for tag, field in fields.iteritems():
      fdata = field.data.to_csv(sep = " ", 
                                index = False, 
                                header = False).split("\n")
      fdata = [11 * " " + l for l in fdata]
      fdata = "\n".join(fdata)
      fstrings_dict[tag]["DATA"] = fdata
      #fields_string += fstrings[tag]     
  elif dataformat == "HDF":
    hdf = pd.HDFStore(path + ".h5")
    hdf.put("COORDS", mesh.nodes.data[list("xyz")])
    flatconn = np.zeros(lconn, dtype = np.int32)
    pos = 0
    for i in xrange(Ne):
      c = connectivities[i]
      c = c[np.where(c != -1)]
      lc = len(c)
      flatconn[pos] = etypes[i]
      flatconn[pos + 1 + np.arange(lc)] = c
      pos += 1 + lc
    hdf.put("CONNECTIVITY", pd.DataFrame(flatconn))  
    nodes_string = 11*" " + "{0}.h5:/COORDS/block0_values".format(path)
    elements_string = 11*" " + "{0}.h5:/CONNECTIVITY/block0_values".format(path)
    for tag, field in fields.iteritems():
      fstrings[tag] = fstrings[tag].replace("#DATA", 
         11*" " + "{0}.h5:/FIELDS/{1}/block0_values".format(path, tag))
      fields_string += fstrings[tag]         
      hdf.put("FIELDS/{0}".format(tag), fields.data)
    hdf.close()
  """
  pattern = pattern.replace("#ELEMENT_NUMBER", str(Ne))
  pattern = pattern.replace("#CONN_DIMENSION", str(lconn))
  pattern = pattern.replace("#CONN_PATH", elements_string)
  pattern = pattern.replace("#NODE_NUMBER", str(Nn))
  pattern = pattern.replace("#NODE_PATH", nodes_string)
  pattern = pattern.replace("#DATAFORMAT", dataformat)
  pattern = pattern.replace("#ATTRIBUTES", fields_string) 
  """
  fields_string = "\n".join([attribute_pattern.substitute(**value) for key, value in fstrings_dict.iteritems()])
  pattern = pattern.substitute(
     ELEMENT_NUMBER = str(Ne),
     CONN_DIMENSION = str(lconn),
     CONN_PATH      = elements_string,
     NODE_NUMBER    = str(Nn),
     NODE_PATH      = nodes_string,
     DATAFORMAT     = dataformat,
     ATTRIBUTES     = fields_string)
  open(path + ".xdmf", "wb").write(pattern)

def write_inp(mesh, path = None, element_map = {}, maxwidth = 80):
  """
  Exports the mesh to the INP format.
  """
  def set_to_inp(sets, keyword):
    ss = ""
    for tag, labels in sets.iteritems():
      labels = list(labels)
      labels.sort()
      if len(labels)!= 0:
        ss += "*{0}, {0}={1}\n".format(keyword, tag)
        line = ""
        counter = 0
        for l in labels:
          counter += 1
          s = "{0},".format(l)
          if (len(s) + len(line) < maxwidth) and counter <16:
            line += s
          else:
            ss += line + "\n"
            line = s
            counter = 0
        ss += line
      ss += "\n"
    return ss.strip()[:-1]            
  
  # SURFACES 
  surf_string = []
  element_sets = copy.copy(mesh.elements.sets)
  for tag, surface in mesh.elements.surfaces.iteritems():
    surf_string.append( "*SURFACE, TYPE=ELEMENT, NAME={0}".format(tag))
    faces = surface.face.unique()
    for face in faces:
      element_sets["_SURF_{0}_FACE{1}".format(tag, face+1)] =  set(
                   surface[surface.face == face].element)
      surf_string.append("  _SURF_{0}_FACE{1}, S{1}".format(tag, face+1)) 
  # ELEMENTS
  conn_keys = mesh.elements._connectivity_keys()
  elements = mesh.elements.data
  etypes = elements.etype.unique()
  el_string = ""
  for etype in etypes:
    new_etype = etype
    if etype in element_map.keys(): 
      new_etype = element_map[etype]
    el_string += "*ELEMENT, TYPE={0}, ELSET={0}_ELEMENTS\n".format(new_etype)
    els = elements[conn_keys][elements.etype == etype]
    els = els.to_csv(header = False, float_format='%.0f').split()
    el_string += ",\n".join([s.strip(",").replace(",", ", ") for s in els])
  # PATTERN
  pattern = Template(open(MODPATH + "/templates/mesh/inp.inp").read())
  pattern = pattern.substitute(
    NODES     = mesh.nodes.data.to_csv(header = False).replace(",", ", ").strip(),
    NODE_SETS = set_to_inp(mesh.nodes.sets, "NSET"),
    ELEMENTS  = el_string,
    ELEMENT_SETS = set_to_inp(element_sets, "ELSET"),
    ELEMENT_SURFACES = "\n".join(surf_string))
  pattern = pattern.strip()
  if path == None:            
    return pattern
  else:
    open(path, "wb").write(pattern)
  

################################################################################
# TESTS
################################################################################
if __name__ == '__main__':
  import time
  print "# READING MESH"
  t0 = time.time()
  m2 = read_msh("../doc/mesh/demo.msh")
  t1 =  time.time()
  print "# => {0:.2f}s".format(t1-t0)
  
  print "# EXPORTING MESH"
  write_xdmf(m2, "test", dataformat = "HDF")
     
  t2 =  time.time()
  print "# => {0:.2f}s".format(t2-t1)  
