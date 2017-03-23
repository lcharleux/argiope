import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import collections
import os, subprocess, inspect, io, copy, collections
import argiope
from string import Template

MODPATH = os.path.dirname(inspect.getfile(argiope))

################################################################################
# CONSTANTS AND DEFINITIONS
################################################################################
class Element:
  def __init__(self, nvert, edges = None, simplices = None ):
    self.nvert = nvert
    self.edges = edges
    self.simplices = simplices
    
  
  def __repr__(self):
    return "<{0}D element w. {1} vert.>".format(self.space, self.nvert)
  
  def split(self, elements, to, at = "labels"):
    """
    Splits elements of the rigth type.
    """
    return

class Element1D(Element):
  space = 1
  pass
  
class Element2D(Element):
  space = 2
  
  def __init__(self, angles, optimal_angles, *args, **kwargs):
    self.angles = angles
    self.optimal_angles = optimal_angles
    super().__init__(*args, **kwargs)

class Element3D(Element):
  space = 3
  
  def __init__(self, faces, faces_types, *args, **kwargs):
    self.faces = faces
    self.faces_types = faces_types
    super().__init__(*args, **kwargs)

  def get_angles(self):
    return np.concatenate([self.faces[i][ELEMENTS[self.faces_types[i]].angles] 
                           for i in range(len(self.faces))])
  angles = property(get_angles)  
  
  
   
ELEMENTS = {
    "line2": Element1D(
        nvert =  2),
    "tri3":  Element2D(
        nvert = 3,
        edges = np.array([[0, 1],[1, 2], [2, 0]]),                   
        simplices = np.array([[0, 1, 2]]),
        angles = np.array([[2, 0, 1], [0, 1, 2], [1, 2, 0]]),
        optimal_angles = np.array([60., 60., 60.])),               
    "quad4": Element2D(
        nvert = 4,
        edges = np.array([[0, 1], [1, 2], [2, 3],[3, 0]]),
        simplices = np.array([[0, 1, 3], [1, 2, 3]]),
        angles = np.array([[3, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 0]]),
        optimal_angles = np.array([90., 90., 90., 90.])), 
    "tetra4": Element3D(
        nvert = 4,
        edges = np.array([[0, 1], 
                          [1, 2],
                          [2, 3],
                          [3, 0]]),
        faces = np.array([[0, 1, 2],
                          [0, 3, 1],
                          [1, 3, 2],
                          [2, 3, 0]]),                   
        faces_types = np.array(["tri3",
                                "tri3",
                                "tri3",
                                "tri3"]),
        simplices = np.array([[0, 1, 3, 4]])),
    "pyra5": Element3D(
        nvert = 5,
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                          [0, 4], [1, 4], [2, 4], [3, 4]]),
        faces = [np.array([0, 1, 2, 3]),
                 np.array([0, 1, 4]), 
                 np.array([1, 2, 4]), 
                 np.array([2, 3, 4]), 
                 np.array([3, 0, 4])],          
        faces_types = np.array(["quad4",
                               "tri3", "tri3", "tri3", "tri3"]),
        simplices = np.array([[0, 1, 3, 4],  [1, 2, 3, 4]])), 
    "prism6": Element3D( 
        nvert = 6,
        edges = np.array([[0, 1], [1, 2], [2, 0], [3, 4],
                          [4, 5], [5, 3], [0, 3], [1, 4], [2, 5]]),
        faces = [np.array([0, 1, 2]), 
                 np.array([3, 5, 4]),
                 np.array([0, 3, 4, 1]), 
                 np.array([1, 4, 5, 2]), 
                 np.array([2, 5, 3, 0])],
        faces_types = np.array(["tri3", "tri3",
                               "quad4", "quad4", "quad4"]),                      
        simplices = np.array([[0, 1, 2, 3],
                              [1, 2, 3, 4],
                              [2, 3, 4, 5]])),
    "hexa8":Element3D(  
        nvert = 8,
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]]),
        faces = np.array([[0, 1, 2, 3], 
                          [4, 7, 6, 5],
                          [0, 4, 5, 1],
                          [1, 5, 6, 2],
                          [2, 6, 7, 3],
                          [3, 7, 4, 0]]),
        faces_types = np.array(["quad4", "quad4", "quad4", "quad4", 
                                "quad4", "quad4"]),                      
        simplices =np.array([[0, 1, 3, 4],
                            [1, 2, 3, 4],
                            [3, 2, 7, 4],  
                            [2, 6, 7, 4],
                            [1, 5, 2, 4],
                            [2, 5, 6, 4]])),
                                        
       }    
    
     
################################################################################
                  

################################################################################
# MESH CLASSES 
################################################################################
  
class Mesh:
  """
  A single class to handle meshes.
  """
  _null = "o"
  def __init__(self, nlabels = None, 
                     coords = None, 
                     nsets = None, 
                     elabels = None, 
                     etypes = None, 
                     stypes = None, 
                     conn = None, 
                     esets = None, 
                     surfaces = None, 
                     fields = None,
                     materials = None):
    self.set_nodes(labels = nlabels, 
                   coords = coords, 
                   sets = nsets)
    self.set_elements(labels = elabels, 
                      types = etypes, 
                      stypes = stypes, 
                      conn = conn, 
                      sets = esets, 
                      surfaces = surfaces,
                      materials = materials)
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
    self.nodes.index.name = "node"
    if sets != None:
      for k, v in sets.items(): self.nodes["sets", k] = v
        
  def set_elements(self, labels= None, 
                         types = None, 
                         stypes = None, 
                         conn = None, 
                         sets = None, 
                         surfaces = None, 
                         materials = None):
    """
    Sets the element data
    """
    # COLUMNS BUILDING
    columns = pd.MultiIndex.from_tuples([("type", "argiope", self._null)])
    self.elements = pd.DataFrame(data = types, 
                                 columns = columns,
                                 index = labels)
    self.elements.index.name = "element"
    self.elements.loc[:, ("type", "solver", self._null)] = stypes
    # Connectivity 
    c = pd.DataFrame(conn, index = labels)
    c.fillna(0, inplace = True)
    c[:] = c.values.astype(np.int32)
    c.columns = pd.MultiIndex.from_product([["conn"], 
                                            np.arange(c.shape[1]), 
                                            [self._null]])
    
    self.elements = self.elements.join(c)
    # Sets
    if sets != None:
      for k, v in sets.items(): self.elements["sets", k, self._null] = v
    if surfaces != None:
      for k, v in surfaces.items():
        for fk, vv in v.items():
           self.elements["surfaces", k, fk] = vv
    # Materials
    self.elements["materials"] = materials
    
  def check_elements(self):
    """
    Checks element definitions.
    """
    # ELEMENT TYPE CHECKING
    existing_types = set(self.elements.type.argiope.values.flatten())
    allowed_types = set(ELEMENTS.keys())
    if (existing_types <= allowed_types) == False:
      raise ValueError("Element types {0} not in know elements {1}".format(
                       existing_types - allowed_types, allowed_types))
    print("<Elements: OK>")                   
    
  def set_fields(self, fields = None):
    """
    Sets the field data
    """
    self.fields = {}
    if fields != None:
      for k, v in fields.items():
        self.fields[k] = v
   
  def space(self):
    """
    Returns the dimension of the embedded space of each element.
    """
    return self.elements.type.argiope.applymap(
                               lambda t: ELEMENTS[t]["space"])  
  
  def nvert(self):
    """
    Returns the number of vertices of eache element according to its type/
    """
    return self.elements.type.argiope.applymap(
                               lambda t: ELEMENTS[t]["nvert"])  
  
  def split(self, into = "edges", loc = None, 
            at = "labels", sort_index = True):
    """
    Returns the decomposition of the elements.
    
    Inputs:
    * into: must be in ['edges', 'faces', 'simplices', 'angles']
    * loc: None or labels of the chosen elements.
    * at: must be in ['labels', 'coords']
    """
    if type(loc) == type(None):
      elements = self.elements
    else:  
      elements = self.elements.loc[loc]
    out = []
    for etype, group in elements.groupby([("type", "argiope", "o")]):
      try:
        output_maps = getattr(ELEMENTS[etype], into)
        for om in range(len(output_maps)):
          oshape = len(output_maps[om])
          conn = group.conn
          columns = pd.MultiIndex.from_product([(om,), np.arange(oshape)], 
                                                names = [into, "vertex"])
          data = (conn.values[:, output_maps[om]].reshape(len(conn), oshape))
          df = pd.DataFrame(data = data, 
                            columns = columns,
                            index = conn.index).stack((0,1))
          out.append(df)
      except:
        print("Can not extract '{0}' from '{1}'".format(into, etype))                           
    if len(out) != 0:
      out = pd.concat(out)
      out.sort_index(inplace = True)
      if at == "coords":
        data = self.nodes.loc[out.values].values
        out = pd.DataFrame(index = out.index, data = data, 
                           columns = ["x", "y", "z"])
      return out 

  def centroids_and_volumes(self):
    """
    Returns a dataframe containing volume and centroids of all the elements.
    """
    elements = self.elements
    out = []
    for etype, group in self.elements.groupby([("type", "argiope", "o")]):
      etype_info = ELEMENTS[etype]
      simplices_info = etype_info.simplices
      index = group.index
      simplices_data = self.split(into = "simplices", 
                             loc = index,
                             at = "coords") 
      simplices = simplices_data.values.reshape(
                  index.size, 
                  simplices_info.shape[0], 
                  simplices_info.shape[1], 
                  3) 
      edges = simplices[:,:,1:] - simplices[:,:,:1] 
      simplices_centroids = simplices.mean(axis = 2)
      if etype_info.space == 2:
        simplices_volumes = np.linalg.norm(
                  np.cross(edges[:,:,0], 
                           edges[:,:,1], 
                           axis = 2),
                  axis = 2)
      elif etype_info.space == 3:          
        simplices_volumes =  (np.cross(edges[:,:,0], 
                                       edges[:,:,1], axis = 2) 
                             * edges[:,:, 2]).sum(axis = 2)
      elements_volumes = simplices_volumes.sum(axis = 1)
      elements_centroids = ((simplices_volumes.reshape(*simplices_volumes.shape, 1) 
                          * simplices_centroids).sum(axis = 1) 
                          / elements_volumes.reshape(*elements_volumes.shape,1))
      volumes_df = pd.DataFrame(index = index,
                                data = elements_volumes,
                                columns = pd.MultiIndex.from_product(
                                [["volume"], [self._null]]))
      centroids_df = pd.DataFrame(index = index,
                                data = elements_centroids,
                                columns = pd.MultiIndex.from_product(
                                [["centroid"], ["x", "y", "z"]]))                          
      out.append(pd.concat([volumes_df, centroids_df], axis = 1))             
    out = pd.concat(out)  
    return out
         
  def angles(self):
    """
    Returns the internal angles of all elements and the associated statistics 
    """
    elements = self.elements
    etypes = elements.type.argiope.o.unique()
    out = []
    for etype in etypes:
      etype_info = ELEMENTS[etype]
      angles_info = etype_info.angles
      loc = elements.type.argiope.o == etype
      index = elements.loc[loc].index
      angles_data = self.split(into = "angles", 
                             loc = loc,
                             at = "coords")
      data = angles_data.values.reshape(index.size, 
                                        angles_info.shape[0],
                                        angles_info.shape[1],
                                        3)        
      edges = data[:,:,[0,2],:] - data[:,:,1:2,:]
      edges /= np.linalg.norm(edges, axis = 3).reshape(
               index.size, angles_info.shape[0], 2, 1)
      angles = np.degrees(np.arccos((
               edges[:,:,0] * edges[:,:,1]).sum(axis = 2)))

      deviation = angles - etype_info.optimal_angles
      angles_df = pd.DataFrame(index = index, 
                               data = angles, 
                               columns = pd.MultiIndex.from_product(
                                        [["angles"], range(angles_info.shape[0])]))
      deviation_df = pd.DataFrame(index = index, 
                               data = deviation, 
                               columns = pd.MultiIndex.from_product(
                                        [["deviation"], range(angles_info.shape[0])]))
      df = pd.concat([angles_df, deviation_df], axis = 1)
      df["stats", "max_angle"] = df.angles.max(axis = 1)
      df["stats", "min_angle"] = df.angles.min(axis = 1)
      df["stats", "max_deviation"] = df.deviation.max(axis = 1)
      df["stats", "min_deviation"] = df.deviation.min(axis = 1)
      out.append(df)
    out = pd.concat(out)  
    return out
  
  def stats(self):
    """
    Returns geometric and 
    """
    cv = self.centroids_and_volumes()
    angles  = self.angles()
    return pd.concat([cv , angles[["stats"]] ], axis = 1)
        
  def element_set_to_node_set(self, tag):
    """
    Makes a node set from an element set.
    """
    loc = self.elements.loc[:, ("sets", tag, self._null)].as_matrix().flatten()
    nlabels = np.unique(self.elements.conn.as_matrix()[loc].flatten())
    self.nodes[("sets", tag)] = False
    self.nodes.loc[nlabels, ("sets", tag)] = False 

  def node_set_to_surface(self, tag):
    """
    Converts a node set to surface.
    """
    faces = self.faces()
    surf = pd.DataFrame(np.prod(self.nodes.sets[tag].loc[faces.values.flatten()]
           .values.reshape(faces.shape),axis = 1)
           .astype(np.bool),
           index = faces.index).unstack()
    for k in surf.keys():
      self.elements["surfaces", tag, k[1]] = surf.loc[:, k]
    
  def to_polycollection(self, *args, **kwargs):
    """
    Returns the mesh as matplotlib polygon collection. (tested only for 2D meshes)
    """                          
    nodes, elements = self.nodes, self.elements
    #NODES
    nodes_map = np.arange(nodes.index.max()+1)
    nodes_map[nodes.index] = np.arange(len(nodes.index))
    nodes_map[0] = -1
    coords = nodes.coords.as_matrix()
    #ELEMENTS
    connectivities  = elements.conn.as_matrix()
    connectivities[np.isnan(connectivities)] = 0
    connectivities = connectivities.astype(np.int32)
    connectivities = nodes_map[connectivities]
    labels          = np.array(elements.index)
    etype           = np.array(elements.type.argiope.iloc[:,0])
    #FACES
    verts = []
    for i in range(len(etype)):
      face = connectivities[i][argiope.mesh.ELEMENTS[etype[i]]["faces"]]
      vert = np.array([coords[n] for n in face])
      verts.append(vert[:,:2])
    verts = np.array(verts)
    patches = collections.PolyCollection(verts, *args,**kwargs )
    return patches
  
  def to_triangulation(self):
    """
    Returns the mesh as a matplotlib.tri.Triangulation instance. (2D only)
    """
    nodes, elements = self.nodes, self.elements
    #NODES
    nodes_map = np.arange(nodes.index.max()+1)
    nodes_map[nodes.index] = np.arange(len(nodes.index))
    nodes_map[0] = -1
    coords = nodes.coords.as_matrix()
    #ELEMENTS
    connectivities  = elements.conn.as_matrix()
    connectivities[np.isnan(connectivities)] = 0
    connectivities = connectivities.astype(np.int32)
    connectivities = nodes_map[connectivities]
    labels          = np.array(elements.index)
    etype           = np.array(elements.type.argiope).flatten()
    print(etype)
    #TRIANGLES
    x, y, tri = [], [], []
    for i in range(len(etype)):
      triangles = connectivities[i][argiope.mesh.ELEMENTS[etype[i]]["simplices"]]
      for t in triangles:
        tri.append(t)
    triangulation = mpl.tri.Triangulation(coords[:,0], coords[:,1], tri)
    return triangulation     
  
  
class Field:
  _positions = ["node", "element"]
  def __init__(self, position = "node", step = None, frame = None, time = None,   
               data = None, index = None, custom = None):
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
     self.data =  pd.DataFrame(index = index, data = data, 
                               columns = self._columns)
     self.data.index.name = position
     # Custom data
     self.custom = pd.Series(custom)
     

class Tensor6Field(Field):
  _columns = ["v11", "v22", "v33", "v12", "v13", "v23"]
   
class VectorField(Field):
  _columns = ["v1", "v2", "v3"]
  
class ScalarField(Field):
  _columns = ["v"]  
    
    
''' 
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
    for tag, data in surfaces.items(): 
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
    for k, s in self.sets.items():
      hdf["elements/sets/{0}".format(k)] = pd.Series(list(s))
    for k, s in self.surfaces.items():
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

    for i in range(len(element_labels)):
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
  
     
  def _connectivity_keys(self):
    return ["n{0}".format(i) for i in range(self.data.shape[1]-1)]      

class Field(object):
  """
  A field meta class
  """
  def __init__(self, metadata = None, data = None, master = None):
    if metadata == None: metadata = {}
    self.metadata = pd.Series(metadata)
    self.data = pd.DataFrame(data, columns = self._columns)
    self.master = master

  def save(self, tag): 
    hdf = pd.HDFStore(self.master.h5path)
    hdf["fields/{0}/data".format(tag)] = self.data
    hdf["fields/{0}/metadata".format(tag)] = self.metadata
    hdf.close()

def ScalarField(Field):
  """
  A scalar field class.
  """
  _columns = ["v"]
  

class VectorField(Field):
  """
  A vector field class.
  """
  _columns = ["v1", "v2", "v3"]
  
class Tensor6Field(Field):
  """
  A symmetrictensor field class.
  """
  _columns = ["v11", "v22", "v33", "v12", "v13", "v23"]
  
    

       
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
    for tag, field in self.fields.items():
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
    for i in range(len(faces)):
      if nset.issuperset(faces[i]): surf.append((elabels[i], flabels[i]))
    self.elements.add_surface(tag, surf)
  
  def surface_to_mesh(self, tag):
    """
    Converts a surface to a new mesh.
    """
    elements, nodes = self.elements, self.nodes
    surf = elements.surfaces[tag]
    out = {"nlabels": [], "coords":[], "connectivity":[]}
    for row in surf.iterrows():
      print(row) 
    
    
    
  def add_field(self, tag, field):
    """
    Add a field to the mesh instance.
    """
    field.master = self
    self.fields[tag] = field  
  
  def to_polycollection(self, *args, **kwargs):
    """
    Returns the mesh as matplotlib polygon collection. (tested only for 2D meshes)
    """                          
    nodes, elements = self.nodes.data, self.elements.data
    #NODES
    nodes_map = np.arange(nodes.index.max()+1)
    nodes_map[nodes.index] = np.arange(len(nodes.index))
    nodes_map[0] = -1
    coords = nodes.as_matrix()
    #ELEMENTS
    cols = self.elements._connectivity_keys()
    connectivities  = elements[cols].as_matrix()
    connectivities[np.isnan(connectivities)] = 0
    connectivities = connectivities.astype(np.int32)
    connectivities = nodes_map[connectivities]
    labels          = np.array(elements.index)
    etype           = np.array(elements.etype)
    #FACES
    verts = []
    for i in range(len(etype)):
      face = connectivities[i][argiope.mesh.ELEMENTS[etype[i]]["faces"]]
      vert = np.array([coords[n] for n in face])
      verts.append(vert[:,:2])
    verts = np.array(verts)
    patches = collections.PolyCollection(verts, *args,**kwargs )
    return patches
  
  def centroids_and_volumes(self):
    """
    Returns the centroid and the volume of each element.
    """
    nodes, elements = self.nodes.data, self.elements.data
    #NODES
    nodes_map = np.arange(nodes.index.max()+1)
    nodes_map[nodes.index] = np.arange(len(nodes.index))
    nodes_map[0] = -1
    coords = nodes.as_matrix()
    #ELEMENTS
    cols = self.elements._connectivity_keys()
    connectivities  = elements[cols].as_matrix()
    connectivities[np.isnan(connectivities)] = 0
    connectivities = connectivities.astype(np.int32)
    connectivities = nodes_map[connectivities]
    etype           = np.array(elements.etype)
    #CENTROIDS & VOLUME
    centroids, volumes = [], []
    for i in range(len(etype)):
      simplices = connectivities[i][argiope.mesh.ELEMENTS[etype[i]]["simplex"]]
      simplices = np.array([ [coords[n] for n in simp] for simp in simplices])
      v = np.array([argiope.mesh.tri_area(simp) for simp in simplices])
      g = np.array([simp.mean(axis=0) for simp in simplices])
      vol = v.sum()
      centroids.append((g.transpose()*v).sum(axis=1) / vol)
      volumes.append(vol)
    centroids = np.array(centroids)
    volumes   = np.array(volumes)
    return centroids, volumes
  
  def to_triangulation(self):
    """
    Returns the mesh as a matplotlib.tri.Triangulation instance. (2D only)
    """
    nodes, elements = self.nodes.data, self.elements.data
    #NODES
    nodes_map = np.arange(nodes.index.max()+1)
    nodes_map[nodes.index] = np.arange(len(nodes.index))
    nodes_map[0] = -1
    coords = nodes.as_matrix()
    #ELEMENTS
    cols = self.elements._connectivity_keys()
    connectivities  = elements[cols].as_matrix()
    connectivities[np.isnan(connectivities)] = 0
    connectivities = connectivities.astype(np.int32)
    connectivities = nodes_map[connectivities]
    labels          = np.array(elements.index)
    etype           = np.array(elements.etype)
    #TRIANGLES
    x, y, tri = [], [], []
    for i in range(len(etype)):
      triangles = connectivities[i][argiope.mesh.ELEMENTS[etype[i]]["simplex"]]
      for t in triangles:
        tri.append(t)
    triangulation = mpl.tri.Triangulation(coords[:,0], coords[:,1], tri)
    return triangulation  
 
'''  
  
                          
################################################################################
    

################################################################################
# PARSERS
################################################################################
def read_h5(hdfstore, group = ""):
  """
  Reads a mesh saved in the HDF5 format.
  """
  m = Mesh()
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
      if key.endswith("/metadata"):
        tag = key.split("/")[2]
        f = Field()
        f.metadata = hdf["fields/{0}/metadata".format(tag)]
        f.metadata = hdf["fields/{0}/data".format(tag)]
        f.master = m
        m.add_field(tag, f)
  hdf.close()  
  return m
  

def read_msh(path):
  elementMap = { 1:"line2",
                 2:"tri3",
                 3:"quad4",
                 4:"tetra4",
                 5:"hexa8",
                 6:"prism6",
                 7:"pyra4",
               }
  lines = np.array(open(path, "r").readlines())
  locs = {}
  nl = len(lines)
  for i in range(nl):
    line = lines[i].lower().strip()
    if line.startswith("$"):
      if line.startswith("$end"):
        locs[env].append(i)
      else:
        env = line[1:] 
        locs[env] = [i]
  nodes = pd.read_csv(
          io.StringIO("\n".join(
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
  """
  for tag, values in sets.items():
    elements["sets"][tag] = elements["labels"][values]     
  """
  elements["sets"] = sets
  return Mesh(nlabels = nodes["labels"], 
              coords = np.array(nodes[["x", "y", "z"]]),
              elabels = elements["labels"],
              conn = elements["conn"],
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
      if inpeltype[3] == "3": return "tri3"
      if inpeltype[3] == "4": return "quad4"
    if inpeltype[:3] in ["c3d"]:
      if inpeltype[3] == "4": return "tetra4"
      if inpeltype[3] == "5": return "pyra5"
      if inpeltype[3] == "6": return "prism6"
      if inpeltype[3] == "8": return "hexa8"
    
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
  lines = [line for line in  lines if len(line) != 0]
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
        esets[currentset] = []
      
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
        nsets[currentset] += [int(w) for w in words if len(w) != 0]   
        
      if env == "elset":
        esets[currentset] += [int(w) for w in words if len(w) != 0]  
      
      if env == "surface":
        if opt["type"] == "element":
          surfaces[currentsurface].append([w.strip() for w in words])
  
  surfaces2 = {}        
  for tag, surface in surfaces.items():
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
      "tri3":   4,
      "quad4":  5,
      "tetra4": 6,
      "pyra5":  7,
      "prism6": 8,
      "hexa8":  9}
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
  cols = ["n{0}".format(i) for i in range(elements.shape[1]-1)]
  connectivities  = mesh.elements.data[cols].as_matrix()
  connectivities[np.isnan(connectivities)] = 0
  connectivities = connectivities.astype(np.int32)
  connectivities = nodes_map[connectivities]
  labels          = np.array(elements.index)
  etypes          = np.array([cell_map[t] for t in elements.etype])
  lconn           = Ne + (connectivities != -1).sum()
  # FIELDS
  fields_string = ""
  field_data = {}
  for tag, field in fields.items():
      field_data[tag] = {}
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
      if field.metadata.position == "Nodal": 
        position = "Node"
      if field.metadata.position == "Element":
        position = "Cell"  
      field_data[tag]["TAG"]           = tag
      field_data[tag]["ATTRIBUTETYPE"] = ftype
      field_data[tag]["FORMAT"]        = dataformat
      field_data[tag]["FIELD_DIMENSION"] = " ".join([str(l) for l in field.data.shape])
      field_data[tag]["POSITION"]      = position                             
  if dataformat == "XML":
    #NODES
    nodes_string = "\n".join([11*" " + "{0} {1} {2}".format(
                              n.x, 
                              n.y, 
                              n.z) 
                        for i, n in nodes.iterrows()])
    # ELEMENTS
    elements_string = ""
    for i in range(Ne):
      elements_string += 11*" " + str(etypes[i]) + " "
      c = connectivities[i]
      c = c[np.where(c != -1)]
      elements_string += " ".join([str(i) for i in c]) + "\n"
    elements_strings = elements_string[:-1]  
    # FIELDS
    for tag, field in fields.items():
      fdata = field.data.to_csv(sep = " ", 
                                index = False, 
                                header = False).split("\n")
      fdata = [11 * " " + l for l in fdata]
      fdata = "\n".join(fdata)
      field_data[tag]["DATA"] = fdata
      fields_string += attribute_pattern.substitute(**field_data[tag])     
  elif dataformat == "HDF":
    hdf = pd.HDFStore(path + ".h5")
    hdf.put("COORDS", mesh.nodes.data[list("xyz")])
    flatconn = np.zeros(lconn, dtype = np.int32)
    pos = 0
    for i in range(Ne):
      c = connectivities[i]
      c = c[np.where(c != -1)]
      lc = len(c)
      flatconn[pos] = etypes[i]
      flatconn[pos + 1 + np.arange(lc)] = c
      pos += 1 + lc
    hdf.put("CONNECTIVITY", pd.DataFrame(flatconn))  
    nodes_string = 11*" " + "{0}.h5:/COORDS/block0_values".format(path)
    elements_string = 11*" " + "{0}.h5:/CONNECTIVITY/block0_values".format(path)
    for tag, field in fields.items():
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
  fields_string = "\n".join([attribute_pattern.substitute(**value) for key, value in field_data.items()])
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
    for tag, labels in sets.items():
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
  for tag, surface in mesh.elements.surfaces.items():
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
    el_string += "\n"
  el_string = el_string.strip()  
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
"""
if __name__ == '__main__':
  import time
  print("# READING MESH")
  t0 = time.time()
  m2 = read_msh("../doc/mesh/demo.msh")
  t1 =  time.time()
  print "# => {0:.2f}s".format(t1-t0)
  
  print "# EXPORTING MESH"
  write_xdmf(m2, "test", dataformat = "HDF")
     
  t2 =  time.time()
  

  print "# => {0:.2f}s".format(t2-t1)  
  
"""    
