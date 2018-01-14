import numpy as np
import pandas as pd
import matplotlib as mpl
import os, subprocess, inspect, io, copy, collections, warnings
import argiope
from string import Template

MODPATH = os.path.dirname(inspect.getfile(argiope))

################################################################################
# CONSTANTS AND DEFINITIONS
################################################################################
class Element:
  """
  The Element meta class.
  """
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

class Element0D(Element):
  space = 0
  pass

class Element1D(Element):
  space = 1
  pass
  
class Element2D(Element):
  space = 2
  
  def __init__(self, angles, optimal_angles, *args, **kwargs):
    self.angles = angles
    self.optimal_angles = optimal_angles
    super().__init__(*args, **kwargs)
  
  def get_surfaces(self):
    return self.edges
  surfaces = property(get_surfaces)  
    
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
  
  def get_optimal_angles(self):
    return  np.concatenate([ELEMENTS[e].optimal_angles 
            for e in self.faces_types])
  optimal_angles = property(get_optimal_angles)  
  
  def get_surfaces(self):
    return self.faces
  surfaces = property(get_surfaces)
  
   
ELEMENTS = {
    "point1": Element0D(nvert = 1),
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
        simplices = np.array([[0, 1, 2, 3]])),
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
  
class Mesh(argiope.utils.Container):
  """
  A single class to handle meshes and associated data. The class constructor only a redirection to the following methods:
    
  * :func:`Mesh.set_nodes`
  * :func:`Mesh.set_elements`
  * :func:`Mesh.set_fields`
    
  A simple example is given below.
    
  .. literalinclude:: examples/mesh/Mesh.py
  
  For more complex examples, follow the notebook tutorials.
  """
  def __init__(self,**kwargs):
    self.set_nodes(**kwargs)
    self.set_elements(**kwargs)
    self.set_fields(**kwargs)
    
  
  def __repr__(self):
    Ne, Nn = 0, 0
    if self.elements is not None : Ne = self.elements.index.size
    if self.nodes is not None: Nn = self.nodes.index.size
    return "<Mesh, {0} nodes, {1} elements, {2} fields>".format(
           Nn, Ne, len(self.fields))
  
  def set_nodes(self, nlabels = [], coords = [], nsets = {}, **kwargs):
    r"""
    Sets the node data.
    
    :arg nlabels: node labels. Items be strictly positive and int typed 
                  in 1D array-like with shape :math:`(N_n)`.
    :type nlabels: 1D uint typed array-like
    :arg coords: node coordinates. Must be float typed 2D array-like of shape 
                 :math:`(N_n \times 3)`.  
    :type coords: 2D float typed array-like            
    :arg nsets: node sets. Contains boolean array-like of shape :math:`(N_n)`.
    :type nsets: dict
    """ 
    # DATA PREPROCESSING
    nlabels = np.array(nlabels).astype(np.int64)
    coords = np.array(coords).astype(np.float64)
    if (nlabels < 0).sum() > 0: 
      raise ValueError("Node labels must be strictly positive.")
    if len(nlabels) != len(coords):
      raise ValueError("'nlabels' and 'coords' must have the same length")
    if coords.shape[1] != 3:
      raise ValueError("coordinates must be 3 dimensional.")    
    # ATTRIBUTES CREATION
    columns = pd.MultiIndex.from_tuples((("coords", "x"), 
                                         ("coords", "y"), 
                                         ("coords", "z")))
    self.nodes = pd.DataFrame(data = coords, 
                              columns = columns,
                              index = nlabels)
    self.nodes.index.name = "node"
    for k, v in nsets.items(): 
      v = np.array(v)
      if v.dtype != 'bool':
        raise ValueError("Sets must be boolean array-likes.")   
      self.nodes["sets", k] = v
        
  def set_elements(self, elabels = None, 
                         types = None, 
                         stypes = "", 
                         conn = None, 
                         esets = {}, 
                         surfaces = {}, 
                         materials = "",
                         **kwargs):
    """
    Sets the element data.
    
    :arg elabels: element labels. Items be strictly positive and int typed 
                  in 1D array-like with shape :math:`(N_e)`.
    :type elabels: 1D uint typed array-like
    :arg types: element types chosen among argiope specific element types. 
    :type types: str typed array-like 
    :arg stypes: element types chosen in solver (depends on the chosen solver) specific element types. 
    :type stypes: str typed array-like 
    :arg conn: connectivity table. In order to deal with non rectangular tables, :math:`0` can be used to fill missing data. 
    :type conn: uint typed array-like
    :arg esets: element sets. Contains boolean array-like of shape :math:`(N_e)`.
    :type esets: dict  
    :arg surfaces: surfaces. Contains boolean array-like of shape :math:`(N_e, N_s )` with :math:`N_s` being the maximum number of faces on a single element. 
    :type surfaces: dict   
    :arg materials: material keys. Any number a of materials can be used.
    :type materials: str typed array-like 
     
      
      
           
    """
    # COLUMNS BUILDING
    if elabels is None:
       warnings.warn(
       "Since no element labels where provided, no elements where created", 
       Warning)
       self.elements = None
    else:   
      columns = pd.MultiIndex.from_tuples([("type", "argiope", "")])
      self.elements = pd.DataFrame(data = types,
                                   columns = columns,
                                   index = elabels)
      self.elements.index.name = "element"
      self.elements.loc[:, ("type", "solver", "")] = stypes
      # Connectivity 
      c = pd.DataFrame(conn, index = elabels)
      c.fillna(0, inplace = True)
      c[:] = c.values.astype(np.int32)
      c.columns = pd.MultiIndex.from_product([["conn"], 
                                              ["n{0}".format(n) for 
                                               n in np.arange(c.shape[1])], 
                                              [""]])
      self.elements = self.elements.join(c)
      # Sets
      for k, v in esets.items(): self.elements[("sets", k, "")] = v
      for k, v in surfaces.items():
        for fk, vv in v.items():
          self.elements[("surfaces", k, "s{0}".format(fk))] = vv
      # Materials
      self.elements[("materials", "", "") ] = materials
      self.elements.sort_index(axis = 1, inplace = True)
       
  def set_fields(self, fields = None, **kwargs):
    """
    Sets the fields.
    """
    self.fields = []
    if fields != None:
      for field in fields: 
        self.fields.append(field)
      
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
    
  
  def space(self):
    """
    Returns the dimension of the embedded space of each element.
    """
    return self.elements.type.argiope.map(
           lambda t: ELEMENTS[t].space)  
  
  def nvert(self):
    """
    Returns the number of vertices of eache element according to its type/
    """
    return self.elements.type.argiope.map(
           lambda t: ELEMENTS[t].nvert)   
  
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
    for etype, group in elements.groupby([("type", "argiope", "")]):
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
        data = self.nodes.coords.loc[out.values].values
        out = pd.DataFrame(index = out.index, data = data, 
                           columns = ["x", "y", "z"])
      return out 

  def centroids_and_volumes(self, sort_index = True):
    """
    Returns a dataframe containing volume and centroids of all the elements.
    """
    elements = self.elements
    out = []
    for etype, group in self.elements.groupby([("type", "argiope", "")]):
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
                  axis = 2)/2.
      elif etype_info.space == 3:          
        simplices_volumes =  (np.cross(edges[:,:,0], 
                                       edges[:,:,1], axis = 2) 
                             * edges[:,:, 2]).sum(axis = 2) / 6.
      elements_volumes = simplices_volumes.sum(axis = 1)
      elements_centroids = ((simplices_volumes.reshape(*simplices_volumes.shape, 1) 
                          * simplices_centroids).sum(axis = 1) 
                          / elements_volumes.reshape(*elements_volumes.shape,1))
      volumes_df = pd.DataFrame(index = index,
                                data = elements_volumes,
                                columns = pd.MultiIndex.from_product(
                                [["volume"], [""]]))
      centroids_df = pd.DataFrame(index = index,
                                data = elements_centroids,
                                columns = pd.MultiIndex.from_product(
                                [["centroid"], ["x", "y", "z"]]))                          
      out.append(pd.concat([volumes_df, centroids_df], axis = 1))             
    out = pd.concat(out)  
    if sort_index: out.sort_index(inplace = True)
    return out.sort_index(axis= 1)
         
  def angles(self, zfill = 3):
    """
    Returns the internal angles of all elements and the associated statistics 
    """
    elements = self.elements.sort_index(axis = 1)
    etypes = elements[("type", "argiope")].unique()
    out = []
    for etype in etypes:
      etype_info = ELEMENTS[etype]
      angles_info = etype_info.angles
      loc = elements[("type", "argiope", "")] == etype
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
      [["angles"], ["a" + "{0}".format(s).zfill(zfill) 
              for s in range(angles_info.shape[0])]]))
      deviation_df = pd.DataFrame(index = index, 
                               data = deviation, 
                               columns = pd.MultiIndex.from_product(
      [["deviation"], ["d" + "{0}".format(s).zfill(zfill) 
              for s in range(angles_info.shape[0])]]))
      
      df = pd.concat([angles_df, deviation_df], axis = 1).sort_index(axis = 1)
      df["stats", "max_angle"] = df.angles.max(axis = 1)
      df["stats", "min_angle"] = df.angles.min(axis = 1)
      df["stats", "max_angular_deviation"] = df.deviation.max(axis = 1)
      df["stats", "min_angular_deviation"] = df.deviation.min(axis = 1)
      df["stats", "max_abs_angular_deviation"] = abs(df.deviation).max(axis = 1)    
      df = df.sort_index(axis = 1)  
      out.append(df)
      
    out = pd.concat(out).sort_index(axis = 1)
    return out
  
  def edges(self, zfill = 3):
    """
    Returns the aspect ratio of all elements.
    """
    edges = self.split("edges", at = "coords").unstack()
    edges["lx"] = edges.x[1]-edges.x[0]
    edges["ly"] = edges.y[1]-edges.y[0]
    edges["lz"] = edges.z[1]-edges.z[0]
    edges["l"] = np.linalg.norm(edges[["lx", "ly", "lz"]], axis = 1)
    edges = (edges.l).unstack()
    edges.columns = pd.MultiIndex.from_product([["length"], 
                    ["e" + "{0}".format(s).zfill(zfill) 
                    for s in np.arange(edges.shape[1])]])
    edges[("stats", "lmax")] = edges.length.max(axis = 1)
    edges[("stats", "lmin")] = edges.length.min(axis = 1)
    edges[("stats", "aspect_ratio")] = edges.stats.lmax / edges.stats.lmin
    return edges.sort_index(axis = 1)
  
  def stats(self):
    """
    Returns mesh quality and geometric stats.
    """
    cv = self.centroids_and_volumes()
    angles  = self.angles()
    edges = self.edges()
    return pd.concat([cv , angles[["stats"]], edges[["stats"]] ], 
                     axis = 1).sort_index(axis = 1)
        
  def element_set_to_node_set(self, tag):
    """
    Makes a node set from an element set.
    """
    nodes, elements = self.nodes, self.elements
    loc = (elements.conn[elements[("sets", tag, "")]]
           .stack().stack().unique())
    loc = loc[loc != 0]
    nodes[("sets", tag)] = False
    nodes.loc[loc, ("sets", tag) ] = True

  def node_set_to_surface(self, tag):
    """
    Converts a node set to surface.
    """
    # Create a dummy node with label 0
    nodes = self.nodes.copy()
    dummy = nodes.iloc[0].copy()
    dummy["coords"] *= np.nan
    dummy["sets"] = True
    nodes.loc[0] = dummy
    # Getting element surfaces
    element_surfaces= self.split("surfaces").unstack()
    # killer hack !
    surf = pd.DataFrame(
             nodes.sets[tag].loc[element_surfaces.values.flatten()]
                   .values.reshape(element_surfaces.shape)
                   .prod(axis = 1)
                   .astype(np.bool),
             index = element_surfaces.index).unstack().fillna(False)
    for k in surf.keys():
      self.elements["surfaces", tag, "f{0}".format(k[1]+1) ] = surf.loc[:, k]
    
    
  def surface_to_element_sets(self, tag):
    """
    Creates elements sets corresponding to a surface.
    """
    surface = self.elements.surfaces[tag]
    for findex in surface.keys():
      if surface[findex].sum() != 0:
        self.elements[("sets", "_SURF_{0}_FACE{1}"
                     .format(tag, findex[1:]), "")] = surface[findex]
    
  def to_polycollection(self, *args, **kwargs):
    """
    Returns the mesh as matplotlib polygon collection. (tested only for 2D meshes)
    """                          
    from matplotlib import collections
    nodes, elements = self.nodes, self.elements.reset_index()
    verts = []
    index = []
    for etype, group in elements.groupby([("type", "argiope", "")]):
      index += list(group.index)
      nvert = ELEMENTS[etype].nvert
      conn = group.conn.values[:, :nvert].flatten()
      coords = nodes.coords[["x", "y"]].loc[conn].values.reshape(
                                            len(group), nvert, 2)
      verts += list(coords)
    verts = np.array(verts)
    verts= verts[np.argsort(index)]
    return collections.PolyCollection(verts, *args,**kwargs )
    
  def to_triangulation(self):
    """
    Returns the mesh as a matplotlib.tri.Triangulation instance. (2D only)
    """
    from matplotlib.tri import Triangulation
    conn = self.split("simplices").unstack()
    coords = self.nodes.coords.copy()
    node_map  = pd.Series(data = np.arange(len(coords)), index = coords.index)
    conn = node_map.loc[conn.values.flatten()].values.reshape(*conn.shape)
    return Triangulation(coords.x.values, coords.y.values, conn)
  
  def write_inp(self, *args, **kwargs):
    """
    Exports the mesh to the Abaqus INP format.
    """
    return write_inp(self, *args, **kwargs)
    
  def fields_metadata(self):
    """
    Returns fields metadata as a dataframe.
    """  
    return (pd.concat([f.metadata() for f in self.fields], axis = 1)
            .transpose()
            .sort_values(["step_num", "frame", "label", "position"]))
    
  
################################################################################
# FIELDS
################################################################################  
      
class MetaField(argiope.utils.Container):
  """
  A field mother class.
  
  :param label: field label
  :type label: str
  :param position: physical position
  :type position: in ["node", "element"]
  """ 
  _positions = ["node", "element"]
  
  def __init__(self, label = None, position = "node", 
               step_num = None, step_label = None,
               part = None,
               frame = None, frame_value = None, 
               data = None, **kwargs):
     self.label = label
     self.position = position
     self.step_num = step_num
     self.step_label = step_label
     self.frame = frame
     self.part = part
     self.frame_value = frame_value   
          
     # Data
     self.data =  data
     if hasattr(self, "_columns"): 
       self.data.columns = self._columns
     self.data.index.name = position
  
  def __repr__(self):
    return "<{0} {1} at {2}; step={3} ('{4}'); frame={5} >".format(
    self.__class__.__name__, self.label, 
    self.position, self.step_num, self.step_label, self.frame)
  
  def metadata(self):
    """
    Returns metadata as a dataframe.
    """
    return pd.Series({
           "part": self.part,
           "step_num": self.step_num,
           "step_label": self.step_label,
           "frame": self.frame,
           "frame_value": self.frame_value,
           "label": self.label,
           "position": self.position,                    
    })

class Field(MetaField):
  pass     

class Tensor6Field(MetaField):
  """
  Second order symmetric tensor field.
  """
  _columns = ["v11", "v22", "v33", "v12", "v13", "v23"]


class Tensor4Field(MetaField):
  """
  Second order symmetric 2D tensor field.
  """
  _columns = ["v11", "v22", "v33", "v12"]


class Tensor3Field(MetaField):
  """
  Second order diagonal tensor field.
  """
  _columns = ["v11", "v22", "v12"]
   
class Vector3Field(MetaField):
  """
  3D vector field.
  """
  _columns = ["v1", "v2", "v3"]

class Vector2Field(MetaField):
  """
  2D vector field.
  """
  _columns = ["v1", "v2"]  
  
class ScalarField(MetaField):
  _columns = ["v"]  
    
    
################################################################################
    

################################################################################
# PARSERS
################################################################################
def read_h5(hdfstore, group = ""):
  """
  DEPRECATED
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
  """
  Reads a GMSH MSH file and returns a :class:`Mesh` instance. 
  
  :arg path: path to MSH file.
  :type path: str
  
  """
  elementMap = { 15:"point1",
                 1:"line2",
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
              types = elements["etype"],
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

def write_inp(mesh, path = None, maxwidth = 40, sections = "solid"):
  """
  Exports the mesh to the INP format.
  """
  def set_to_inp(sets, keyword):
    ss = ""
    for sk in sets.keys():
      labels = sets[sk].loc[sets[sk]].index.values
      labels = list(labels)
      labels.sort()
      if len(labels)!= 0:
        ss += "*{0}, {0}={1}\n".format(keyword, sk)
        line = "  "
        counter = 2
        for l in labels:
          counter += 1
          s = "{0},".format(l)
          if (len(s) + len(line) < maxwidth) and counter < maxwidth:
            line += s
          else:
            ss += line + "\n  "
            line = s
            counter = 2
        ss += line + "\n"
    return ss.strip()[:-1]            

  # DATA
  mesh = mesh.copy()

  # NODES
  nodes_output = (mesh.nodes.coords.to_csv(header = False).split())
  nodes_output = ("\n".join(["  " + s.replace(",", ", ") for s in nodes_output]))
  
  # NODE SETS
  if "sets" in mesh.nodes.columns.levels[0]: 
    nsets = set_to_inp(mesh.nodes.sets, "NSET")
  else:
    nsets = "**"
  
  # SURFACES 
  surf_output = []
  if "surfaces" in mesh.elements.keys():
    sk = mesh.elements.surfaces.keys()
    for sindex in  np.unique(sk.labels[0]):
      slabel = sk.levels[0][sindex]
      surface = mesh.elements.surfaces[slabel]
      if surface.values.sum() != 0:
        mesh.surface_to_element_sets(slabel)
        surf_output.append( "*SURFACE, TYPE=ELEMENT, NAME={0}".format(slabel))
        for findex in surface.keys():
          if surface[findex].sum() != 0:
            surf_output.append("  _SURF_{0}_FACE{1}, S{1}".format(slabel, 
                                                                  findex[1:])) 
  else:
    surf_output.append("**")
  
  # ELEMENTS
  elements_output = ""
  for etype, group in mesh.elements.groupby((("type", "solver", ""),)):
    els = group.conn.replace(0, np.nan).to_csv(header = False, 
                                               float_format='%.0f').split()
    elements_output += "*ELEMENT, TYPE={0}\n".format(etype)
    elements_output += ("\n".join(["  " + s.strip().strip(",").
                             replace(",", ", ") for s in els]))
    elements_output += "\n"
  elements_output = elements_output.strip() 
  el_sets = {} 

  # MATERIALS
  section_output = ""
  for material, group in mesh.elements.groupby("materials"):
    slabel = "_MAT_{0}".format(material)
    mesh.elements[("sets", slabel, "")] = False
    mesh.elements.loc[group.index, ("sets", slabel, "")] = True
    if sections == "solid":
      section_output += "*SOLID SECTION, ELSET=_MAT_{0}, MATERIAL={0}\n".format(
       material)

  # ELEMENTS SETS
  if "sets" in mesh.elements.columns.levels[0]: 
    esets = set_to_inp(mesh.elements.sets.swaplevel(1,0, axis = 1)[""],"ELSET")
  else:
    esets = "**"
  """
  ek = mesh.elements.sets.keys()
  for esindex in  np.unique(ek.labels[0]):
    eslabel = ek.levels[0][esindex]
    eset = mesh.elements.sets[slabel]
  """
     
  # PATTERN
  pattern = Template(open(MODPATH + "/templates/mesh/inp.inp").read())
  pattern = pattern.substitute(
    NODES     = nodes_output,
    NODE_SETS = nsets,
    ELEMENTS  = elements_output,
    ELEMENT_SETS = esets,
    ELEMENT_SURFACES = "\n".join(surf_output),
    SECTIONS = section_output.strip())
  pattern = pattern.strip()
  if path == None:            
    return pattern
  else:
    open(path, "w").write(pattern)
  

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
