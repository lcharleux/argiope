import argiope as ag
import numpy as np
import pandas as pd

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
    "Line2": Element1D(
        nvert =  2),
    "Tri3":  Element2D(
        nvert = 3,
        edges = np.array([[0, 1],[1, 2], [2, 0]]),                   
        simplices = np.array([[0, 1, 2]]),
        angles = np.array([[2, 0, 1], [0, 1, 2], [1, 2, 0]]),
        optimal_angles = np.array([60., 60., 60.])),               
    "Quad4": Element2D(
        nvert = 4,
        edges = np.array([[0, 1], [1, 2], [2, 3],[3, 0]]),
        simplices = np.array([[0, 1, 3], [1, 2, 3]]),
        angles = np.array([[3, 0, 1], [0, 1, 2], [1, 2, 3], [2, 3, 0]]),
        optimal_angles = np.array([90., 90., 90., 90.])), 
    "Tetra4": Element3D(
        nvert = 4,
        edges = np.array([[0, 1],
                          [1, 2],
                          [2, 3],
                          [3, 0]]),
        faces = np.array([[0, 1, 2],
                          [0, 3, 1],
                          [1, 3, 2],
                          [2, 3, 0]]),                   
        faces_types = np.array(["Tri3",
                                "Tri3",
                                "Tri3",
                                "Tri3"]),
        simplices = np.array([[0, 1, 3, 4]])),
    "Pyra5": Element3D(
        nvert = 5,
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                          [0, 4], [1, 4], [2, 4], [3, 4]]),
        faces = [np.array([0, 1, 2, 3]),
                 np.array([0, 1, 4]), 
                 np.array([1, 2, 4]), 
                 np.array([2, 3, 4]), 
                 np.array([3, 0, 4])],          
        faces_types = np.array(["Quad4",
                               "Tri3", "Tri3", "Tri3", "Tri3"]),
        simplices = np.array([[0, 1, 3, 4],  [1, 2, 3, 4]])), 
    "Prism6": Element3D( 
        nvert = 6,
        edges = np.array([[0, 1], [1, 2], [2, 0], [3, 4],
                          [4, 5], [5, 3], [0, 3], [1, 4], [2, 5]]),
        faces = [np.array([0, 1, 2]), 
                 np.array([3, 5, 4]),
                 np.array([0, 3, 4, 1]), 
                 np.array([1, 4, 5, 2]), 
                 np.array([2, 5, 3, 0])],
        faces_types = np.array(["Tri3", "Tri3",
                               "Quad4", "Quad4", "Quad4"]),                      
        simplices = np.array([[0, 1, 2, 3],
                              [1, 2, 3, 4],
                              [2, 3, 4, 5]])),
    "Hexa8":Element3D(  
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
        faces_types = np.array(["Quad4", "Quad4", "Quad4", "Quad4", 
                                "Quad4", "Quad4"]),                      
        simplices =np.array([[0, 1, 3, 4],
                            [1, 2, 3, 4],
                            [3, 2, 7, 4],  
                            [2, 6, 7, 4],
                            [1, 5, 2, 4],
                            [2, 5, 6, 4]])),
                                        
       }    
"""
ELEMENTS = { "Line2": {"space": 1, 
                       "nvert": 2,
                       "faces": None,
                       "edges": None,
                       "simplices": None,
                       "angles":    None,
                       "optimal_angles": None
                      },
             "Tri3": {"space": 2, 
                      "nvert": 3,
                      "faces": np.array([[0, 1, 2]]),
                      "faces_types": np.array(["Tri3"]),
                      "edges": np.array([[0, 1],
                                         [1, 2],
                                         [2, 0]]),                   
                      "simplices": np.array([[0, 1, 2]]),
                      "angles": np.array([[2, 0, 1],
                                          [0, 1, 2],
                                          [1, 2, 0]]),
                      "optimal_angles": np.array([60., 
                                                  60., 
                                                  60.])                 
                      },
             "Quad4": {"space": 2, 
                       "nvert": 4,
                       "faces": np.array([[0, 1, 2, 3]]),
                       "faces_types": np.array(["Quad4"]),
                       "edges": np.array([[0, 1],
                                          [1, 2],
                                          [2, 3],
                                          [3, 0]]),
                       "simplices": np.array([[0, 1, 3], 
                                            [1, 2, 3]]),
                       "angles": np.array([[3, 0, 1],
                                           [0, 1, 2],
                                           [1, 2, 3],
                                           [2, 3, 0]]),
                       "optimal_angles": np.array([90., 
                                                  90., 
                                                  90.,
                                                  90.])                     
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
                       "faces_types": np.array(["Tri3",
                                                "Tri3",
                                                "Tri3",
                                                "Tri3"]),
                       "simplices": np.array([[0, 1, 3, 4]])
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
                       "faces_types": np.array(["Quad4",
                                               "Tri3",
                                               "Tri3",
                                               "Tri3",
                                               "Tri3"]),
                       "simplices": np.array([[0, 1, 3, 4],
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
                       "faces_types": np.array(["Tri3",
                                               "Tri3",
                                               "Quad4",
                                               "Quad4",
                                               "Quad4"]),                      
                       "simplices": np.array([[0, 1, 2, 3],
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
                       "faces_types": np.array(["Quad4",
                                               "Quad4",
                                               "Quad4",
                                               "Quad4",
                                               "Quad4",
                                               "Quad4"]),                      
                       "simplices": np.array([[0, 1, 3, 4],
                                            [1, 2, 3, 4],
                                            [3, 2, 7, 4],  
                                            [2, 6, 7, 4],
                                            [1, 5, 2, 4],
                                            [2, 5, 6, 4]])
                       },     
             }
      
  

ELEMENTS = ag.mesh.ELEMENTS
"""
mesh = ag.mesh.read_msh("demo.msh")
#mesh.elements = mesh.elements.iloc[:2]

into = "angles"
at  = "labels"
out = []
loc = None

if True:
  if type(loc) == type(None):
    elements = mesh.elements
  else:  
    elements = mesh.elements.loc[loc]
  out = []
  for etype, group in mesh.elements.groupby([("type", "argiope", "o")]):
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
      data = mesh.nodes.loc[out.values].values
      out = pd.DataFrame(index = out.index, data = data, columns = ["x", "y", "z"])
  
"""

  

  
  surfconn = surfconn.reshape(len(surfconn) * surfshape[0], 
                              surfshape[1] )
  loc = (np.ones((surfshape[1],len(surfconn)), dtype = np.int32) 
       * np.arange(len(surfconn))).T
  index = pd.MultiIndex.from_tuples([*zip(elements_labels[loc].flatten(), 
                                     surfaces_labels[loc].flatten())], 
                                     names = ["element", into])
  df = pd.DataFrame(data = surfconn, index = index )
         
  out.append(df)
  
out = pd.concat(out).stack()
out.index.names = ["element", into, "vertex"]
out.columns = ["node"]
#out = out.sort_values(["element", into])  
"""                          
"""
element_faces = []
face_labels   = []
face_number   = []
element_labels = np.array(elements.index)
element_connectivity = elements.conn.as_matrix()
element_etype        = elements["type", "argiope", self._null].as_matrix()
for i in range(len(element_labels)):
  etype = element_etype[i]
  conn  = element_connectivity[i]
  label = element_labels[i]
  faces = ELEMENTS[etype][kind]
  for fn in range(len(faces)):
    element_faces.append(np.int32(conn[faces[fn]]))
    face_labels.append(label)
    face_number.append(fn+1)
index = pd.MultiIndex.from_tuples(
        list(zip(*[face_labels, face_number])),
        names = ["element", "face"])
out = pd.DataFrame(data = element_faces, index = index)
"""
