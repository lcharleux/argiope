import argiope as ag
import numpy as np
import pandas as pd
import hardness as hd

ELEMENTS = ag.mesh.ELEMENTS
#mesh = ag.mesh.read_msh("dummy.msh")
part = hd.models.SpheroconicalIndenter2D(
                                   R = 1.,
                                   psi= 30., 
                                   r1 = 1., 
                                   r2 = 3., 
                                   r3 = 3., 
                                   lc1 = .05, 
                                   lc2 = 1.,
                                   rigid = False,
                                   gmsh_path = "gmsh",
                                   file_name = "dummy", 
                                   workdir = "./", 
                                   gmsh_space = 2, 
                                   gmsh_options = "-algo 'delquad'",
                                   element_map = None,
                                   material_map = None)
part.make_mesh()
mesh = part.mesh
elements = mesh.elements


into = "simplices"
at  = "coords"
out = []
loc = None

if True:
  if type(loc) == type(None):
      elements = mesh.elements
  else:  
    elements = mesh.elements.loc[loc]
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
      data = mesh.nodes.coords.loc[out.values].values
      print(data)

      out = pd.DataFrame(index = out.index, data = data, 
                         columns = ["x", "y", "z"])
   
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
