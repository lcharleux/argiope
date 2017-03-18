import argiope as ag
import numpy as np
import pandas as pd

ELEMENTS = ag.mesh.ELEMENTS
mesh = ag.mesh.read_msh("demo.msh")
#mesh.elements = mesh.elements.iloc[:10]

kind = "edges"
out = []
elements = mesh.elements
etypes = elements.type.argiope.o.unique()
for etype in etypes:
  typedata = ELEMENTS[etype]
  surfmap = typedata[kind]
  surfshape = surfmap.shape
  conn = elements.conn[elements.type.argiope.o == etype]
  elements_labels = ((conn.index.values * np.ones((surfshape[0], 1), dtype = np.int32))
              .T.flatten())
  surfaces_labels = (np.arange(1,surfshape[0]+1).reshape(surfshape[0],1) 
                   * np.ones(len(conn), dtype = np.int32)
                   ).T.flatten()

  surfconn = conn.values[:, surfmap]
  surfconn = surfconn.reshape(len(surfconn) * surfshape[0], surfshape[1] )
  df = pd.DataFrame(data = {"element": elements_labels,
                            "face": surfaces_labels}, )
  for i in range(surfshape[1]):
    df[i] = surfconn[:, i]                  
  out.append(df)

out = pd.concat(out)
out.sort_values(["element", "face"])  
                          
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
