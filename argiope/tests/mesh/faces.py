import argiope as ag
import numpy as np
import pandas as pd

mesh = ag.mesh.read_msh("demo.msh")


kind = "edges"
ELEMENTS = ag.mesh.ELEMENTS
elements = mesh.elements
element_faces = []
face_labels   = []
face_number   = []
element_labels = np.array(elements.index)
element_connectivity = elements.conn.as_matrix()
element_etype        = elements["type", "argiope", mesh._null].as_matrix()
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
