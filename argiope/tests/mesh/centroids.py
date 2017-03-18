import argiope as ag
import numpy as np
import pandas as pd

def tri_area(vert):
  return np.linalg.norm(np.cross( v[1]-v[0], v[2]-v[0])) / 2.   


ELEMENTS = ag.mesh.ELEMENTS
mesh = ag.mesh.read_msh("demo.msh")
mesh.elements = mesh.elements.iloc[:2]

elements, nodes = mesh.elements, mesh.nodes
ne = len(elements)
centroids = np.zeros((ne, 3))
volumes = np.zeros(ne)
etypes = elements.type.argiope.o.unique()

for etype in etypes:
  typedata = ELEMENTS[etype]
  simpmap = typedata["simplex"]
  simpshape = simpmap.shape
  conn = elements.conn[elements.type.argiope.o == etype].values
  lconn = len(conn)
  simplices = nodes.coords.loc[conn[:, simpmap].flatten()].values.reshape(
              lconn, simpshape[0], simpshape[1], 3) 

  
"""
#ELEMENTS
connectivities  = elements.conn.as_matrix()
connectivities[np.isnan(connectivities)] = 0
connectivities = connectivities.astype(np.int32)
connectivities = nodes_map[connectivities]
etype           = np.array(elements.type.argiope[mesh._null])

#CENTROIDS & VOLUME
centroids, volumes = [], []
for i in range(len(etype)):
  simplices = connectivities[i][ag.mesh.ELEMENTS[etype[i]]["simplex"]]
  simplices = np.array([ [coords[n] for n in simp] for simp in simplices])
  v = np.array([ag.mesh.tri_area(simp) for simp in simplices])
  g = np.array([simp.mean(axis=0) for simp in simplices])
  vol = v.sum()
  centroids.append((g.transpose()*v).sum(axis=1) / vol)
  volumes.append(vol)

centroids = np.array(centroids)
volumes   = np.array(volumes)
out["volume", mesh._null] = volumes
out["centroid", "x"] = centroids[:, 0]
out["centroid", "y"] = centroids[:, 1]
out["centroid", "z"] = centroids[:, 2]
"""
