import argiope as ag
import numpy as np
import pandas as pd


ELEMENTS = ag.mesh.ELEMENTS
mesh = ag.mesh.read_msh("demo.msh")
elements = mesh.elements
edges = mesh.split("edges", at = "coords").unstack()
edges["lx"] = edges.x[1]-edges.x[0]
edges["ly"] = edges.y[1]-edges.y[0]
edges["lz"] = edges.z[1]-edges.z[0]
edges["l"] = np.linalg.norm(edges[["lx", "ly", "lz"]], axis = 1)
edges = (edges.l).unstack()
edges.columns = pd.MultiIndex.from_product([["length"], 
                np.arange(edges.shape[1])])
edges[("stats", "lmax")] = edges.length.max(axis = 1)
edges[("stats", "lmin")] = edges.length.min(axis = 1)
edges[("stats", "aspect_ratio")] = edges.stats.lmax / edges.stats.lmin

#edges["length"] = np.linalg.norm(edges.values, axis = 1)


  
"""
coords = mesh.nodes.coords.copy()
node_map  = pd.Series(data = np.arange(len(coords)), index = coords.index)
conn = node_map.loc[conn.values.flatten()].values.reshape(*conn.shape)
"""
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
"""
