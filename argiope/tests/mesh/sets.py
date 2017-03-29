import argiope as ag
import numpy as np
import pandas as pd


ELEMENTS = ag.mesh.ELEMENTS
mesh = ag.mesh.read_msh("demo.msh")
tag = "SURFACE"

nodes, elements = mesh.nodes, mesh.elements
loc = elements.conn[elements.sets.SURFACE[mesh._null]].stack().stack().unique()
loc = loc[loc != 0]
nodes.loc[loc, ("sets", tag) ] = True

"""
loc = mesh.elements.loc[:, ("sets", tag, self._null)].as_matrix().flatten()
nlabels = np.unique(self.elements.conn.as_matrix()[loc].flatten())
self.nodes[("sets", tag)] = False
self.nodes.loc[nlabels, ("sets", tag)] = False 
"""
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
