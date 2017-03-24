import argiope as ag
import numpy as np
import pandas as pd
from matplotlib.tri import Triangulation

mesh = ag.mesh.read_msh("demo.msh")
conn = mesh.split("simplices").unstack()
coords = mesh.nodes.coords.copy()
node_map  = pd.Series(data = np.arange(len(coords)), index = coords.index)
conn = node_map.loc[conn.values.flatten()].values.reshape(*conn.shape)
triangulation = Triangulation(coords.x.values, coords.y.values, conn)
"""
nodes, elements = mesh.nodes, mesh.elements
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
"""
