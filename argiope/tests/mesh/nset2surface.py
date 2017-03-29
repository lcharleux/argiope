import argiope as ag
import numpy as np
import pandas as pd
import hardness as hd


mesh= hd.models.sample_mesh_2D()

ELEMENTS = ag.mesh.ELEMENTS
tag = "SURFACE"
# Create a dummy node with label 0
nodes = mesh.nodes.copy()
dummy = nodes.iloc[0].copy()
dummy["coords"] *= np.nan
dummy["sets"] = True
nodes.loc[0] = dummy
# Getting element surfaces
element_surfaces= mesh.split("surfaces").unstack()
# killer hack !
surf = pd.DataFrame(
         nodes.sets[tag].loc[element_surfaces.values.flatten()]
               .values.reshape(element_surfaces.shape)
               .prod(axis = 1)
               .astype(np.bool),
         index = element_surfaces.index).unstack()
         
"""
element_surfaces= self.split("surfaces").unstack()
    surf = pd.DataFrame(np.prod(self.nodes.sets[tag].loc[
                element_surfaces.values.flatten()]
               .values.reshape(element_surfaces.shape),axis = 1)
               .astype(np.bool),
               index = element_surfaces.index).unstack()
    for k in surf.keys():
      self.elements["surfaces", tag, k[1]] = surf.loc[:, k]
"""
#mesh.element_set_to_node_set(tag)
#mesh.node_set_to_surface(tag)

"""
loc = mesh.elements.loc[:, ("sets", tag, mesh._null)].as_matrix().flatten()
nlabels = np.unique(mesh.elements.conn.replace(0, np.nan).as_matrix()[loc].flatten())
mesh.nodes[("sets", tag)] = False
mesh.nodes.loc[nlabels, ("sets", tag)] = False 
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
