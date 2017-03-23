import argiope as ag
import numpy as np
import pandas as pd

ELEMENTS = ag.mesh.ELEMENTS
mesh = ag.mesh.read_msh("demo.msh")
#mesh.elements = mesh.elements.iloc[:5]

elements, nodes = mesh.elements, mesh.nodes
etypes = elements.type.argiope.o.unique()
out = []
for etype in etypes:
  etype_info = ELEMENTS[etype]
  edges_info = etype_info["edges"]
  loc = elements.type.argiope.o == etype
  index = elements.loc[loc].index
  edges_data = mesh.split(into = "edges", 
                         loc = loc,
                         at = "coords")
  data = edges_data.values.reshape(index.size, 
                                    edges_info.shape[0],
                                    edges_info.shape[1],
                                    3)        
  length = np.linalg.norm((data[:,:,1] - data[:,:,0]), axis = 2)
  df = pd.DataFrame(index = index, 
                    data = length, 
                    columns = pd.MultiIndex.from_product(
                              [["length"], range(edges_info.shape[0])]))  
  df["stats", "aspect_ratio"] = df.length.max(axis = 1) / df.length.min(axis = 1)
  out.append(df)
out = pd.concat(out)      
                         
                          
"""
  
  loc = elements.type.argiope.o == etype
  index = elements.loc[loc].index
  simplices_data = mesh.split(into = "simplices", 
                         loc = loc,
                         at = "coords") 
  simplices = simplices_data.values.reshape(
              index.size, 
              simplices_info.shape[0], 
              simplices_info.shape[1], 
              3) 
  edges = edges = simplices[:,:,1:] - simplices[:,:,:1] 
  simplices_centroids = simplices.mean(axis = 2)
  if etype_info["space"] == 2:
    simplices_volumes = np.linalg.norm(
              np.cross(edges[:,:,0], 
                       edges[:,:,1], 
                       axis = 2),
              axis = 2)
  elif etype_info["space"] == 3:          
    simplices_volumes =  (np.cross(edges[:,:,0], 
                                   edges[:,:,1], axis = 2) 
                         * edges[:,:, 2]).sum(axis = 2)
  elements_volumes = simplices_volumes.sum(axis = 1)
  elements_centroids = ((simplices_volumes.reshape(*simplices_volumes.shape, 1) 
                      * simplices_centroids).sum(axis = 1) 
                      / elements_volumes.reshape(*elements_volumes.shape,1))
  out.append(pd.DataFrame(index = index, 
                          data = {"volume" : elements_volumes,
                          "xg": elements_centroids[:,0],
                          "yg": elements_centroids[:,1],
                          "zg": elements_centroids[:,2],}))             

out = pd.concat(out)  
"""
"""
  typedata = ELEMENTS[etype]
  simpmap = typedata["simplices"]
  simpshape = simpmap.shape
  conn = elements.conn[elements.type.argiope.o == etype]
  simplices = nodes.coords.loc[conn.values[:, simpmap].flatten()].values.reshape(
              len(conn), simpshape[0], simpshape[1], 3) 
  edges = edges = simplices[:,:,1:] - simplices[:,:,:1] 
  simplices_centroids = simplices.mean(axis = 2)
  if typedata["space"] == 2:
    simplices_volumes = np.linalg.norm(np.cross(edges[:,:,0], edges[:,:,1], axis = 2)
           , axis = 2)
  elif typedata["space"] == 3:          
    print("todo")
  elements_volumes = simplices_volumes.sum(axis = 1)
  elements_centroids = ((simplices_volumes.reshape(*simplices_volumes.shape, 1) 
                      * simplices_centroids).sum(axis = 1) 
                      / elements_volumes.reshape(*elements_volumes.shape,1))
  out.append(pd.DataFrame(index = conn.index, 
                          data = {"volume" : elements_volumes,
                          "xg": elements_centroids[:,0],
                          "yg": elements_centroids[:,1],
                          "zg": elements_centroids[:,2],}))
out = pd.concat(out)
"""  
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
