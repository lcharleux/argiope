import meshio
import numpy as np
import argiope as ag

mesh = meshio.read("sample.msh")
# NODES
nodes_coordinates = mesh.points
nodes_labels = np.arange(len(nodes_coordinates)) + 1
# ELEMENTS
elementMap = {"point": "point1",
              "line" : "line2",
              "tri"  : "tri3",
              "quad" : "quad4",
             }
cells = mesh.cells_dict
cells_sets = mesh.cell_sets_dict
elements_connectivities = []
elements_types = []
elements_sets_raw = {k:[] for k in cells_sets.keys()}
for etype, conn in cells.items(): 
    elements_connectivities += (conn+1).tolist()
    etype_ag = elementMap[etype]
    elements_types +=  [ etype_ag for i in range(len(conn))] 
    # ELEMENT SETS
    for skey, sdata in cells_sets.items():
       elements_sets_raw[skey].append(np.zeros(len(conn), dtype = bool))  
       for setype, loc in sdata.items():
            if setype == etype:
                elements_sets_raw[skey][-1][loc] = True      

elements_sets = {}
for skey, data in elements_sets_raw.items():
    elements_sets[skey] = np.concatenate(data)
elements_labels = np.arange(len(elements_connectivities)) + 1

amesh = ag.mesh.Mesh(nlabels = nodes_labels,
                     coords = nodes_coordinates,
                     elabels = elements_labels,
                     conn = elements_connectivities,
                     types = elements_types,
                     esets = elements_sets)
