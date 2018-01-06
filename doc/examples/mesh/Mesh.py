import argiope as ag
import numpy as np

nlabels = np.arange(4) + 1 
coords = np.array([[0., 0., 0.],
                   [1., 0., 0.],
                   [1., 1., 0.],
                   [0., 1., 0.],])
elabels = np.array([1])
conn = np.array([[1, 2, 3, 4]])
types = ["quad4"]   
stypes = ["CPS4"]   
                
mesh = ag.mesh.Mesh(nlabels = nlabels,
                    coords = coords,
                    elabels = elabels,
                    conn = conn,
                    types = types,
                    stype = stypes)                   
