import numpy as np
import pandas as pd
execfile("mesh.py")

mesh = read_msh("../doc/mesh/demo.msh")
path = "mesh.h5"
mesh.h5path = path
mesh.save()
del mesh
mesh = read_h5(path)
