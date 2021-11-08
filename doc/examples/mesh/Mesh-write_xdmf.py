import numpy as np
import pandas as pd
import argiope

# Load a mesh
mesh = argiope.mesh.read_msh("demo.msh")

# Get some stats on the mesh
df = mesh.stats()

# Fields definition
field_Vol = argiope.mesh.ScalarField(label='Vol', data=df.volume, position='element')
mesh.set_fields([field_Vol])

field_C = argiope.mesh.Vector3Field(label='centroid', data=df.centroid, position='element')
mesh.add_fields([field_C])

dist = np.sqrt(mesh.nodes.coords.x**2+mesh.nodes.coords.y**2)
field_N = argiope.mesh.Vector3Field(label='dist', data=dist, position='node')
mesh.add_fields([field_N])

# Write the mesh and the fields into xdmf file
argiope.mesh.write_xdmf(mesh, "workdir/mesh")

# Open it with paraview (or other)