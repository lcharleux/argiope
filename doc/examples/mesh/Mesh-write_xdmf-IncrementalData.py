import numpy as np
import pandas as pd
import argiope

"""
This example shows how to export a mesh and fields at different increment.
A series of files is created: one file per increment.
"""

# Load a mesh
mesh = argiope.mesh.read_msh("rec.msh")

dist = np.sqrt(mesh.nodes.coords.x**2+mesh.nodes.coords.y**2)
for t in range(10):
    field_N = argiope.mesh.Vector3Field(label='dist', data=t * dist, position='node',
                                        frame=t,
                                        frame_value=t/10.)
    mesh.set_fields([field_N])



    # Write the mesh and the fields into xdmf file
    argiope.mesh.write_xdmf(mesh, "workdir/mesh_{:03d}".format(t))

# Open it with paraview (or other)