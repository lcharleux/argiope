import argiope

mesh = argiope.indentation.sample_mesh_2D("gmsh", "./workdir/")
argiope.indentation.indentation_input(sample_mesh = mesh, 
                                      path = "workdir/indentation_demo.inp")
