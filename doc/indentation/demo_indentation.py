from argiope import indentation

workdir = "./workdir/"
simName = "indentation_demo"


mesh = indentation.sample_mesh_2D("gmsh", 
                                   workdir, 
                                   lx = 1., 
                                   ly = 1., 
                                   r1 = 2., 
                                   r2 = 1000., 
                                   Nx = 32, 
                                   Ny = 16, 
                                   lc1 = 0.08, 
                                   lc2 = 200.,)

indentation.indentation_input(sample_mesh = mesh, 
                                     path = workdir + simName + ".inp")
                                      
indentation.indentation_abqpostproc(
        workdir     =  workdir, 
        path        = simName + "_abqpostproc.py", 
        odbPath     = simName + ".odb", 
        histPath    = simName + "_hist", 
        contactPath = simName + "_contact", 
        fieldPath   = simName + "_fields")
        
indentation.indentation_pypostproc(
        workdir     =  workdir, 
        path        = simName + "_pypostproc.py", 
        histPath    = simName + "_hist", 
        contactPath = simName + "_contact", 
        fieldPath   = simName + "_fields")                                              
