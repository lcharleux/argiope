import numpy as np
import pandas as pd
import argiope
import os, subprocess, inspect
MODPATH = os.path.dirname(inspect.getfile(argiope))


def sample_mesh_2D(gmsh_path, workdir, geo_name = "dummy"):
  """
  Builds an indentation mesh.
  """
  geo = open(MODPATH + "/templates/indentation/indentation_mesh_2D.geo").read()
  open(workdir + geo_name + ".geo", "w").write(geo)
  p = subprocess.Popen("{0} -2 {1}".format(gmsh_path, geo_name + ".geo"), cwd = workdir, shell=True, stdout = subprocess.PIPE)
  trash = p.communicate()
  mesh = argiope.mesh.read_msh(workdir + geo_name + ".msh")
  mesh.element_set_to_node_set(tag = "SURFACE")
  mesh.element_set_to_node_set(tag = "BOTTOM")
  mesh.element_set_to_node_set(tag = "AXIS")
  del mesh.elements.sets["SURFACE"]
  del mesh.elements.sets["BOTTOM"]
  del mesh.elements.sets["AXIS"]
  mesh.elements.data = mesh.elements.data[mesh.elements.data.etype != "Line2"] 
  mesh.node_set_to_surface("SURFACE")
  mesh.elements.add_set("ALL_ELEMENTS", mesh.elements.data.index)  
  return mesh
  
  
def indentation_input(sample_mesh, path = None):
  """
  Returns a indentation INP file.
  """
  pattern = open(MODPATH + "/templates/indentation/indentation.inp").read()
  element_map = {"Tri3":  "CAX3", 
                 "Quad4": "CAX4", }
  pattern = pattern.replace("#SAMPLE_MESH", 
                            sample_mesh.to_inp(element_map = element_map))
  if path == None:            
    return pattern
  else:
    open(path, "wb").write(pattern)  
