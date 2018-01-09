import numpy as np
import pandas as pd
import os, inspect, argiope, subprocess, time
from string import Template

MODPATH = os.path.dirname(inspect.getfile(argiope))

################################################################################
# MODEL DEFINITION
################################################################################
class Model(argiope.utils.Container):
  """
  Model meta class. 
  """
  def __init__(self, 
               label, 
               parts, 
               steps, 
               materials, 
               solver = "abaqus", 
               solver_path = "",
               workdir = "./workdir",
               verbose = False):
    self.label       = label
    self.parts       = parts
    self.steps       = steps
    self.materials   = materials
    self.solver      = solver
    self.solver_path = solver_path
    self.workdir     = workdir
    self.verbose     = verbose
    self.data = {}
   
  def make_directories(self):
    """
    Checks if required directories exist and creates them if needed.
    """
    if os.path.isdir(self.workdir) == False: os.mkdir(self.workdir)
  
  def run_simulation(self):
    """
    Runs the simulation.
    """
    self.make_directories()
    t0 = time.time()
    if self.verbose: 
      print('<Running "{0}" using {1}>'.format(self.label, 
                                               self.solver))  
    if self.solver == "abaqus":
      command = '{0} job={1} input={1}.inp interactive ask_delete=OFF'.format(
                self.solver_path, 
                self.label) 
      process = subprocess.Popen(command, 
                                 cwd = self.workdir, 
                                 shell=True, 
                                 stdout = subprocess.PIPE)
      trash = process.communicate()
    if self.verbose:
      print(trash)  
    t1 = time.time()
    if self.verbose: 
      print('<Ran {0}: duration {1:.2f}s>'.format(self.label, t1 - t0))   
  
  def run_postproc(self):
    """
    Runs the post-proc script.
    """
    t0 = time.time()
    if self.verbose: 
      print('<Post-Processing"{0}" using {1}>'.format(self.label, 
                                               self.solver))  
    if self.solver == "abaqus":
      command = '{0} viewer noGUI={1}_abqpp.py'.format(self.solver_path, self.label)
      print(command)
      process = subprocess.Popen( 
                command, 
                cwd = self.workdir,
                shell=True,
                stdout = subprocess.PIPE )
      trash = process.communicate()
    if self.verbose:
      print(trash) 
    t1 = time.time()
    if self.verbose: 
      print('<Post-Processed {0}: duration {1:.2f}s>'.format(self.label, 
                                                                  t1 - t0)) 
                                                                  
################################################################################ 


################################################################################
# PART DEFINITION
################################################################################
class Part(argiope.utils.Container):

  def __init__(self, gmsh_path = "gmsh",
                     file_name = "dummy", 
                     workdir = "./", 
                     gmsh_space = 2, 
                     gmsh_options = "",
                     element_map = None,
                     material_map = None):
    self.gmsh_path  = gmsh_path
    self.file_name  = file_name
    self.workdir    = workdir 
    self.gmsh_space = gmsh_space
    self.gmsh_options = gmsh_options
    self.mesh       = None
    self.element_map = element_map
    self.material_map = material_map
    
  def run_gmsh(self):
    """
    Makes the mesh using gmsh.
    """
    argiope.utils.run_gmsh(gmsh_path = self.gmsh_path,
                           gmsh_space = self.gmsh_space,
                           gmsh_options = self.gmsh_options,
                           name = self.file_name + ".geo",
                           workdir = self.workdir)  
    self.mesh = argiope.mesh.read_msh(self.workdir + self.file_name + ".msh")
    
  def make_mesh(self):
    self.preprocess_mesh()
    self.run_gmsh()
    self.postprocess_mesh()  
    

################################################################################ 
################################################################################                                                                 
