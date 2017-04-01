import numpy as np
import pandas as pd
import os, inspect, argiope, subprocess, time
from string import Template

MODPATH = os.path.dirname(inspect.getfile(argiope))

################################################################################
# MODEL DEFINITION
################################################################################
class Model:
  """
  Model meta class. 
  
  Note: should move Argiope as soon as it is working properly.
  """
  def __init__(self, 
               label, 
               meshes, 
               steps, 
               materials, 
               solver = "abaqus", 
               solver_path = "",
               workdir = "./workdir",
               verbose = True):
    self.label        = label
    self.meshes      = meshes
    self.steps       = steps
    self.materials   = materials
    self.solver      = solver
    self.solver_path = solver_path
    self.workdir     = workdir
    self.verbose     = verbose
   
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
      process = subprocess.Popen( 
                [self.solver_path,  'viewer', 'noGUI={0}_abqpp.py'.format(
                                                                   self.label)], 
                cwd = self.workdir,
                stdout = subprocess.PIPE )
      trash = process.communicate()
      print(trash)
    t1 = time.time()
    if self.verbose: 
      print('<Post-Processed {0}: duration {1:.2f}s>'.format(self.label, 
                                                                  t1 - t0)) 
                                                                  
################################################################################                                                                  
