import numpy as np
import pandas as pd
import os, inspect, argiope
from string import Template

MODPATH = os.path.dirname(inspect.getfile(argiope))

class Material:
  """
  A material meta class to rule them all.
  """
  _scalar_attributes = {"label": "Material"}
  _tabular_attributes = {}
  _template = "Material"
  def __init__(self, **kwargs):
    scalar_attributes = self._scalar_attributes.copy()
    for k,v in kwargs.items():
      if k in scalar_attributes.keys():
        scalar_attributes[k] = v
    self.scalar_data = pd.Series(scalar_attributes)
    
    tabular_attributes = self._tabular_attributes.copy()
    for k in tabular_attributes.keys():
      if k in kwargs.keys(): tabular_attributes[k] = kwargs[v]
      setattr(self, k, tabular_attributes[k])
    
      
    
  def write_inp(self):
     """
     Returns the material definition as a string in Abaqus INP format.
     """
     out = self.scalar_data.to_csv(path= None)
     out = out.split()
     out = "\n".join(["** " + s.replace(",", " = ") for s in out]) + "\n"
     pattern = Template(open(MODPATH + "/templates/materials/{0}.inp".format(
               self._template)).read())
     out += pattern.substitute(self.scalar_data.to_dict())
     out = (80 * "*" + "\n" + "** MATERIAL CLASS: {0} / LABEL: {1}\n".format(
           self.__class__.__name__, self.scalar_data.label) 
           + 80 * "*" + "\n" + out )
     return out 
  
class Elastic(Material):
   """
   An isotropic elastic material class.
   """
   _scalar_attributes = {"label"         : "Material",
                          "Young_modulus" : 1. , 
                          "Poisson_ratio" : .3}
   _template = "Elastic"
   
class ElasticPlastic(Elastic):
  """
  An elastic plastic material meta class.
  """
  _template = "Elastic"
  
  def write_inp(self):
    out = super().write_inp()
    out += "*PLASTIC\n"
    out += self.plastic_data[["stress", "plastic_strain"]].to_csv(
           header = False, 
           index = False,
           sep = ",").strip()

    return out
    
   
class Hollomon(ElasticPlastic):
   """
   An Hollomon material.
   """
   _scalar_attributes = {"label"             : "Material",
                         "Young_modulus"     : 1. , 
                         "Poisson_ratio"     : .3,
                         "hardening_exponent": .2,
                         "yield_stress"      : 0.01,
                         "max_strain"        : 10.,
                         "strain_data_points": 1000}
                          
   def get_plastic_data(self):
     """
     Calculates the plastic data
     """
     data = self.scalar_data
     E = data["Young_modulus"]
     sy = data["yield_stress"]
     n = data["hardening_exponent"]
     eps_max = data["max_strain"]
     Np = data["strain_data_points"]
     ey = sy/E
     s = 10.**np.linspace(0., np.log10(eps_max/ey), Np)
     strain = ey * s
     stress = sy * s**n
     plastic_strain = strain - stress / E 
     return pd.DataFrame({"strain": strain, 
                          "stress": stress, 
                          "plastic_strain": plastic_strain})
   plastic_data = property(get_plastic_data)
                                                
                       
