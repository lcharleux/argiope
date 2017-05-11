import numpy as np
import pandas as pd
import os, inspect, argiope
from string import Template

MODPATH = os.path.dirname(inspect.getfile(argiope))

class Material:
  """
  A material meta class to rule them all.
  """
  _template = "Material"
  
  def __init__(self, label = "Material", **kwargs):
    self.label = label
    
  def get_template(self):
    return Template(open(MODPATH + "/templates/materials/{0}.inp".format(
           self._template)).read())    
  
  def write_inp(self):
     """
     Returns the material definition as a string in Abaqus INP format.
     """
     template = self.get_template()
     return template.substitute({"class": self.__class__.__name__,
                                 "label": self.label}).strip()
  
class Elastic(Material):
   """
   An isotropic elastic material class.
   """
   _template = "Elastic"
   
   def __init__(self, young_modulus = 1., poisson_ratio = 0.3, **kwargs):
     self.young_modulus = young_modulus
     self.poisson_ratio = poisson_ratio
     super().__init__(**kwargs)
   
   def write_inp(self):
     """
     Returns the material definition as a string in Abaqus INP format.
     """
     template = self.get_template()
     return template.substitute({"class": self.__class__.__name__,
                                 "label": self.label,
                                 "young_modulus": self.young_modulus,
                                 "poisson_ratio": self.poisson_ratio}).strip()
   
   
class _ElasticPlastic(Elastic):
  """
  An elastic plastic material **meta** class.
  """
  _template = "ElasticPlastic"
  
  def write_inp(self):
    """
    Returns the material definition as a string in Abaqus INP format.
    """
    template = self.get_template()
    plastic_table = self.get_plastic_table()
    return template.substitute({
        "class": self.__class__.__name__,
        "label": self.label,
        "young_modulus": self.young_modulus,
        "poisson_ratio": self.poisson_ratio,
        "plastic_table": (self.get_plastic_table()[["stress", "plastic_strain"]]
                          .to_csv(header = False, 
                                  index = False,
                                  sep = ",").strip())}).strip()


class ElasticPerfectlyPlastic(_ElasticPlastic):
  """
  A elastic perfectly plastic material.
  """
  def __init__(self,  yield_stress = 0.01,
                      **kwargs):
     self.yield_stress = yield_stress
     super().__init__(**kwargs)
  
  def get_plastic_table(self):
     """
     Calculates the plastic data
     """
     return pd.DataFrame({"stress": [self.yield_stress], 
                          "plastic_strain": [0.]})



class _PowerLawHardening(_ElasticPlastic):
  """
  A power law hardening meta class.
  """
  def __init__(self,  max_strain = 10,
                      strain_data_points = 100,
                      **kwargs):
     self.max_strain = max_strain
     self.strain_data_points = strain_data_points
     super().__init__(**kwargs)    
  
class Hollomon(_PowerLawHardening):
   """
   An Hollomon material.
   """
   def __init__(self, hardening_exponent = 0.3,  
                      yield_stress = 0.01,
                      **kwargs):
     self.hardening_exponent = hardening_exponent
     self.yield_stress = yield_stress
     super().__init__(**kwargs)
                          
   def get_plastic_table(self):
     """
     Calculates the plastic data
     """
     E = self.young_modulus
     sy = self.yield_stress
     n = self.hardening_exponent
     eps_max = self.max_strain
     Np = self.strain_data_points
     ey = sy/E
     s = 10.**np.linspace(0., np.log10(eps_max/ey), Np)
     strain = ey * s
     stress = sy * s**n
     plastic_strain = strain - stress / E 
     return pd.DataFrame({"strain": strain, 
                          "stress": stress, 
                          "plastic_strain": plastic_strain})
   
                                                
class PowerLin(_PowerLawHardening):
   """
   A Power Linear material :
   S = Sy  + K (Ep)**n 
   """
   def __init__(self, hardening_exponent = 0.3,  
                      yield_stress = 150.,
                      consistency = 200.,
                      **kwargs):
     self.hardening_exponent = hardening_exponent
     self.yield_stress = yield_stress
     self.consistency = consistency
     super().__init__(**kwargs)
                          
   def get_plastic_table(self):
     """
     Calculates the plastic data
     """
     K = self.consistency
     sy = self.yield_stress
     n = self.hardening_exponent
     eps_max = self.max_strain
     Np = self.strain_data_points
     plastic_strain = np.linspace(0., eps_max, Np)
     stress = sy + K * plastic_strain**n 
     return pd.DataFrame({"stress": stress, 
                          "plastic_strain": plastic_strain})                       
