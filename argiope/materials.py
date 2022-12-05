import numpy as np
import pandas as pd
import os
import inspect
import argiope
from string import Template
import textwrap

MODPATH = os.path.dirname(inspect.getfile(argiope))


class Material:
    """
    A material meta class to rule them all.
    """
    _template = "Material"

    def __init__(self, label="Material", **kwargs):
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

    def __init__(self, young_modulus=1., poisson_ratio=0.3, **kwargs):
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

class ThermoElastic(Material):
    """
    An isotropic elastic material class.
    """
    _template = "ThermoElastic"

    def __init__(self, young_modulus=1., poisson_ratio=0.3,density=7800., conductivity=40.,specific_heat = 350., **kwargs):
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.density = density
        self.conductivity = conductivity
        self.specific_heat = specific_heat
        super().__init__(**kwargs)

    def write_inp(self):
        """
        Returns the material definition as a string in Abaqus INP format.
        """
        template = self.get_template()
        return template.substitute({"class": self.__class__.__name__,
                                    "label": self.label,
                                    "young_modulus": self.young_modulus,
                                    "poisson_ratio": self.poisson_ratio,
                                    "conductivity": self.conductivity,
                                    "density": self.density,
                                    "specific_heat": self.specific_heat}).strip()

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
                              .to_csv(header=False,
                                      index=False,
                                      sep=",").strip())}).strip()


class ElasticPerfectlyPlastic(Elastic):
    """
    A elastic perfectly plastic material.
    """
    _template = "ElasticPerfectlyPlastic"

    def __init__(self,  yield_stress=0.01,
                 **kwargs):
        self.yield_stress = yield_stress
        super().__init__(**kwargs)

    def write_inp(self):
        """
        Returns the material definition as a string in Abaqus INP format.
        """
        template = self.get_template()
        return template.substitute({"class": self.__class__.__name__,
                                    "label": self.label,
                                    "young_modulus": self.young_modulus,
                                    "poisson_ratio": self.poisson_ratio,
                                    "yield_stress": self.yield_stress}).strip()


class ElasticPlasticRateDep(Elastic):
    """
    An elastic plastic rate dependent material.
    """
    _template = "ElasticPlasticRateDep"

    def __init__(self, yield_stress=100,
                 multiplier=1.0e-6,
                 exponent=1.4,
                 **kwargs):
        self.yield_stress = yield_stress
        self.multiplier = multiplier
        self.exponent = exponent
        super().__init__(**kwargs)

    def write_inp(self):
        """
        Returns the material definition as a string in Abaqus INP format.
        """
        template = self.get_template()
        return template.substitute({"class": self.__class__.__name__,
                                    "label": self.label,
                                    "young_modulus": self.young_modulus,
                                    "poisson_ratio": self.poisson_ratio,
                                    "yield_stress": self.yield_stress,
                                    "multiplier": self.multiplier,
                                    "exponent": self.exponent}).strip()


class _PowerLawHardening(_ElasticPlastic):
    """
    A power law hardening meta class.
    """

    def __init__(self,  max_strain=10,
                 strain_data_points=100,
                 **kwargs):
        self.max_strain = max_strain
        self.strain_data_points = strain_data_points
        super().__init__(**kwargs)


class Hollomon(_PowerLawHardening):
    """
    An Hollomon material.
    """

    def __init__(self, hardening_exponent=0.3,
                 yield_stress=0.01,
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

    def __init__(self, hardening_exponent=0.3,
                 yield_stress=150.,
                 consistency=200.,
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
                             
class TabularHardening(_ElasticPlastic):
    """
    A Tabular hardening law
    """

    def __init__(self, plastic_strain= "",stress="stress",
                 **kwargs):
        self.plastic_strain = plastic_strain
        self.stress = stress   
        super().__init__(**kwargs)

    def get_plastic_table(self):
        """
        get_plastic_table 
        """
        return pd.DataFrame({"stress": self.stress,
                             "plastic_strain": self.plastic_strain})                             

class LinearDruckerPrager(Material):
    """
    A linear Drucker-Prager model (no hardening).
    """
    _template = "LinearDruckerPrager"

    def __init__(self, young_modulus=1., 
                 poisson_ratio=0.3,
                 yield_stress_d = 0.1, 
                 friction_angle_beta = 0.,
                 flow_angle_psi = 0.,
                 shape_factor_K = 1., **kwargs):
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio
        self.yield_stress_d = yield_stress_d
        self.friction_angle_beta = friction_angle_beta
        self.flow_angle_psi = flow_angle_psi
        self.shape_factor_K = shape_factor_K
        
        super().__init__(**kwargs)
    
    def write_inp(self):
        """
        Returns the material definition as a string in Abaqus INP format.
        """
        template = self.get_template()
        return template.substitute({"class": self.__class__.__name__,
                                    "label": self.label,
                                    "young_modulus": self.young_modulus,
                                    "poisson_ratio": self.poisson_ratio,
                                    "yield_stress_d":self.yield_stress_d,
                                    "friction_angle_beta": self.friction_angle_beta,
                                    "flow_angle_psi":self.flow_angle_psi,
                                    "shape_factor_K":self.shape_factor_K,
                                     }).strip()        
                                     
                                     
class ElasticEngineering(Material):
    """
    An anistropic elastic material class using engineering constants.
    """
    def __init__(self, E1=1., E2=1., E3=1.,
                 nu12=.3, nu13=.3, nu23=0.3,
                 G12=.71, G13=.71, G23=.71,
                 **kwargs):
        self.E1 = E1
        self.E2 = E2
        self.E3 = E3
        self.nu12 = nu12
        self.nu13 = nu13
        self.nu23 = nu23
        self.G12 = G12
        self.G13 = G13
        self.G23 = G23
        super().__init__(**kwargs)

    def write_inp(self):
        """
        Returns the material definition as a string in Abaqus INP format.
        """
        template = """
        ********************************************************************************
        ** ARGIOPE MATERIAL $class
        ********************************************************************************
        *MATERIAL, NAME = $label
        *ELASTIC, TYPE=ENGINEERING CONSTANTS
          $E1, $E2, $E3, $nu12, $nu13, $nu23, $G12, $G13,
          $G23
        """
        template = Template(textwrap.dedent(template).strip())
        return template.substitute({"class": self.__class__.__name__,
                                    "label": self.label,
                                    "E1": self.E1,
                                    "E2": self.E2,
                                    "E3": self.E3,
                                    "nu12": self.nu12,
                                    "nu13": self.nu13,
                                    "nu23": self.nu23,
                                    "G12": self.G12,
                                    "G13": self.G13,
                                    "G23": self.G23}).strip()                                     
