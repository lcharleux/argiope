import argiope as ag
import numpy as np
import pandas as pd
import hardness as hd
import copy, os, inspect 
from string import Template
MODPATH = os.path.dirname(inspect.getfile(ag))

mesh= hd.models.sample_mesh_2D( Nx = 4, Ny = 4, Nr = 2, Nt = 8, r2 = 4)
maxwidth = 80
sections = "solid"

mesh.write_inp(path = "inp.inp")
"""
def set_to_inp(sets, keyword):
  ss = ""
  for sk in sets.keys():
    labels = sets[sk].loc[sets[sk]].index.values
    labels = list(labels)
    labels.sort()
    if len(labels)!= 0:
      ss += "*{0}, {0}={1}\n".format(keyword, sk)
      line = "  "
      counter = 2
      for l in labels:
        counter += 1
        s = "{0},".format(l)
        if (len(s) + len(line) < maxwidth) and counter <maxwidth:
          line += s
        else:
          ss += line + "\n  "
          line = s
          counter = 2
      ss += line
    ss += "\n"
  return ss.strip()[:-1]            

# DATA
mesh = mesh.copy()

# NODES
nodes_output = (mesh.nodes.coords.to_csv(header = False).split())
nodes_output = ("\n".join(["  " + s.replace(",", ", ") for s in nodes_output]))

# SURFACES 
surf_output = []
sk = mesh.elements.surfaces.keys()
for sindex in  np.unique(sk.labels[0]):
  slabel = sk.levels[0][sindex]
  surface = mesh.elements.surfaces[slabel]
  if surface.values.sum() != 0:
    mesh.surface_to_element_sets(slabel)
    surf_output.append( "*SURFACE, TYPE=ELEMENT, NAME={0}".format(slabel))
    for findex in surface.keys():
      surf_output.append("  _SURF_{0}_FACE{1}, S{1}".format(slabel, findex+1)) 

# ELEMENTS
elements_output = ""
for etype, group in mesh.elements.groupby((("type", "solver", mesh._null),)):
  els = group.conn.replace(0, np.nan).to_csv(header = False, 
                                             float_format='%.0f').split()
  elements_output += "*ELEMENT, TYPE={0}\n".format(etype)
  elements_output += ("\n".join(["  " + s.strip().strip(",").
                           replace(",", ", ") for s in els]))
  elements_output += "\n"
elements_output = elements_output.strip() 
el_sets = {} 

# MATERIALS
section_output = ""
for material, group in mesh.elements.groupby("materials"):
  slabel = "_MAT_{0}".format(material)
  mesh.elements[("sets", slabel, mesh._null)] = False
  mesh.elements.loc[group.index, ("sets", slabel, mesh._null)] = True
  if sections == "solid":
    section_output += "*SOLID SECTION, ELSET=_MAT_{0}, MATERIAL={0}\n".format(
     material)

# ELEMENTS SETS
ek = mesh.elements.sets.keys()
for esindex in  np.unique(ek.labels[0]):
  eslabel = ek.levels[0][esindex]
  eset = mesh.elements.sets[slabel]
   
# PATTERN
pattern = Template(open(MODPATH + "/templates/mesh/inp.inp").read())
pattern = pattern.substitute(
  NODES     = nodes_output,
  NODE_SETS = set_to_inp(mesh.nodes.sets, "NSET"),
  ELEMENTS  = elements_output,
  ELEMENT_SETS = set_to_inp(mesh.elements.sets
                 .swaplevel(1,0, axis = 1)[mesh._null], "ELSET"),
  ELEMENT_SURFACES = "\n".join(surf_output),
  SECTIONS = section_output.strip())
pattern = pattern.strip()
"""

#open("inp.inp", "w").write(pattern)
