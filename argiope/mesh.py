import numpy as np
import pandas as pd
import matplotlib as mpl
import StringIO



ELEMENTS = { "Line2": {"space": 1, 
                       "nvert": 2,
                       "simplex": np.array([[0, 1]])
                      },
             "Tri3": {"space": 2, 
                      "nvert": 3,
                      "faces": np.array([0, 1, 2]),
                      "edges": np.array([[0, 1],
                                         [1, 2],
                                         [2, 0]]),
                      "simplex": np.array([[0, 1, 2]])
                      },
             "Quad4": {"space": 2, 
                       "nvert": 4,
                       "faces": np.array([0, 1, 2, 3]),
                       "edges": np.array([[0, 1],
                                          [1, 2],
                                          [2, 3],
                                          [3, 0]]),
                       "simplex": np.array([[0, 1, 3], 
                                            [1, 2, 3]])
                      },                           
             "Tetra4":{"space": 3, 
                       "nvert": 4,
                       "edges": np.array([[0, 1],
                                          [1, 2],
                                          [2, 3],
                                          [3, 0]]),
                       "faces": np.array([[0, 1, 2],
                                               [0, 3, 1],
                                               [1, 3, 2],
                                               [2, 3, 0]]),                   
                       "simplex": np.array([[0, 1, 3, 4]])
                       },                           
             "Pyra5":{"space": 3, 
                       "nvert": 5,
                       "edges": np.array([[0, 1],
                                          [1, 2],
                                          [2, 3],
                                          [3, 0],
                                          [0, 4],
                                          [1, 4],
                                          [2, 4],
                                          [3, 4]]),
                       "faces": np.array([[0, 1, 2, 3],
                                          [0, 1, 4],
                                          [1, 2, 4],
                                          [2, 3, 4],
                                          [3, 0, 4]]),          
                       "simplex": np.array([[0, 1, 3, 4],
                                            [1, 2, 3, 4]])
                       },     
             "Prism6":{"space": 3, 
                       "nvert": 6,
                       "edges": np.array([[0, 1],
                                          [1, 2],
                                          [2, 0],
                                          [3, 4],
                                          [4, 5],
                                          [5, 3],
                                          [0, 3],
                                          [1, 4],
                                          [2, 5]]),
                       "faces": np.array([[0, 1, 2],
                                          [3, 5, 4],
                                          [0, 3, 4, 1],
                                          [1, 4, 5, 2],
                                          [2, 5, 3, 0]]),   
                       "simplex": np.array([[0, 1, 2, 3],
                                            [1, 2, 3, 4],
                                            [2, 3, 4, 5]]) 
                       },     
             "Hexa8":{"space": 3, 
                      "nvert": 8,
                      "edges": np.array([[0, 1],
                                         [1, 2],
                                         [2, 3],
                                         [3, 0],
                                         [4, 5],
                                         [5, 6],
                                         [6, 7],
                                         [7, 4],
                                         [0, 4],
                                         [1, 5],
                                         [2, 6],
                                         [3, 7]]),
                       "faces": np.array([[0, 1, 2, 3],
                                          [4, 7, 6, 5],
                                          [0, 4, 5, 1],
                                          [1, 5, 6, 2],
                                          [2, 6, 7, 3],
                                          [3, 7, 4, 0]]),
                       "simplex": np.array([[0, 1, 3, 4],
                                            [1, 2, 3, 4],
                                            [3, 2, 7, 4],  
                                            [2, 6, 7, 4],
                                            [1, 5, 2, 4],
                                            [2, 5, 6, 4]])
                       },     
             }

def tri_area(vertices):
  u = vertices[0]
  v = vertices[1]
  w = vertices[2]
  return np.linalg.norm(np.cross( v-u, w-u)) / 2.
    
def tetra_volume(vertices):
  u = vertices[0]
  v = vertices[1]
  w = vertices[2]
  x = vertices[3]
  return np.cross(v-u, w-u).dot(x-u) / 6. 

                  

class Container(object):
  
  def __init__(self, master = None, sets = {}):
     self.master = master
     for tag, labels in sets.iteritems(): self.add_set(tag, labels)
  
  def __repr__(self): return self.data.__repr__()
  
  
  def add_set_by_labels(self, tag, labels):
     df = self.data
     tag = "set_" + tag
     #labels = np.array(list(set(labels)))
     df[tag] = np.zeros(len(self.data), dtype = np.bool)
     df.ix[labels, tag] = True
  
  def add_set(self, tag, value):
     df = self.data
     df[tag] = value
     
     
    
     
class Nodes(Container):
  def __init__(self, labels, coords, *args, **kwargs):
    self.data = pd.DataFrame(coords, columns = list("xyz"), index = labels)        
    Container.__init__(self, *args, **kwargs)  

  def add_set_by_func(self, tag, func):
    df = self.data
    x, y, z = np.array(df.x), np.array(df.y), np.array(df.z)
    labels = np.array(df.index)
    df["set_" + tag] = func(x, y, z, labels)
  
     
    
class Elements(Container):
  def __init__(self, labels, etypes, connectivity, surfaces = None, maxconn = 8, *args, **kwargs):
    maxconn = max( maxconn, max([len(c) for c in connectivity]))
    data = {"etype": etypes}
    for i in range(maxconn): data["n_{0}".format(i)] = []
    for c in connectivity:
      lc = len(c)
      for i in range(maxconn):
        if i >= lc: 
          data["n_{0}".format(i)] .append(0)
        else:
          data["n_{0}".format(i)] .append(c[i])   
    self.data = pd.DataFrame(data, index = labels)
    if surfaces == None: surfaces = {"tag":[], "element":[], "face":[]}
    self.surfaces = pd.DataFrame(surfaces)        
    Container.__init__(self, *args, **kwargs)       
  
  def space(self):
    df = self.data
    return pd.Series([ELEMENTS[e]["space"] for e in df.etype], index = df.index)
  
  def nvert(self):
    df = self.data
    return pd.Series([ELEMENTS[e]["nvert"] for e in df.etype], index = df.index)
  
  def to_faces(self):
    pass
  
  def to_polycollection(self):
    pass
      

def ScalarField(object):
  def __init__(self, labels, time, values, master = None):
    self.data = pd.DataFrame({"label":labels, "time": time, "value": values})
    self.master = master
  
  
class Mesh(object):
  def __repr__(self): return "*Nodes:\n{0}\n*Elements\n{1}".format(
                          str(self.nodes), str(self.elements))
  
  def __init__(self, nlabels, coords, elabels, etypes, connectivity, nsets = {}, esets = {}, surfaces = None, h5path = None):
    self.nodes    = Nodes(    labels = nlabels, coords = coords, sets = nsets, 
                              master = self)
    self.elements = Elements( labels = elabels, connectivity = connectivity, 
                              etypes = etypes, sets = esets, 
                              surfaces = surfaces, master = self)
    self.h5path = h5path


    

def read_msh(path):
  elementMap = { 1:"Line2",
                 2:"Tri3",
                 3:"Quad4",
                 4:"Tetra4",
                 5:"Hexa8",
                 6:"Prism6",
                 7:"Pyra4",
               }
  lines = np.array(open(path, "rb").readlines())
  locs = {}
  nl = len(lines)
  for i in xrange(nl):
    line = lines[i].lower().strip()
    if line.startswith("$"):
      if line.startswith("$end"):
        locs[env].append(i)
      else:
        env = line[1:]  
        locs[env] = [i]
  nodes = pd.read_csv(
          StringIO.StringIO("\n".join(
                   lines[locs["nodes"][0]+2:locs["nodes"][1]])), 
                   sep = " ",
                   names = ["labels", "x", "y", "z"])
  
  elements = {"labels":[], "etype":[], "conn": [], "tags":[]}
  for line in lines[locs["elements"][0]+2:locs["elements"][1]]:
    d = np.array([int(w) for w in line.split()])
    elements["labels"].append( d[0] )
    elements["etype"].append(elementMap[d[1]] )
    elements["tags"].append( d[3: 3+d[2]] ) 
    elements["conn"].append(d[3+d[2]:])
  physicalNames = {}
  for line in lines[locs["physicalnames"][0]+2:locs["physicalnames"][1]]:
    w = line.split()
    physicalNames[int(w[1])] = w[2].replace('"', '')
  sets = {}
  tags = np.array([t[0] for t in elements["tags"]])
  for k in physicalNames.keys():
    sets[physicalNames[k]] = [t == k for t in tags]     
  elements["sets"] = sets  
   
  return Mesh(nlabels = nodes["labels"], 
              coords = np.array(nodes[["x", "y", "z"]]),
              elabels = elements["labels"],
              connectivity = elements["conn"],
              etypes = elements["etype"],
              esets = elements["sets"])
  

"""   
Nn, Ne = 100000, 10000
maxconn = 8
nlabels = np.arange(Nn) + 1
elabels = np.arange(Ne) + 1
coords   = np.random.rand(Nn, 3)
esets = {}
nsets = {"truc":coords[:,0]> .2 , "bidule":coords[:,1]> .5}
#connectivity  = np.random.randint(Nn+1, size = maxconn * Ne).reshape(Ne, maxconn)
connectivity = []
for i in range(Ne):
  connectivity.append(np.random.randint(1, Nn+1, size = np.random.randint(3, maxconn+1)))
etypes = np.random.randint(1, 8, size = Ne)
m = Mesh(nlabels = nlabels, elabels = elabels, etypes = etypes, coords = coords, connectivity = connectivity, esets = esets, nsets = nsets, surfaces = None)
func = lambda x, y, z, labels : x >.5
m.nodes.add_set_by_func("by_func", lambda x, y, z, labels: x >.5)
m.nodes.add_set_by_labels("by_labels", np.random.randint(1, Nn+1, size = Nn / 2))
"""
m2 = read_msh("demo.msh")

#def fonc(data):
out = {k:[] for k in d.keys()}
nconn = len([k for k in out.keys() if k.startswith("n_")])
for i in m2.elements.data.index:
  el = m2.elements.data.loc[i]
  eldata = ELEMENTS[el.etype]
  facesconn = eldata["edges"]
  nv = eldata["nvert"]
  conn = np.array([data["n_{0}".format(i)] for i in xrange(nv)])
  faces = conn[facesconn]
  

  
"""
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect("equal")
plt.plot(m2.nodes.data.x, m2.nodes.data.y, ",k")
plt.grid()
plt.show()
"""
