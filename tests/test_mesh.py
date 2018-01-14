import unittest
import numpy as np
import argiope as ag

# MESH SETUP
import numpy as np
import argiope as ag

################################################################################
# NODE COORDINATES
################################################################################

coords =  np.array([[0., 0., 0.], #1
                    [1., 0., 0.], #2 
                    [2., 0., 0.], #3
                    [1., 1., 0.], #4
                    [0., 1., 0.], #5
                    [0., 0., 1.], #6
                    [1., 0., 1.], #7 
                    [2., 0., 1.], #8
                    [1., 1., 1.], #9
                    [0., 1., 1.], #10
                    [0., 0., 2.], #11
                    [1., 0., 2.], #12
                    [2., 0., 2.], #13
                    [1., 1., 2.], #14
                    [0., 1., 2.], #15
                    [1., 0., 3.], #16
                    ])

# NODE LABELS
nlabels = np.arange(len(coords)) +1                    

# NODE SETS
nsets = {"nset1": nlabels > 2}              

# CONNECTIVITY : 
# Warning = nothing, only used to ensure renctangularity of the table.
conn =  [[1, 2, 4, 5, 0, 0, 0, 0], #1 = QUAD4
         [2, 3, 4, 0, 0, 0, 0, 0], #2 = TRI3
         [6, 7, 9, 10, 11, 12, 14, 15], # 3 = HEXA8
         [7, 8, 9, 12, 13, 14, 0, 0], # 4 = PRISM6
         [12, 13, 14, 16, 0, 0, 0, 0], # 5 = TETRA4
        ]

elabels = np.arange(1, len(conn) + 1)

types =  np.array(["quad4", "tri3", "hexa8", "prism6", "tetra4"])         

stypes = np.array(["CPS4", "CAX3", "C3D8", 
                   "C3D6", "C3D4"]) # Abaqus element naming convention.

esets = {"eset1": elabels < 2}       

materials = np.array(["mat1", "mat2", "mat2", "mat2", "mat2"])

mesh = ag.mesh.Mesh(nlabels = nlabels,
                    coords  = coords,
                    nsets   = nsets,
                    conn    = conn,
                    elabels = elabels,
                    esets   = esets,
                    types   = types,
                    stypes  = stypes,
                    materials = materials
                    )
fields = {}
################################################################################
# TESTING
################################################################################

class MyTest(unittest.TestCase):
  
  def setUp(self):
    return
  
  def test_coords(self):
    """
    Tests if coords are respected.
    """
    self.assertEqual( 
       ((mesh.nodes.coords.values -coords)**2).sum(), 0 )
  
  def test_nlabels(self):
    """
    Tests if node labels are respected.
    """
    self.assertEqual( 
       ((mesh.nodes.index.values - nlabels)**2).sum(), 0 )   
       
  def test_nsets(self):
    """
    Tests if node sets are respected.
    """
    for k, v in nsets.items():
      self.assertEqual( 
       (v ^ mesh.nodes.sets[k].values).sum(), 0 )    
       
  
  def test_conn(self):
    """
    Tests if element connectivity is respected.
    """
    iconn = np.array(conn).flatten()
    mconn = mesh.elements.conn.values
    self.assertEqual( 
       ((mesh.elements.conn.values -conn)**2).sum(), 0 ) 
  
  def test_volumes(self):
    """
    Tests if volume computations are correct.
    """     
    vout = mesh.centroids_and_volumes().volume.values
    vin = np.array([1., .5, 1., .5, 1./6.])
    self.assertTrue( 
       ((vin - vout)**2).sum() < 1.e-10 ) 
  
  def test_centroids(self):
    """
    Tests if centroid computations are correct.
    """     
    vout = mesh.centroids_and_volumes().centroid.values
    vin = np.array(
      [[ 0.5       ,  0.5       ,  0.        ],
       [ 1.33333333,  0.33333333,  0.        ],
       [ 0.5       ,  0.5       ,  1.5       ],
       [ 1.33333333,  0.33333333,  1.5       ],
       [ 1.25      ,  0.25      ,  2.25      ]])
    self.assertTrue( 
       ((vin -vout)**2).sum() < 1.e-10) 
  
       
  def test_angles(self):
    """
    Tests if volume computations are correct.
    """     
    vout = mesh.angles().angles.stack().values
    vin = np.array(
      [ 90.,  90.,  90.,  90.,  90.,  45.,  45.,  90.,  90.,  90.,  90.,
        90.,  90.,  90.,  90.,  90.,  90.,  90.,  90.,  90.,  90.,  90.,
        90.,  90.,  90.,  90.,  90.,  90.,  90.,  90.,  90.,  90.,  45.,
        45.,  90.,  45.,  45.,  90.,  90.,  90.,  90.,  90.,  90.,  90.,
        90.,  90.,  90.,  90.,  90.,  90.,  45.,  45.,  90.,  45.,  45.,
        60.,  60.,  60.,  45.,  45.,  90.])
    self.assertTrue( 
       ((vin -vout)**2).sum() < 1.e-10)      

  def test_edges(self):
    """
    Tests if edges computations are correct.
    """     
    vout = mesh.edges().length.stack().values
    vin = np.array(
      [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.41421356,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.        ,
        1.41421356,  1.        ,  1.        ,  1.41421356,  1.        ,
        1.        ,  1.        ,  1.        ,  1.        ,  1.41421356,
        1.41421356,  1.        ])

    self.assertTrue( 
       ((vin -vout)**2).sum() < 1.e15, 0 )      



    
if __name__ == '__main__':
    unittest.main()      
