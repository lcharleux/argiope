import unittest
import numpy as np
import argiope as ag
from mesh_example import *

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
    vin = np.array([1., .5])
    self.assertEqual( 
       ((vin -vout)**2).sum(), 0 ) 
    
if __name__ == '__main__':
    unittest.main()      
