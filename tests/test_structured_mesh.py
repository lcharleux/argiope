import unittest
import numpy as np
import argiope as ag

# MESH SETUP
import numpy as np
import argiope as ag

mesh = ag.mesh.structured_mesh(shape = (2,2,2), dim = (1,2,3))

################################################################################
# TESTING
################################################################################

class MyTest(unittest.TestCase):
  
  def setUp(self):
    return
  
  def test_connectivity(self):
    """
    Tests if connectivity is correct.
    """
    c =  np.array([[ 1,  2,  5,  4, 10, 11, 14, 13],
                   [ 2,  3,  6,  5, 11, 12, 15, 14],
                   [ 4,  5,  8,  7, 13, 14, 17, 16],
                   [ 5,  6,  9,  8, 14, 15, 18, 17],
                   [10, 11, 14, 13, 19, 20, 23, 22],
                   [11, 12, 15, 14, 20, 21, 24, 23],
                   [13, 14, 17, 16, 22, 23, 26, 25],
                   [14, 15, 18, 17, 23, 24, 27, 26]])
    self.assertTrue(
        (mesh.elements.conn.values == c).all() )
  
  def test_coords(self):
    """
    Tests if coordinates are correct.
    """
    c = np.array([
                 [ 0. ,  0. ,  0. ],
                 [ 0.5,  0. ,  0. ],
                 [ 1. ,  0. ,  0. ],
                 [ 0. ,  1. ,  0. ],
                 [ 0.5,  1. ,  0. ],
                 [ 1. ,  1. ,  0. ],
                 [ 0. ,  2. ,  0. ],
                 [ 0.5,  2. ,  0. ],
                 [ 1. ,  2. ,  0. ],
                 [ 0. ,  0. ,  1.5],
                 [ 0.5,  0. ,  1.5],
                 [ 1. ,  0. ,  1.5],
                 [ 0. ,  1. ,  1.5],
                 [ 0.5,  1. ,  1.5],
                 [ 1. ,  1. ,  1.5],
                 [ 0. ,  2. ,  1.5],
                 [ 0.5,  2. ,  1.5],
                 [ 1. ,  2. ,  1.5],
                 [ 0. ,  0. ,  3. ],
                 [ 0.5,  0. ,  3. ],
                 [ 1. ,  0. ,  3. ],
                 [ 0. ,  1. ,  3. ],
                 [ 0.5,  1. ,  3. ],
                 [ 1. ,  1. ,  3. ],
                 [ 0. ,  2. ,  3. ],
                 [ 0.5,  2. ,  3. ],
                 [ 1. ,  2. ,  3. ]])
    self.assertTrue(
        (mesh.nodes.coords.values == c).all() )





if __name__ == '__main__':
    unittest.main()      

