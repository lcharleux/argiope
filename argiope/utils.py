import pickle, gzip, copy

def load(path):
  """
  Loads a file.
  """
  return pickle.load(gzip.open(path))


class Container:
  """
  A container meta class with utilities
  """
  
  def save(self, path):
    """
    Saves the instance into a compressed serialized file.
    """
    pickle.dump(self, gzip.open(path, "w"), 3)
  
  def copy(self):
    """
    Returns a copy of self.
    """
    return copy.deepcopy(self)
  

    
