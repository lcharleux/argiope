import pickle, gzip

def load(path):
  """
  Loads a file.
  """
  return pickle.load(gzip.open(path))

class Container:
  def save(self, path):
    """
    Saves the instance into a compressed serialized file.
    """
    pickle.dump(self, gzip.open(path, "w"), 3)

  

    
