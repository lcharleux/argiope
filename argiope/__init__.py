__version__ = "0.4"

try:
    from . import utils, mesh, materials, models, abq
except:
    print("Failed ot import python dedicated part side of Argiope")
