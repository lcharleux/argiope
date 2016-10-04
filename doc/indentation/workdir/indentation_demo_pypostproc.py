import pandas as pd
import numpy as np

histPath    = "indentation_demo_hist.rpt"
contactPath = "indentation_demo_contact.rpt"
fieldPath   = "indentation_demo_fields.rpt"

#HISTORY OUTPUTS
hist = pd.read_csv(histPath, delim_whitespace = True,)
X = np.array(hist.X)
step = np.zeros_like(X).astype(np.int32)

for i in xrange(1, len(X)):
  step[i] = step[i-1]  
  if X[i] == X[i-1]:
    step[i] += 1
hist.to_csv(histPath[:-4] + ".csv")

