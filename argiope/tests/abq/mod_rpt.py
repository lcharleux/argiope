import pandas as pd
# Use in Python 2.x
inpath = "test.rpt"
outpath = "out.csv"

def rewriteReport(path, outpath = None, cols = "auto", header = None, sep = ","):
  lines = open(path).readlines()
  count = 0
  for nl in range(len(lines)):
    if lines[nl].startswith(5*"-"): count += 1
    if count == 2: break
  
  if cols == None: 
    out = ""
  elif cols == "auto":
    c = lines[nl-2].split(" ")
    c = [cc for cc in c if len(cc) != 0]
    out = sep.join([w.strip() for w in c])
  else:
    out = sep.join(header) 
  out += "\n".join([sep.join(line.split()) for line in lines[nl+1:]])  
  if outpath == None:
    return out 
  else:
    open(outpath, "w").write(out)
    
  
t = rewriteReport(inpath, outpath)

df = pd.read_csv(outpath)
