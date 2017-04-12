path = "abaqus.rpt"


lines = [line.strip() for line in open(path).readlines()]
"""
counter = 0
for i in xrange(len(lines)):
  if lines[i].startswith("--"): 
    counter += 1
  if counter == 2:
    break  
# DATA
"""
isdata = -1
#data = lines[i+1:]
data = []
for line in lines:
 if isdata == 1:
   if len(line) == 0: 
     isdata -= 1
   else:
     data.append(line)   
 elif isdata < 1:
   if line.startswith("--"):
     isdata += 1
data = "\n".join([",".join(line.split()) for line in data if len(line) != 0])

