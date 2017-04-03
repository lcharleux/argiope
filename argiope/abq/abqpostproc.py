from abaqus import *
from abaqusConstants import *

#import visualization, xyPlot
import displayGroupOdbToolset as dgo
#import __main__

def get_steps(odb):
  """
  Retrieves the steps keys in an odb object
  """
  return odb.steps.keys()

def get_frames(odb, stepKey):
  """
  Retrieves the number of frames in a step.
  """   
  return len(odb.steps[stepKey].frames)
 

def write_xy_report(odb, path, tags, columns, steps):
  """
  Writes a xy_report based on xy data.
  """
  xyData = [session.XYDataFromHistory(name = columns[i], 
                    odb = odb, 
                    outputVariableName = tags[i],
                    steps = steps) 
            for i in xrange(len(tags))]
  session.xyReportOptions.setValues(numDigits=8, numberFormat=SCIENTIFIC)
  session.writeXYReport(fileName=path, appendMode=OFF, xyData=xyData)
  
  
def write_field_report(odb, path, variable, instance, output_position, 
                       step = -1, frame = -1, sortItem='Node Label'):
  """
  Writes a field report and rewrites it in a cleaner format.
  """
  stepKeys = get_steps(odb)
  step = xrange(len(stepKeys))[step]
  frame = xrange(get_frames(odb, stepKeys[step]))[frame]
  nf = NumberFormat(numDigits=9, 
                    precision=0, 
                    format=SCIENTIFIC)
  session.fieldReportOptions.setValues(
          printTotal=OFF, 
          printMinMax=OFF, 
          numberFormat=nf)
  leaf = dgo.LeafFromPartInstance(
          partInstanceName = instance)
  session.viewports['Viewport: 1'].odbDisplay.displayGroup.replace(leaf=leaf)
  session.writeFieldReport(
          fileName       = path, 
          append         = OFF, 
          sortItem       = sortItem,
          odb            = odb, 
          step           = step, 
          frame          = frame, 
          outputPosition = output_position, 
          variable       = variable)
  lines = open(path).readlines()
  counter = 0
  for i in xrange(len(lines)):
    if lines[i].startswith("--"): 
      counter += 1
    if counter == 2:
      break  
  # DATA
  data = lines[i+1:]
  data = [line.strip() for line in data]
  data = "\n".join([",".join(line.split()) for line in data if len(line) != 0])
  # HEADER
  header = str(output_position).lower() + "," 
  header += ",".join([v[1] for v in variable[0][2]]) + "\n"
  # METADATA
  metadata = (
          ("odb", odb.path), 
          ("instance", instance),
          ("position", output_position),
          ("step_num", step),
          ("step_label", stepKeys[step]),
          ("frame", frame),
          ("frame_value", odb.steps[stepKeys[step]].frames[frame].frameValue)
              )
  out = "*METADATA\n{0}\n*DATA\n{1}".format(
        "\n".join(["{0}={1}".format(k, v) for k, v in metadata]),
        header + data)           
  open(path, "w").write(out)
       
       
          
          

