from abaqus import *
from abaqusConstants import *
import displayGroupOdbToolset as dgo


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
  
  
def write_field_report(odb, path, label, argiope_class, variable, instance, output_position, 
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
  lines = [line.strip() for line in open(path).readlines()]
  isdata = -1
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
  # HEADER
  header = str(output_position).lower() + "," 
  header += ",".join([v[1] for v in variable[0][2]]) + "\n"
  # METADATA
  metadata = (
          ("label", label),
          ("argiope_class", argiope_class) ,
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
       
       
          
          

