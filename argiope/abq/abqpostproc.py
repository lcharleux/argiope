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
  
  
def write_field_report(odb, path, variable, instance, output_position, step = -1, frame = -1, sortItem='Node Label'):
  """
  Writes a field report.
  """
  stepKeys = get_steps(odb)
  step = xrange(len(stepKeys))[step]
  frame = xrange(get_frames(odb, stepKeys[step]))[frame]
  print frame, step
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

