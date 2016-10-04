from argiope.abq.pypostproc import read_field_report

df = read_field_report("data/S.rpt", 
                       names = ["S11", "S22", "S33", "S12"])
