import numpy as np

def save_res(fname, regs, sregs):
    np.savetxt(fname, regs)
    np.savetxt(fname, sregs)

def linestyle2dashes(style):
  if style == "--":
    return (3, 3)
  elif style == ":":
    return (1, 2)
  elif style == "-.":
    return (5,1,1,2)
  elif style == "-.-":
    return (2,1,2,1)
  elif style == "-.--":
    return (.5,.5,1,.5)
  else:
    return (None, None)


alg_labels = {"OracleTS": ("OracleTS",  "cyan",  "--"),
              "TS":       ("TS",        "blue",  ":"),
              "AdaTS":    ("B-metaSRM", "red",   "-"),
              "MisAdaTS": ("MisB-metaSRM", "salmon",   "-.--"),
              'mts':      ('f-metaSRM', "green", "-."),

              "mts-mean": "f-MetaSRM-m",
              "mts-no-cov": "MetaTS-BMLcode",
              "MetaTS": ("MetaTS",  "gray", "-"),
              "AdaTSx": ('AdaTSx',"red", "--"),
              "AdaTSd": "AdaTSd",
              }

# import datetime
from datetime import datetime
from pytz import timezone, utc
def get_pst_time():
    date_format='%m-%d-%Y-%H-%M-%S'
    date = datetime.now(tz=utc)
    date = date.astimezone(timezone('US/Pacific'))
    pstDateTime=date.strftime(date_format)
    return pstDateTime