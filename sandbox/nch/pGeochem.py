
import sys, os

basepath = "/Users/nick/Dropbox/projects/consult/core-scan"
sys.path.append(basepath)

# For prototyping/debugging (reload if changes)
import minalyze._core
from importlib import reload
reload(minalyze._core)
from minalyze import Geochem

# Read geochem data and create a custom dataframe 
filepath = os.path.join(basepath,'./data/OOLDEA2_1m_intervals.csv')
this = Geochem.read_csv( filepath )

#this.head()
#this.tail()
print(this.features())

this.element("U")

this.debug()

