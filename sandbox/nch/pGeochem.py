import sys, os

basepath = "/Users/nick/Dropbox/projects/consult/csm/core-scan"
sys.path.append(basepath)

# For prototyping/debugging (reload if changes)
import digitalcore.geochemistry
from importlib import reload
reload(digitalcore.geochemistry)
from digitalcore import GeochemCluster, GeochemClusterExperiment

# Read geochem data and create a custom dataframe 
filepath = os.path.join(basepath,'./data/OOLDEA2_1m_intervals.csv')
this = GeochemCluster.read_csv( filepath )

#this.head()
#this.tail()
print(this.get_features())

this.get_element("U")

#this.debug()


filepath = os.path.join(basepath,'./data/OOLDEA2_1m_intervals.csv')
this = GeochemClusterExperiment.read_csv( filepath )

