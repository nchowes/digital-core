import os,sys
import pandas as pd
import pytest


sys.path.append(
    os.path.join( os.path.dirname( os.path.abspath(__file__)) , "..")
)

from digitalcore import GeochemCluster, GeochemClusterExperiment

#Unit tests for geochemistry module 
class TestGeochemCluster:
    def test_classmethod(self):
        location =os.path.join( 
            os.path.dirname( os.path.abspath(__file__) ), 
               "..",
               "data", 
               "OOLDEA2_1m_intervals.csv"
         )

        o = GeochemCluster.read_csv( location )
        assert isinstance(o.data, pd.DataFrame)
        assert [] == o.figure
        assert o.data.equals(o._original)


class TestGeochemClusterExperiment:
    def test_classmethod(self):
        location =os.path.join( 
            os.path.dirname( os.path.abspath(__file__) ), 
               "..",
               "data", 
               "OOLDEA2_1m_intervals.csv"
         )

        o = GeochemClusterExperiment.read_csv( location )
        assert isinstance(o.data, pd.DataFrame)
        assert [] == o.figure
        assert o.data.equals(o._original)