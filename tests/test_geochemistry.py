import os,sys
import pandas as pd
import pytest


sys.path.append(
    os.path.join( os.path.dirname( os.path.abspath(__file__)) , "..")
)

from digitalcore import Geochem, GeochemML

#Unit tests for geochemistry module 
class TestGeochem:
    def test_classmethod(self):
        location =os.path.join( 
            os.path.dirname( os.path.abspath(__file__) ), 
               "..",
               "data", 
               "OOLDEA2_1m_intervals.csv"
         )

        o = Geochem.read_csv( location )
        assert isinstance(o.data, pd.DataFrame)
        assert [] == o.figure
        assert o.data.equals(o._original)


class TestGeochemML:
    def test_classmethod(self):
        location =os.path.join( 
            os.path.dirname( os.path.abspath(__file__) ), 
               "..",
               "data", 
               "OOLDEA2_1m_intervals.csv"
         )

        o = GeochemML.read_csv( location )
        assert isinstance(o.data, pd.DataFrame)
        assert [] == o.figure
        assert o.data.equals(o._original)