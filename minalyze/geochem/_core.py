import re
import pandas as pd
import numpy as np

from IPython.display import display
from ._base import PreprocessMixin, AutomlMixin
from ..base import PlotMixin

class Geochem(PreprocessMixin, PlotMixin):
    """Custom dataframe for geochem data"""

    def __init__(self):
        self.data = []
        self.prepared = []
        self.__original = []

    def head(self):
        """Return the first 5 rows of geochem dataframe"""
        if len(self.data) != 0:
           this = self.data.head()
           display(this)

    def tail(self):
        """Return the last 5 rows of geochem dataframe"""
        if len(self.data) != 0:
           this = self.data.tail()
           display(this)

    def reset(self):
        """Reset data"""
        self.data = self.__original
        self.prepared = []

    def variables(self):
        """List all variables"""
        this = list( self.data.columns )
        return this

    def ignore_features(self):
        """List all ignored features"""
        this = [
            'id', 
            'result_master_id',
            'DDH_name',
            'from_m',
            'to_m',
            'Sample_Length_m',
            'Scan_Length_m',
            'Scan_Recovery_pct',
            'Comp(c/s)', 
            'Rayl(c/s)', 
            'LT(secs)',
            'minaloggerlink']
        return this

    def features(self):
        """List all features"""
        this = list( 
            set( self.variables() ) - set( self.ignore_features()  )
            )
        return this 

    def element(self, item):
        """List the features of an element"""
        ptrn = "^"+item+"_"
        this = [x for x in self.variables() if re.search(ptrn, x)]
        return this

    def debug(self):
        """Temporary method for dev/prototyping"""
        return self.__original

    @staticmethod
    def read_csv( location ):
        """Import geochem data from csv"""
        this = Geochem()
        data = pd.read_csv(location)
        this.data = data
        this.__original = data
        return this

class GeochemML(Geochem, AutomlMixin):
    """Custom dataframe for geochem data"""

    def __init__(self):
        super().__init__() #call superclass constructor
        self.experiment = []
    
    @staticmethod
    def read_csv( location ):
        """Import geochem data from csv"""
        this = GeochemML()
        data = pd.read_csv(location)
        this.data = data
        this.__original = data
        return this