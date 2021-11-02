"""
Geochem

Methods
    Select subset of elements for clustering (automate?)

Visualization
    Label overlay on scatter (elemental cross plot) and downcore plots (look at striplog)
    Dimension reduction plots 
    Summarize the clusters as tables and bar charts 
    Parallel coordinates plot 
    Pair/correlation plot (use seaborn)

Image 

corebreakout/corecolumn RGB 

"""

import re
import pandas as pd
import numpy as np

from IPython.display import display
from .geochem._base import PreprocessMixin, AutomlMixin, ClusterPlotMixin
from .base import PlotMixin

class Geochem(PreprocessMixin, PlotMixin):
    """Custom dataframe for geochem data
    
    data: pandas.DataFrame
        Shape (n_instance, n_features), where n_instance is the number of instances and 
        n_features is the number of features.

    todo: remove prepared attribute 
    
    """

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
        print("Nick")
        #return self.__original

    @staticmethod
    def read_csv( location ):
        """Import geochem data from csv"""
        this = Geochem()
        data = pd.read_csv(location)
        this.data = data
        this.__original = data
        return this

class GeochemML(Geochem, AutomlMixin, ClusterPlotMixin):
    """Custom dataframe for geochem data supporting autoML with PyCaret

    data: pandas.DataFrame
        Training data with shape (n_instance, n_features), where n_instance is the number of instances and 
        n_features is the number of features.
    
    unseen: pandas.DataFrame
        Test data with shape (n_instance, n_features), where n_instance is the number of instances and 
        n_features is the number of features.

    experiment: global variables that can be changed using the ``set_config`` funcion
        Global variables configuring the experiment 
    
    name: str, default = ["kmeans", "kmodes"]
        Array of models for training 

    model: scikit-learn compatible object
        Trained model object

    active: index of active model scikit-learn compatible object
        Active model

    plottype: str, default = 'cluster'
        List of available plots (ID - Name):

        * 'cluster' - Cluster PCA Plot (2d)              
        * 'tsne' - Cluster TSnE (3d)
        * 'elbow' - Elbow Plot 
        * 'silhouette' - Silhouette Plot         
        * 'distance' - Distance Plot   
        * 'distribution' - Distribution Plot
    
    """

    def __init__(self):
        super().__init__() #call superclass constructor
        self.unseen = []
        self.labels = []
        self.experiment = []
        self.name = ["kmeans", "kmodes"]
        self.model = []
        self.active = 0
        self.plottype = "cluster"

    @staticmethod
    def read_csv( location ):
        """Import geochem data from csv"""
        this = GeochemML()
        data = pd.read_csv(location)
        this.data = data
        this.__original = data
        return this