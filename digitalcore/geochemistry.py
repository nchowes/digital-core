"""
    geochemistry
    ------------

    This module contains utilities to visualize, analyze, and cluster geochemistry/XRF data. 

"""

import re
import pandas as pd
import numpy as np

from IPython.display import display
from .base import PlotMixin

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import AgglomerativeClustering
from pycaret.clustering import *

import matplotlib.pyplot as plt
import seaborn as sns

class PreprocessMixin:
    """TODO"""
    def fit_transform(self):
        """TODO"""
        var_str = list( self.data.columns )
        features = ([i for i, s in enumerate(var_str) if "pct"   in s] + 
            [i for i, s in enumerate(var_str) if "ppm"   in s])
        self.data = self.data.iloc[:,features[1:]]

        pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=10))])
        self.prepared=pipeline.fit_transform(self.data)


class AutomlMixin:
    """autoML methods"""
    def prepare(self, silent=True):
        """Setup experiment"""
        for item in self.get_features():
            if self.data[item].dtype == 'int64':
                self.data[item] = self.data[item].astype( 'float64' )

        #merge default constraints with user defined constaints
        constraints = {
            "session_id": 123,
            "normalize": True, 
            "pca": False
            }
        constraints.update( self.dataopts )
        self.dataopts = constraints

        self.experiment = setup(self.data, 
            ignore_features = self.get_ignorefeatures(),
            silent = silent,
            **self.dataopts 
            )        

    def add(self, name, modelopts={}):
        """_summary_

        Parameters
        ----------
        name : _type_
            _description_
        modelopts : dict, optional
            _description_, by default {}
        """
        self.name.append( name )
        self.modelopts.append( modelopts )

    def create(self):
        """Create a cluster model"""

        if len(self.name) != 0:
            
            #reset 
            self.model = []

            for i,item in enumerate(self.name):       

                if item == "hclust": 
                    # Connectivity constraint for hierarchical cluster 
                    depth_constraint = self.data.from_m.values.reshape(-1,1)
                    connect = kneighbors_graph(depth_constraint, n_neighbors=2, include_self=False)

                    constraints = { "connectivity": connect, 
                        "affinity": 'euclidean', 
                        "linkage": 'ward', 
                        "compute_full_tree": True,
                                }

                    if isinstance(self.modelopts[i], dict):
                        self.modelopts[i].update( constraints )
                    else:
                        self.modelopts[i] = constraints

                if len(self.modelopts[i]) == 0:
                    mdl = create_model( item )
                else:
                    mdl = create_model(item, **self.modelopts[i])
                self.model.append(mdl)


    def label(self):
        """Assign model labels"""
        if type(self.model) != "list":
            for i, (name, model) in enumerate(zip(self.name, self.model)):
                trial = name + str(i)
                print(trial)
                self.labels = assign_model(model)
                self.data[trial+'_Cluster'] = self.labels['Cluster'].values
    
    def run(self):
        """Prepare, fit, and label dataset"""
        self.prepare()
        self.create()
        self.label()

    def aggregate(self, output="stack"):
        """Aggregate by cluster"""

        features_and_label = self.get_features() + [self.get_activemodel()]

        df = self.data.loc[:,features_and_label]\
            .groupby([self.get_activemodel()])\
            .agg(["median"])\
            .reset_index()

        df.columns = [' '.join(col).strip() for col in df.columns.values]
        df = df.T

        names = df.iloc[0,:]
        df = df.drop(df.index[0])
        df.columns = names
        df["type"] = df.index.str.extract(r'(ppm|pct)').values
        df.index.name = "element"
        df = df.sort_values(by=["type","element"])
        df = df.reset_index()

        if output == "stack":
            df = df.reset_index()\
                .melt(id_vars=["element", "type"], value_vars=names, var_name="cluster")\
                .sort_values(["element", "cluster"])\
                .reset_index(drop=True)

        df.columns = df.columns.str.lower()
        return df

    def get_listmodels(self):
        """List cluster models"""
        if self.experiment: 
            return models()
        else:
            return None

    def get_label(self):
        """Return label array for the active model"""
        label_name = self.name[self.active]+str(self.active)+'_Cluster'
        label_array = list( self.data[label_name] )   

        value = [int(sub.split(' ')[1]) for sub in label_array]
        return value
    
  
class ClusterPlotMixin:
    """Cluster plotting methods"""

    def plotmodel(self):
        """Plot cluster performance of a trained model"""
        plot_model( self.model[self.active], plot=self.plottype )


    def plot_aggregates(self, by="feature", type="pct"):
        """Plot features by cluster"""
        
        n = self.data[self.get_activemodel()].nunique()

        if by == "cluster":
            df = self.aggregate( output = "unstack" )

            df.head()

            fig, axes = plt.subplots(1,n, figsize=(18,4), sharey = "all" )
            axes = axes.flatten()

            for i in range(0,n):
                sns.barplot(ax=axes[i], x="element", y=f"cluster {i}", data=df[ df["type"] == type ], palette="Blues_d")
                axes[i].tick_params(axis='x', rotation=90)

        elif by == "feature":
            df = self.aggregate( output = "stack" )
            metrics = self._evaluate_metric( df, type=type )
            filters = metrics.loc[0:2, "element"]

            n = len( filters )

            fig, ax = plt.subplots(1,n, figsize=(18,4) )
            fig.tight_layout(pad=3.5)
            for i,filter in enumerate(filters):
            
                this = df[df["element"].str.contains(filters[i])]

                ax = ax.flatten()
                sns.barplot(ax=ax[i], x="element", y="value", hue="cluster", data=this, palette="tab10")

                ax[i].tick_params(axis='x', rotation=45)

                ax[i].set_title(filter)
                ax[i].set_xlabel("")

        else:
            return None

    def plot_scatter(self, elementX, elementY):
        """Element scatterplot with cluster labels"""

        if any( "pct" in s for s in self.get_element(elementX) ):
            elementX = elementX + "_pct"
        else:
            elementX = elementX + "_ppm"

        if any( "pct" in s for s in self.get_element(elementY) ):
            elementY = elementY + "_pct"
        else:
            elementY = elementY + "_ppm"

        fig, ax = plt.subplots( figsize=(12,8) )
        sns.scatterplot(data=self.data, x=elementX, y=elementY, hue=self.get_activemodel(), 
            palette="tab10");

    @staticmethod
    def _evaluate_metric(df, metric="std", type="pct"):
        """Evaluate element variation across clusters"""

        vars = ["element", "type", "value"]

        metric = df.loc[:, vars]\
                .groupby(["element", "type"])\
                .agg([metric])\
                .reset_index()  
        metric.columns = vars

        if type == "pct":
            metric = metric[metric["type"] == "pct"]\
                .sort_values( by="value", ascending=False ).reset_index(drop=True)
        elif type == "ppm":
            metric = metric[metric["type"] == "ppm"]\
                .sort_values( by="value", ascending=False ).reset_index(drop=True)
        else:    
            metric = metric.sort_values( by=["value", "type"], ascending=False )

        return metric


class GeochemBase(PreprocessMixin, PlotMixin):
    """Base class custom dataframe for geochem data
    
    Parameters
    ----------

    data: pd.DataFrame
        Shape (n_instance, n_features), where n_instance is the number of instances and 
        n_features is the number of features.
    
    """

    def __init__(self, data):
        self.data = data
        self.figure = []
        self._original = None
        
    def head(self):
        """Return the first 5 rows of geochem dataframe"""
        value = self.data.head()
        display(value)

    def tail(self):
        """Return the last 5 rows of geochem dataframe"""
        value = self.data.tail()
        display(value)

    def reset(self):
        """Reset data"""
        self.data = self._original.copy(deep=True)
        self.figure = []

    def get_variables(self):
        """List all variables"""
        value = list( self.data.columns )
        return value

    def get_ignorefeatures(self):
        """List all ignored features"""
        value = [
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

        additional = (
            [s for i, s in enumerate(self.get_variables()) if "mdl" in s] +
            [s for i, s in enumerate(self.get_variables()) if "2SE" in s]
            )

        tF_labeled = self.data.columns\
            .str.contains( "_Cluster" )

        if any( tF_labeled ):
            labeled = list(self.data.columns[tF_labeled])
        else:
            labeled = []
        value = value + additional + labeled

        return value

    def get_features(self):
        """List all features"""
        value = list( 
            set( self.get_variables() ) - set( self.get_ignorefeatures()  )
            )
        return value 

    def get_element(self, item):
        """List the features of an element"""
        ptrn = "^"+item+"_"
        value = [x for x in self.get_variables() if re.search(ptrn, x)]
        return value

    def savefig(self):
        """Save figures"""
        for item in self.figure:
            item.savefig(item.get_label()+".png", dpi=300, transparent=False)

    #@staticmethod
    #def read_csv( location ):
    #    """Import geochem data from csv"""
    #    data = pd.read_csv(location)
    #    this = GeochemBase(data)
    #    this._original = data.copy(deep=True)
    #    return this


class GeochemCluster(GeochemBase):
    """Cluster geochemistry data using depth constrained hierarchical clustering. 

    Parameters
    ----------
    GeochemBase : geochemistry.GeochemBase
        Base class custom dataframe for geochem data
    """
    def __init__(self,data):
        super().__init__(data) #call superclass constructor
        self.model = []
        self.num_clusters = 3
        self._prepared = [] 

    def reset(self):
        super().reset()
        self.model = []
        self.num_clusters = 3
        self._prepared = [] 
        
    def prepare(self):
        """Prepare data"""
        for item in self.get_features():
            if self.data[item].dtype == 'int64':
                self.data[item] = self.data[item].astype( 'float64' )
        
        data = self.data.copy(deep=True)
        data = data.drop( columns=self.get_ignorefeatures() )

        pipeline = Pipeline([ ('scaling', StandardScaler()) ])
        #('pca', PCA(n_components=10)
        X = pipeline.fit_transform(data)

        self._prepared = pd.DataFrame(X, columns=data.columns)
        
    def create(self):
        if len(self._prepared):
            # Connectivity constraint for hierarchical cluster 
            depth_constraint = self.data.from_m.values.reshape(-1,1)
            connect = kneighbors_graph(depth_constraint, n_neighbors=2, include_self=False)

            # Fit model
            self.model = AgglomerativeClustering(n_clusters=self.num_clusters, 
                connectivity=connect, 
                affinity='euclidean', 
                linkage='ward', 
                compute_full_tree=True)  

    def label(self):
        if self.model:
            self.data["hclust_Cluster"] = self.model.fit_predict(self._prepared) 
            self.data["hclust_Cluster"] = "Cluster " + self.data["hclust_Cluster"].astype("str")

    def get_label(self):
        """Return label array for the active model"""
        label_name = "hclust_Cluster"
        label_array = list( self.data[label_name] )   

        value = [int(sub.split(' ')[1]) for sub in label_array]
        return value

    def get_activemodel(self):
        """Current active model for plotting and visualization"""
        value = "hclust_Cluster"
        return value

    @staticmethod
    def read_csv( location ):
        """Import geochem data from csv"""
        data = pd.read_csv(location)
        this = GeochemCluster( data )
        this._original = data.copy(deep=True)
        return this


class GeochemClusterExperiment(GeochemBase, AutomlMixin, ClusterPlotMixin):
    """Clustering experiments for geochemistry data using PyCaret autoML 

    data: pandas.DataFrame
        Training data with shape (n_instance, n_features), where n_instance is the number of instances and 
        n_features is the number of features.

    experiment: global variables that can be changed using the ``set_config`` funcion
        Global variables configuring the experiment 
    
    name: str, default = []
        Array of models for training 

    model: scikit-learn compatible object, default = TODO
        Trained model object

    active: index of active model (scikit-learn compatible object)
        Active model

    plottype: str, default = 'cluster'
        List of available plots (ID - Name):

        * 'cluster' - Cluster PCA Plot (2d)              
        * 'tsne' - Cluster TSnE (3d)
        * 'elbow' - Elbow Plot 
        * 'silhouette' - Silhouette Plot         
        * 'distance' - Distance Plot   
        * 'distribution' - Distribution Plot

    dataopts: dict, data configuration options

    modelopts: dict, model configuration options

    """

    def __init__(self, data):
        """Geochem autoML class"""
        super().__init__(data) #call superclass constructor
        self.labels = []
        self.experiment = []
        self.name = []
        self.model = []
        self.active = 0
        self.plottype = "cluster"
        self.dataopts = []
        self.modelopts = []

    def reset(self):
        """Reset experiment"""
        super().reset()
        self.labels = []
        self.experiment = []
        self.name = []
        self.model = []
        self.active = 0
        self.plottype = "cluster"
        self.modelopts = []
        self.dataopts = dict()

    def reset_trials(self):
        """Reset trials"""
        self.labels = []
        self.name = []
        self.model = []
        self.modelopts = []
        self.active = 0

    def get_activemodel(self):
        """Current active model for plotting and visualization"""
        value = self.name[self.active]+str(self.active)+"_Cluster"
        return value

    @staticmethod
    def read_csv( location ):
        """Import geochem data from csv"""
        data = pd.read_csv(location)
        this = GeochemClusterExperiment( data )
        this._original = data.copy(deep=True)
        return this