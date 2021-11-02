from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from pycaret.clustering import *
import matplotlib.pyplot as plt


class PreprocessMixin:
    def fit_transform(self):
        """"""
        var_str = list( self.data.columns )
        features = ([i for i, s in enumerate(var_str) if "pct"   in s] + 
            [i for i, s in enumerate(var_str) if "ppm"   in s])
        self.data = self.data.iloc[:,features[1:]]

        pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=10))])
        self.prepared=pipeline.fit_transform(self.data)


class AutomlMixin:
    def prepare(self):
        """Setup experiment"""
        for item in self.features():
            if self.data[item].dtype == 'int64':
                self.data[item] = self.data[item].astype( 'float64' )

        self.experiment = setup(self.data, normalize = True, 
                   ignore_features = self.ignore_features(),
                   session_id = 123)        

    def listmodels(self):
        """List cluster models"""
        return models()

    def create(self):
        """Create a cluster model"""
        if len(self.name) != 0:
            for item in self.name:
                mdl = create_model(item)
                self.model.append(mdl)

    def label(self):
        """Assign model lables"""
        if type(self.model) != "list":
            for name, model in zip(self.name, self.model):
                self.labels = assign_model(model)
                self.data[name+'_Cluster'] = self.labels['Cluster'].values
    

class ClusterPlotMixin:
    """Cluster plotting methods"""

    def plotmodel(self):
        """This function analyzes the performance of a trained model."""
        plot_model( self.model[self.active], plot=self.plottype )
