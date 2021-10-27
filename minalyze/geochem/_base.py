from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from pycaret.clustering import *

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
    def design(self):
        """"""
        for item in self.features():
            if self.data[item].dtype == 'int64':
                self.data[item] = self.data[item].astype( 'float64' )
        
        #items = self.features();
        #self.data.loc[:,self.ignore_features()+items[1:10]]

        self.experiment = setup(self.data, normalize = True, 
                   ignore_features = self.ignore_features(),
                   session_id = 123)        
