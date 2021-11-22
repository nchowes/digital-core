from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neighbors import kneighbors_graph
from pycaret.clustering import *
import matplotlib.pyplot as plt
import seaborn as sns

class PreprocessMixin:
    """"""
    def fit_transform(self):
        """"""
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

        self.experiment = setup(self.data, normalize = True, 
                   ignore_features = self.get_ignorefeatures(),
                   session_id = 123, 
                   silent = silent, 
                   pca = False)        

    def create(self):
        """Create a cluster model"""

        if len(self.name) != 0:

            #Store global model options 
            global_model_options = self.modelopts

            for item in self.name:       

                if item == "hclust": 
                    # Connectivity constraint for hierarchical cluster 
                    depth_constraint = self.data.from_m.values.reshape(-1,1)
                    connect = kneighbors_graph(depth_constraint, n_neighbors=2, include_self=False)

                    constraints = { "connectivity": connect, 
                        "affinity": 'euclidean', 
                        "linkage": 'ward', 
                        "compute_full_tree": True,
                                }

                    if isinstance(self.modelopts, dict):
                        self.modelopts.update( constraints )
                    else:
                        self.modelopts = constraints

                if len(self.modelopts) == 0:
                    mdl = create_model( item )
                else:
                    mdl = create_model(item, **self.modelopts)
                self.model.append(mdl)

                #Restore global model options 
                self.modelopts = global_model_options

    def label(self):
        """Assign model labels"""
        if type(self.model) != "list":
            for name, model in zip(self.name, self.model):
                self.labels = assign_model(model)
                self.data[name+'_Cluster'] = self.labels['Cluster'].values
    
    def run(self):
        """Prepare, fit, and label dataset"""
        self.prepare()
        self.create()
        self.label()

    def aggregate(self, output="stack"):
        """Aggregate by cluster"""
        df = self.data.loc[:,self.get_features()]\
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
        label_name = self.name[self.active]+'_Cluster'
        label_array = list( self.data[label_name] )   

        value = [int(sub.split(' ')[1]) for sub in label_array]

        return value
    
  
class ClusterPlotMixin:
    """Cluster plotting methods"""

    def plotmodel(self):
        """Plot cluster performance of a trained model."""
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