import matplotlib.pyplot as plt

class PlotMixin:
    """Plotting methods"""

    def plot_element(self, element, labels=False):
        """Plot all geochem series for an element"""

        items = self.get_element( element )

        if labels:
            fig_name = "geochem-"+element+"-labeled-series"
        else:
            fig_name = "geochem-"+element+"-series"

        fig, ax = plt.subplots(1, len(items), figsize=(12,8), num=fig_name)
        self.figure.append(fig)

        if labels == True:
            for i, item in enumerate(items):
                self.plot_label( ax[i], item)  
                if i == 0:
                    ax[i].set_ylabel("Depth [m]")
                ax[i].set_xlabel(item)
         
        else:
            for i, item in enumerate(items):
                self.plot( ax[i], item )
                if i == 0:
                     ax[i].set_ylabel("Depth [m]")
                ax[i].set_xlabel(item)
      

    def plot(self, ax, item):
       """Plot a geochem series"""
       x = self.data[item]
       y = self.data.from_m + 0.5
       ax.plot(x,y)
       ax.invert_yaxis()


    def plot_label(self, ax, item):
        """Plot a geochem series with cluster labels"""

        active_name = self.name[self.active]+"_Cluster"

        labels = self.get_label()
        label_colors = plt.get_cmap('tab10') 
        
        df = self.data
        last_row = len(df)

        x = []
        y = []
        for index, row in df.iterrows():
            if index == last_row-1:
                if len(x) != 0:
                    x.append(row[item])
                    y.append(row.from_m)
                    cmap = label_colors(labels[index])[:3]
                    ax.plot(x,y,color=cmap,linewidth=3)
                break

            next_row = df.iloc[index+1]
            current_facies = row[active_name]
            next_facies = next_row[active_name]
            cmap = label_colors(labels[index])[:3]

            x.append(row[item])
            y.append(row.from_m)
            if current_facies != next_facies:
                x.append(next_row[item])
                y.append(next_row.from_m)
                ax.plot(x,y,color=cmap,linewidth=3)
                x = []
                y = []
                
        ax.invert_yaxis()
