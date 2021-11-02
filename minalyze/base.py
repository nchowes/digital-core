import matplotlib.pyplot as plt

class PlotMixin:

    def preview(self, element):
        """Plot all geochem series for an element"""

        items = self.element( element )

        fig, ax = plt.subplots(1, len(items), figsize=(12,8))
        for i, item in enumerate(items):
            self.plot( ax[i], item )
            

    def plot(self, ax, item):
       """Plot a geochem series"""
       x = self.data[item]
       y = self.data.from_m + 0.5
       ax.plot(x,y)
       ax.set_xlabel(item)
       ax.set_ylabel("Depth [m]")
       ax.invert_yaxis()
