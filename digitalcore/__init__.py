"""
    digital-core
    ------------
    
    Machine learning for digital core. 
    
"""

__version__ = '0.1.0'

from .geochemistry import GeochemCluster, GeochemClusterExperiment
from .hyperspectral import MineralMap

__all__ = ['GeochemCluster', 'GeochemClusterExperiment', 'MineralMap']

