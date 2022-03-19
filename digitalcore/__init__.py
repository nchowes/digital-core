"""
    digital-core
    ------------
    
    Machine learning for digital core. 
    See XXXX for complete documentation
"""

__version__ = '0.1.0'

from .geochemistry import Geochem, GeochemML
from .hyperspectral import MineralMap

__all__ = ['Geochem', 'GeochemML', 'MineralMap']

